from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import gc
import os
import psutil
import logging
from functools import lru_cache
import asyncio

# === CONFIG ===
os.environ["YOLO_CONFIG_DIR"] = "/tmp"  # Prevent write issues on Render
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detector")

# Limit concurrent YOLO requests (Render free tier is weak)
semaphore = asyncio.Semaphore(2)

# === Load model once and cache ===
@lru_cache(maxsize=1)
def get_model():
    return YOLO("best.onnx", task="detect")

model = get_model()


async def run_detection(file: UploadFile):
    contents = None
    nparr = None
    img = None
    results = None
    try:
        logger.info("⏳ Received image, starting detection...")

        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning("🚫 Invalid image format.")
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        if img.shape[:2] != (640, 640):
            img = cv2.resize(img, (640, 640))  # YOLOv8 expects fixed input size

        # Estimate object count for adaptive confidence
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        object_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
        num_objects = len(object_contours)
        logger.info(f"🧠 Estimated object count: {num_objects}")

        # YOLO inference (lower conf if many objects)
        conf = 0.5 if num_objects <= 1 else 0.25
        results = model(img, imgsz=640, conf=conf)

        detected = {
            result.names[int(cls)]
            for result in results
            for cls in result.boxes.cls.cpu().numpy().astype(int)
        }

        logger.info(f"✅ Detected ingredients: {sorted(detected)}")
        return {"detected_ingredients": sorted(detected)}

    except Exception as e:
        logger.error(f"❌ Detection error: {e}")
        return JSONResponse({"error": "Detection failed"}, status_code=500)

    finally:
        # Close uploaded file
        try:
            file.file.close()
        except Exception as e:
            logger.warning(f"File close error: {e}")

        # Clean memory
        del contents, nparr, img, results
        gc.collect()

        # Log memory usage
        try:
            mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            logger.info(f"📉 Memory after cleanup: {mem:.2f} MB")
        except Exception as e:
            logger.warning(f"Memory logging failed: {e}")


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    async with semaphore:  # Limit concurrent YOLO calls
        return await run_detection(file)
