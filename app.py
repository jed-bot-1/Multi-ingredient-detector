from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import shutil
import glob
import gc
import os
import psutil
import logging

# === CONFIG ===
os.environ["YOLO_CONFIG_DIR"] = "/tmp"  # Prevent write issues on Render
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detector")

# === Load model only once ===
model = YOLO('best.onnx', task='detect')

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = None
    try:
        logger.info("⏳ Received image, starting detection...")

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning("🚫 Invalid image format.")
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        if img.shape[:2] != (640, 640):
            img = cv2.resize(img, (640, 640))  # Keep consistent for YOLOv8

        # Optional: Count objects to adjust YOLO confidence
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        object_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
        num_objects = len(object_contours)
        logger.info(f"🧠 Estimated object count: {num_objects}")

        # YOLO inference
        conf = 0.5 if num_objects == 1 else 0.002
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
        logger.info("🧹 Starting cleanup...")
        try:
            file.file.close()
        except Exception as e:
            logger.warning(f"File close error: {e}")

        # Safe cleanup of all local variables
        for var in ['contents', 'nparr', 'img', 'gray', 'blur', 'thresh', 'contours']:
            try:
                del globals()[var]
            except:
                pass

        # Clean Ultralytics temp folders
        for path in glob.glob("/tmp/ultralytics*"):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Failed to delete {path}: {e}")

        # Log memory usage
        try:
            mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            logger.info(f"📉 Memory after cleanup: {mem:.2f} MB")
        except Exception as e:
            logger.warning(f"Memory logging failed: {e}")

        # Force GC
        gc.collect()
