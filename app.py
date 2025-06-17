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

# === 🛠️ FIX: Set YOLO config directory ===
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detector")

# === 🛠️ FIX: Explicitly define task ===
model = YOLO('best.onnx', task='detect')

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        logger.info("⏳ Received image, starting detection...")

        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning("🚫 Invalid image format.")
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        img = cv2.resize(img, (640, 640))

        # Object count using contours
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        object_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
        num_objects = len(object_contours)
        logger.info(f"🧠 Object count detected via OpenCV: {num_objects}")

        yolo_conf = 0.5 if num_objects == 1 else 0.002
        results = model(img, imgsz=640, conf=yolo_conf)

        detected_ingredients = set()
        for result in results:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names
            for cls in classes:
                detected_ingredients.add(names[cls])

        logger.info(f"✅ Detected ingredients: {sorted(detected_ingredients)}")

        return {"detected_ingredients": sorted(detected_ingredients)}

    finally:
        logger.info("🧹 Starting cleanup...")

        try:
            file.file.close()
        except Exception as e:
            logger.warning(f"File close error: {e}")

        del file, contents, nparr, img, gray, blur, thresh, contours
        gc.collect()

        # Remove /tmp folders from Ultralytics
        tmp_paths = glob.glob("/tmp/ultralytics*")
        for path in tmp_paths:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    logger.info(f"🗑️ Deleted folder: {path}")
                elif os.path.isfile(path):
                    os.remove(path)
                    logger.info(f"🗑️ Deleted file: {path}")
            except Exception as e:
                logger.warning(f"[Cleanup error] Failed to delete {path}: {e}")

        # Log memory usage
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 * 1024)
        logger.info(f"📉 Memory after cleanup: {mem:.2f} MB")

        # Clear model results
        try:
            model.model = None
            del results
        except Exception as e:
            logger.warning(f"Model cleanup failed: {e}")

        gc.collect()
