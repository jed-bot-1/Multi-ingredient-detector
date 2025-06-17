from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import shutil
import glob
import gc
import os

app = FastAPI()
model = YOLO('best.onnx')

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        # OpenCV object counting
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 500
        object_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        num_objects = len(object_contours)

        # Adjust YOLO confidence based on object count
        yolo_conf = 0.5 if num_objects == 1 else 0.002

        # YOLO detection
        results = model(img, imgsz=640, conf=yolo_conf)
        detected_ingredients = set()
        for result in results:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names
            for cls in classes:
                detected_ingredients.add(names[cls])

        return {"detected_ingredients": sorted(detected_ingredients)}

    finally:
        # Clean up memory and temporary files
        gc.collect()

        tmp_paths = glob.glob("/tmp/ultralytics*")
        for path in tmp_paths:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
            except Exception as e:
                print(f"[Cleanup error] Failed to delete {path}: {e}")
