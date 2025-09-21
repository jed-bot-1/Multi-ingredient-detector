from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2 as cv
import numpy as np 
import io
from PIL import Image 
import os
import tempfile
import gc

app = FastAPI()

# Load model
model = YOLO("best.onnx")
print("Model Loaded Successfully")

def resize_to_640(img):
    """Resize image to 640x640 maintaining aspect ratio"""
    height, width = img.shape[:2]
    scale = 640 / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)
    return resized, scale

def detect_objects_simple(img):
    """Simplified detection that works with 640x640 optimized model"""
    # Resize to 640x640 for optimal model performance
    resized_img, scale = resize_to_640(img)
    
    # Run detection on the resized image
    results = model.predict(
        resized_img, 
        conf=0.25, 
        iou=0.45, 
        verbose=False,
        imgsz=resized_img.shape[:2]  # Use actual resized dimensions
    )
    
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Convert coordinates back to original image scale
        abs_x1 = int(x1 / scale)
        abs_y1 = int(y1 / scale)
        abs_x2 = int(x2 / scale)
        abs_y2 = int(y2 / scale)

        detections.append({
            "class": model.names[cls_id],
            "confidence": conf,
            "bbox": [abs_x1, abs_y1, abs_x2, abs_y2]
        })
    
    return detections

@app.get("/check")
def root():
    return {"message": "Service is up and running!"}

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        # Read uploaded file into OpenCV image
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv.imdecode(np_arr, cv.IMREAD_COLOR)
        
        # Detect objects with simplified approach
        detections = detect_objects_simple(img)
        ingredients = list({det["class"] for det in detections})

        # Force cleanup
        del contents, np_arr, img
        gc.collect()

        return {
            "ingredients": ingredients,
            "detections": detections  # Include full detection info for debugging
        }

    except Exception as e:
        # Cleanup on error
        if 'contents' in locals():
            del contents
        if 'np_arr' in locals():
            del np_arr
        if 'img' in locals():
            del img
        gc.collect()
        
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.middleware("http")
async def cleanup_after_request(request, call_next):
    try:
        response = await call_next(request)
        return response
    finally:
        gc.collect()
