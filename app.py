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

def read_image(uploaded_file: UploadFile):
    """Convert uploaded file to OpenCV image (BGR)."""
    contents = uploaded_file.file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = np.array(pil_img)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return img

def resize_image_maintain_aspect(img, target_size=640):
    """Resize image to target size while maintaining aspect ratio"""
    height, width = img.shape[:2]
    
    # Calculate scaling factor
    scale = target_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized_img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)
    
    # Create a 640x640 canvas with black background
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Calculate padding
    y_offset = (target_size - new_height) // 2
    x_offset = (target_size - new_width) // 2
    
    # Place resized image on canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img
    
    return canvas, scale, (x_offset, y_offset)

def detect_objects(img):
    # Resize to 640x640 for optimal model performance
    resized_img, scale, (x_offset, y_offset) = resize_image_maintain_aspect(img)
    original_resized = resized_img.copy()

    # Using Cv to detect contours on resized image
    gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv.dilate(thresh, kernel, iterations=2)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    detected_any = False
    detected_coords = []
    detections = []

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if w < 30 or h < 30:
            continue

        pad = max(15, min(w, h) // 5)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(resized_img.shape[1], x + w + pad)
        y2 = min(resized_img.shape[0], y + h + pad)

        roi = original_resized[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Run YOLO on ROI - use 640x640 for optimal performance
        results = model.predict(roi, conf=0.25, iou=0.2, verbose=False)
        boxes = results[0].boxes

        if boxes is not None:
            for box in boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Convert coordinates back to original image scale
                abs_x1 = int((x1 + bx1 - x_offset) / scale)
                abs_y1 = int((y1 + by1 - y_offset) / scale)
                abs_x2 = int((x1 + bx2 - x_offset) / scale)
                abs_y2 = int((y1 + by2 - y_offset) / scale)

                # Prevent duplicate detections
                is_duplicate = any(
                    abs(abs_x1 - px1) < 20 and abs(abs_y1 - py1) < 20 and
                    abs(abs_x2 - px2) < 20 and abs(abs_y2 - py2) < 20
                    for px1, py1, px2, py2 in detected_coords
                )
                if is_duplicate:
                    continue

                label = model.names[cls_id]
                detections.append({
                    "class": label,
                    "confidence": conf,
                    "bbox": [abs_x1, abs_y1, abs_x2, abs_y2]
                })

                detected_any = True
                detected_coords.append((abs_x1, abs_y1, abs_x2, abs_y2))

    # --- Step 3: Fallback if no detections ---
    if not detected_any:
        # Run YOLO on full resized image (640x640 optimized)
        results = model.predict(resized_img, conf=0.25, iou=0.45, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Convert coordinates back to original image scale
            abs_x1 = int((x1 - x_offset) / scale)
            abs_y1 = int((y1 - y_offset) / scale)
            abs_x2 = int((x2 - x_offset) / scale)
            abs_y2 = int((y2 - y_offset) / scale)

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
        
        # Detect objects with 640x640 optimized processing
        detections = detect_objects(img)
        ingredients = list({det["class"] for det in detections})

        # Force cleanup
        del contents, np_arr, img
        gc.collect()

        return {"ingredients": ingredients}

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
