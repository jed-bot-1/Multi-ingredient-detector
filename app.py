from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2 as cv
import numpy as np 
import io
from PIL import Image 
import os
import tempfile


app =  FastAPI()

model = YOLO("best.onnx")
print("Model Loaded Successfully")

def read_image(uploaded_file: UploadFile):
    """Convert uploaded file to OpenCV image (BGR)."""
    contents = uploaded_file.file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = np.array(pil_img)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)  # convert to OpenCV BGR
    return img


def detect_objects(img):
    original_img = img.copy()

    # Using Cv to detect contours in detecting image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv.dilate(thresh, kernel, iterations=2)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    detected_any = False
    detection_count = 0
    detected_coords = []
    detections = []  # making sure detections is in json format

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if w < 30 or h < 30:
            continue

        pad = max(15, min(w, h) // 5)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        roi = original_img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Run YOLO on ROI
        results = model.predict(roi, conf=0.25, iou=0.2, verbose=False)
        boxes = results[0].boxes

        if boxes is not None:
            for box in boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                abs_x1 = x1 + bx1
                abs_y1 = y1 + by1
                abs_x2 = x1 + bx2
                abs_y2 = y1 + by2

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
                detection_count += 1
                detected_coords.append((abs_x1, abs_y1, abs_x2, abs_y2))

    # --- Step 3: Fallback if no detections ---
    if not detected_any:
        results = model.predict(original_img, conf=0.25, iou=0.45, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "class": model.names[cls_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
            detection_count += 1

    return detections


@app.get("/check")
def root():
    return {"message": "Service is up and running!"}

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv.imdecode(np_arr, cv.IMREAD_COLOR)

        detections = detect_objects(img)
        ingredients = list({det["class"] for det in detections})

        return {"ingredients": ingredients}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"temporary file {temp_path} successfully deleted")
        else:
            print(f"Error:{temp_path} file not found")
    


