from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import onnxruntime as ort
import cv2
import numpy as np
import os

app = FastAPI()

MODEL_PATH = "best-quant.onnx"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found.")

# Class names from your dataset.yaml
CLASS_NAMES = [
    "Bitter_m", "Calamansi", "Eggplant", "Fish_Sauce", "Garlic", "Ginger",
    "Okra", "Onion", "Pepper", "Pork", "Potato", "Salt",
    "Soy_Sauce", "Sqaush", "Tomato", "Vinegar"
]

# Load ONNX model using ONNX Runtime
ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # --- OpenCV object counting logic (unchanged) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 500
    object_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    num_objects = len(object_contours)

    yolo_conf = 0.5 if num_objects == 1 else 0.002

    try:
        # --- Preprocess image for YOLO (NCHW format) ---
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape  # [1, 3, H, W]
        img_resized = cv2.resize(img, (input_shape[3], input_shape[2]))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))[None, :]  # NHWC to NCHW

        # --- Run inference ---
        outputs = ort_session.run(None, {input_name: img_input})

        # --- Post-processing (assumes output shape [n, 85]) ---
        detections = outputs[0]
        detected_ingredients = set()
        for det in detections:
            x1, y1, x2, y2, obj_conf, *class_scores = det
            conf = obj_conf * max(class_scores)
            if conf > yolo_conf:
                cls_id = int(np.argmax(class_scores))
                if cls_id < len(CLASS_NAMES):
                    detected_ingredients.add(CLASS_NAMES[cls_id])

        return {"detected_ingredients": sorted(detected_ingredients)}
    except Exception as e:
        return JSONResponse({"error": f"ONNX inference failed: {e}"}, status_code=500)
