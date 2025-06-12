from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import cv2
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_with_flex.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, c, h, w = input_details[0]['shape']

CLASSES = [
    "Bitter_m", "Calamansi", "Eggplant", "Fish_Sauce", "Garlic", "Ginger",
    "Okra", "Onion", "Pepper", "Pork", "Potato", "Salt", "Soy_Sauce",
    "Sqaush", "Tomato", "Vinegar"
]
NUM_CLASSES = len(CLASSES)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (w, h))
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)
    return image

def detect_object_count(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 500
    return len([cnt for cnt in contours if cv2.contourArea(cnt) > min_area])

def extract_probabilities(output: np.ndarray) -> np.ndarray:
    flat = np.ravel(output)
    if len(flat) < NUM_CLASSES:
        raise ValueError("Model output is smaller than expected.")
    return flat[:NUM_CLASSES]

def get_predicted_labels(probs: np.ndarray, threshold: float) -> list:
    return [CLASSES[i] for i, p in enumerate(probs) if i < NUM_CLASSES and p >= threshold]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format"})

        object_count = detect_object_count(image)
        is_single_object = object_count == 1
        threshold = 0.5 if is_single_object else 0.002

        input_tensor = preprocess_image(image)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        probs = extract_probabilities(output)

        if is_single_object:
            top_idx = int(np.argmax(probs))
            ingredients = CLASSES[top_idx]
        else:
            ingredients = get_predicted_labels(probs, threshold)

        return JSONResponse(content={"ingredients": ingredients})

    except Exception as e:
        print("ERROR:", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
