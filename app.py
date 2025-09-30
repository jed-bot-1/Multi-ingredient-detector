from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import asyncio
import gc
import os
from PIL import Image
import io
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# === ULTRA-LIGHTWEIGHT CONFIG ===
os.environ.update({
    "YOLO_CONFIG_DIR": "/tmp",
    "YOLO_VERBOSE": "False",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1"
})

# Minimal FastAPI app
app = FastAPI()

# Global model + executor
_model = None
executor = ThreadPoolExecutor(max_workers=2)  # safe concurrency limit

# Lazy model loader
async def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("best.onnx", task="detect")
        _model.overrides["verbose"] = False
    return _model


def detect_lightweight(img_bytes: bytes) -> dict:
    """Ultra-lightweight detection with enforced 640x640 resizing + cleanup"""
    img, img_array, results = None, None, None
    try:
        # Open + preprocess
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((640, 640))

        # Convert to numpy
        img_array = np.array(img, dtype=np.uint8)
        img.close()  # close handle

        # Run YOLO (CPU only for Render free tier)
        results = _model(img_array, conf=0.5, verbose=False, device="cpu")

        detected = set()
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for cls in result.boxes.cls.cpu().numpy().astype(int):
                    detected.add(result.names[cls])

        return {"detected_ingredients": sorted(detected)}

    except Exception as e:
        return {"error": f"Detection failed: {str(e)[:50]}"}

    finally:
        # ✅ Cleanup
        if img is not None:
            del img
        if img_array is not None:
            del img_array
        if results is not None:
            del results
        gc.collect()


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health/")
def health():
    return {"status": "healthy", "model_loaded": _model is not None}


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        # Read file
        content = await file.read()

        # Validate file size
        if len(content) < 100 or len(content) > 3 * 1024 * 1024:
            return JSONResponse({"error": "Invalid file"}, status_code=400)

        # Load model if needed
        await get_model()

        # Run inference in background thread with timeout
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(executor, detect_lightweight, content),
            timeout=20
        )

        return result if "error" not in result else JSONResponse(result, status_code=500)

    except asyncio.TimeoutError:
        return JSONResponse({"error": "Timeout"}, status_code=408)
    except Exception:
        return JSONResponse({"error": "Failed"}, status_code=500)
    finally:
        gc.collect()
