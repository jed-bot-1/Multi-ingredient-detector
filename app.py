from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import asyncio
import gc
import os
from PIL import Image
import io

# === ULTRA-LIGHTWEIGHT CONFIG ===
os.environ.update({
    "YOLO_CONFIG_DIR": "/tmp",
    "YOLO_VERBOSE": "False", 
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1"
})

# Minimal FastAPI app
app = FastAPI()

# Single global model instance
_model = None

async def get_model():
    """Load model only when needed"""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO('best.onnx', task='detect')
        _model.overrides['verbose'] = False
    return _model

def detect_lightweight(img_bytes: bytes) -> dict:
    """Ultra-lightweight detection"""
    try:
        # Simple image processing
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if img.size != (640, 640):
            img = img.resize((640, 640))
        
        # Convert to numpy
        import numpy as np
        img_array = np.array(img)
        img.close()
        
        # YOLO inference - high confidence to reduce processing
        results = _model(img_array, conf=0.5, verbose=False, device='cpu')
        
        # Extract classes
        detected = set()
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for cls in result.boxes.cls.cpu().numpy().astype(int):
                    detected.add(result.names[cls])
        
        return {"detected_ingredients": sorted(detected)}
    
    except Exception as e:
        return {"error": f"Detection failed: {str(e)[:30]}"}
    
    finally:
        # Clean up variables
        if 'img' in locals():
            del img
        if 'img_array' in locals():
            del img_array
        if 'results' in locals():
            del results
        gc.collect()

@app.get("/")
def root():
    """Simple root"""
    return {"status": "ok"}

@app.get("/health/")  
def health():
    """Health check"""
    return {"status": "healthy", "model_loaded": _model is not None}

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    """Main detection endpoint"""
    try:
        # Size limit for memory
        if file.size and file.size > 3 * 1024 * 1024:  # 3MB limit
            return JSONResponse({"error": "File too large"}, status_code=413)
        
        # Read file
        content = await file.read()
        
        # Quick validation
        if len(content) < 50:
            return JSONResponse({"error": "Invalid file"}, status_code=400)
        
        # Load model
        await get_model()
        
        # Process with timeout
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, detect_lightweight, content),
            timeout=20
        )
        
        return result if "error" not in result else JSONResponse(result, status_code=500)
        
    except asyncio.TimeoutError:
        return JSONResponse({"error": "Timeout"}, status_code=408)
    except Exception as e:
        return JSONResponse({"error": "Failed"}, status_code=500)
    finally:
        gc.collect()
