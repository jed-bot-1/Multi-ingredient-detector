from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import gc
import os
import psutil
import logging
from PIL import Image
import io
import tempfile
import shutil

# === RENDER OPTIMIZATION CONFIG ===
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
os.environ["YOLO_VERBOSE"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Reduce logging to prevent memory issues
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("detector")

app = FastAPI(title="Image Detection API", version="1.0.0")

# Global model variable - lazy loaded
_model = None
_model_lock = asyncio.Lock()

def get_memory_mb():
    """Get current memory usage"""
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except:
        return 0

def aggressive_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    gc.collect()  # Call twice
    
    # Clean temp files
    try:
        temp_dirs = ["/tmp", tempfile.gettempdir()]
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                for item in os.listdir(temp_dir):
                    if any(x in item.lower() for x in ["ultralytics", "yolo", "onnx"]):
                        item_path = os.path.join(temp_dir, item)
                        try:
                            if os.path.isfile(item_path):
                                os.remove(item_path)
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                        except:
                            pass
    except:
        pass

async def load_model():
    """Load YOLO model with error handling"""
    global _model
    if _model is None:
        async with _model_lock:
            if _model is None:  # Double-check
                try:
                    # Import here to avoid startup issues
                    from ultralytics import YOLO
                    
                    logger.error("Loading YOLO model...")
                    _model = YOLO('best.onnx', task='detect')
                    _model.overrides['verbose'] = False
                    logger.error("Model loaded successfully")
                    
                    # Log memory after loading
                    memory_mb = get_memory_mb()
                    logger.error(f"Memory after model load: {memory_mb:.1f}MB")
                    
                except Exception as e:
                    logger.error(f"Model loading failed: {e}")
                    raise HTTPException(status_code=500, detail="Model loading failed")
    return _model

def detect_simple(image_bytes: bytes) -> dict:
    """Simplified detection function"""
    try:
        # Check memory before processing
        memory_before = get_memory_mb()
        if memory_before > 450:
            aggressive_cleanup()
        
        # Load and process image
        pil_img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB and resize
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        if pil_img.size != (640, 640):
            pil_img = pil_img.resize((640, 640), Image.Resampling.LANCZOS)
        
        # Convert to array
        import numpy as np
        img_array = np.array(pil_img)
        
        # Close PIL image immediately
        pil_img.close()
        del pil_img
        
        # Run YOLO inference
        results = _model(img_array, imgsz=640, conf=0.25, verbose=False, device='cpu')
        
        # Extract results
        detected = set()
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                if len(result.boxes) > 0:
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    for class_id in class_ids:
                        detected.add(result.names[class_id])
        
        # Cleanup
        del img_array
        del results
        del image_bytes
        
        return {"detected_ingredients": sorted(list(detected))}
    
    except Exception as e:
        logger.error(f"Detection error: {str(e)[:50]}")
        return {"error": "Detection failed"}
    
    finally:
        aggressive_cleanup()

@app.get("/")
async def root():
    """Simple root endpoint"""
    memory_mb = get_memory_mb()
    return {
        "status": "running",
        "memory_mb": round(memory_mb, 1)
    }

@app.get("/health/")
async def health():
    """Health check"""
    memory_mb = get_memory_mb()
    return {
        "status": "healthy",
        "memory_mb": round(memory_mb, 1),
        "model_loaded": _model is not None
    }

@app.post("/detect/")
async def detect_endpoint(file: UploadFile = File(...)):
    """Main detection endpoint"""
    try:
        # Check memory first
        memory_mb = get_memory_mb()
        if memory_mb > 480:
            aggressive_cleanup()
            return JSONResponse({"error": "Memory limit reached"}, status_code=503)
        
        # Validate file size
        if file.size and file.size > 5 * 1024 * 1024:  # 5MB limit
            return JSONResponse({"error": "File too large"}, status_code=413)
        
        # Read file
        file_content = await file.read()
        
        # Validate image
        try:
            Image.open(io.BytesIO(file_content)).verify()
        except:
            return JSONResponse({"error": "Invalid image"}, status_code=400)
        
        # Load model if needed
        await load_model()
        
        # Run detection in a way that doesn't block too long
        try:
            # Add timeout to prevent hanging
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, detect_simple, file_content
                ),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            return JSONResponse({"error": "Detection timeout"}, status_code=408)
        
        if "error" in result:
            return JSONResponse(result, status_code=500)
        
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)[:50]}")
        return JSONResponse({"error": "Processing failed"}, status_code=500)
    
    finally:
        # Final cleanup
        aggressive_cleanup()

@app.on_event("startup")
async def startup():
    """Minimal startup"""
    logger.error("API starting...")
    # Don't preload model - load on first request instead

@app.on_event("shutdown") 
async def shutdown():
    """Cleanup on shutdown"""
    aggressive_cleanup()

# Memory monitoring middleware
@app.middleware("http")
async def monitor_memory(request, call_next):
    """Monitor and limit memory usage"""
    memory_before = get_memory_mb()
    
    # If memory is too high, force cleanup
    if memory_before > 480:
        aggressive_cleanup()
        # If still too high after cleanup, reject request
        if get_memory_mb() > 500:
            return JSONResponse({"error": "Service overloaded"}, status_code=503)
    
    response = await call_next(request)
    
    # Cleanup after request
    memory_after = get_memory_mb()
    if memory_after > 450:
        aggressive_cleanup()
    
    return response
