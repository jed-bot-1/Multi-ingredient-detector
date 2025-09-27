from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import asyncio
import gc
import os
import psutil
import logging
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil

# === RENDER DEPLOYMENT CONFIG ===
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
os.environ["YOLO_VERBOSE"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"  # Single thread for starter tier
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

app = FastAPI(title="Image Detection API", version="1.0.0")

# Minimal logging for production
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("detector")

# === MEMORY-EFFICIENT MODEL MANAGER ===
class OptimizedModelManager:
    def __init__(self):
        self._model = None
        self._lock = asyncio.Lock()
        self._loading = False
    
    async def get_model(self):
        if self._model is None and not self._loading:
            async with self._lock:
                if self._model is None:  # Double-check pattern
                    self._loading = True
                    try:
                        logger.warning("Loading YOLO model...")
                        # Load with minimal memory footprint
                        self._model = YOLO('best.onnx', task='detect')
                        # Optimize for inference only
                        self._model.overrides['verbose'] = False
                        logger.warning("Model loaded successfully")
                    except Exception as e:
                        logger.error(f"Model loading failed: {e}")
                        raise
                    finally:
                        self._loading = False
        return self._model
    
    def is_loaded(self):
        return self._model is not None

# Global instances
model_manager = OptimizedModelManager()
# Single worker for starter tier
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")

# === MEMORY UTILITIES ===
def force_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    gc.collect()  # Call twice for better cleanup
    
    # Clean YOLO temp files
    try:
        for temp_path in ["/tmp", tempfile.gettempdir()]:
            for item in os.listdir(temp_path):
                if "ultralytics" in item.lower() or "yolo" in item.lower():
                    item_path = os.path.join(temp_path, item)
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except:
                        pass
    except:
        pass

def get_memory_mb():
    """Get current memory usage in MB"""
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except:
        return 0

# === OPTIMIZED DETECTION CORE ===
def detect_ingredients_optimized(image_data: bytes) -> dict:
    """Memory-optimized detection function"""
    pil_image = None
    cv_image = None
    
    try:
        # 1. Load image efficiently with PIL
        pil_image = Image.open(io.BytesIO(image_data))
        
        # 2. Convert and resize in one step
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Resize only if needed
        target_size = (640, 640)
        if pil_image.size != target_size:
            pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # 3. Convert to numpy for YOLO
        cv_image = np.array(pil_image)
        
        # 4. Quick object estimation for confidence tuning
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count significant objects
        significant_objects = sum(1 for c in contours if cv2.contourArea(c) > 400)
        
        # 5. YOLO inference with dynamic confidence
        confidence = 0.45 if significant_objects <= 3 else 0.15
        
        # Get model and run inference
        model = model_manager._model
        results = model(cv_image, imgsz=640, conf=confidence, verbose=False, device='cpu')
        
        # 6. Extract results efficiently
        detected_classes = set()
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                if len(boxes) > 0:
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    for class_id in class_ids:
                        class_name = result.names[class_id]
                        detected_classes.add(class_name)
        
        return {
            "detected_ingredients": sorted(list(detected_classes)),
            "count": len(detected_classes)
        }
    
    except Exception as e:
        logger.error(f"Detection failed: {str(e)[:100]}")
        return {"error": "Detection processing failed"}
    
    finally:
        # Immediate cleanup
        if pil_image:
            pil_image.close()
            del pil_image
        if cv_image is not None:
            del cv_image
        if 'gray' in locals():
            del gray
        if 'binary' in locals():
            del binary
        del image_data
        force_cleanup()

# === API ENDPOINTS ===
@app.post("/detect/")
async def detect_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_content = None
    
    try:
        # Memory check before processing
        current_memory = get_memory_mb()
        if current_memory > 450:  # Leave buffer for 512MB limit
            force_cleanup()
            await asyncio.sleep(0.1)  # Brief pause
        
        # Validate file
        if file.size and file.size > 8 * 1024 * 1024:  # 8MB limit
            return JSONResponse({"error": "Image too large (max 8MB)"}, status_code=413)
        
        # Read file
        file_content = await file.read()
        
        # Quick format validation
        if len(file_content) < 100:  # Too small to be valid image
            return JSONResponse({"error": "Invalid image file"}, status_code=400)
        
        # Ensure model is ready
        await model_manager.get_model()
        
        # Run detection in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            detect_ingredients_optimized, 
            file_content
        )
        
        # Schedule background cleanup
        background_tasks.add_task(cleanup_background)
        
        # Handle errors
        if "error" in result:
            return JSONResponse(result, status_code=500)
        
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"API error: {str(e)[:100]}")
        return JSONResponse({"error": "Processing failed"}, status_code=500)
    
    finally:
        if file_content:
            del file_content
        force_cleanup()

async def cleanup_background():
    """Background cleanup task"""
    await asyncio.sleep(0.5)
    force_cleanup()

@app.get("/health/")
async def health_check():
    """Health and status endpoint"""
    memory_mb = get_memory_mb()
    return {
        "status": "healthy" if memory_mb < 500 else "warning",
        "memory_mb": round(memory_mb, 1),
        "memory_percent": round((memory_mb / 512) * 100, 1),
        "model_loaded": model_manager.is_loaded()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Image Detection API", 
        "status": "running",
        "memory_mb": round(get_memory_mb(), 1)
    }

# === STARTUP/SHUTDOWN HANDLERS ===
@app.on_event("startup")
async def startup_event():
    """Initialize service"""
    try:
        logger.warning("Starting Image Detection API...")
        # Preload model to avoid first-request delay
        await model_manager.get_model()
        initial_memory = get_memory_mb()
        logger.warning(f"Startup complete. Memory: {initial_memory:.1f}MB")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.warning("Shutting down...")
    executor.shutdown(wait=False)
    force_cleanup()

# === MIDDLEWARE FOR MEMORY MONITORING ===
@app.middleware("http")
async def memory_middleware(request, call_next):
    """Monitor memory usage per request"""
    
    # Process request
    response = await call_next(request)
    
    # Check memory after request
    memory_after = get_memory_mb()
    if memory_after > 480:  # Near limit
        force_cleanup()
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 8000)),
        workers=1
    )
