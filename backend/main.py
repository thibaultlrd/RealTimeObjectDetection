from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse

from rtod.models import TFLiteYoloDetector, Detection
from rtod.utils.logging_setup import logger

app = FastAPI(title="RealTimeObjectDetection")

# Create API router
api_router = APIRouter(prefix="/api")

MODEL_PATH = str(Path("rtod/models") / "yolov8n.tflite")

if not Path(MODEL_PATH).exists():
	raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

_detector = TFLiteYoloDetector(model_path=MODEL_PATH)


@api_router.get("/health")
async def health() -> dict:
	input_size = [int(x) for x in _detector.input_size]
	return {"status": "ok", "model_path": MODEL_PATH, "input_size": input_size}


@api_router.post("/predict")
async def predict(image: UploadFile = File(...)) -> JSONResponse:
	logger.info(f"Received prediction request for image: {image.filename}")
	data = await image.read()
	arr = np.frombuffer(data, dtype=np.uint8)
	img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
	if img is None:
		logger.error("Failed to decode image")
		return JSONResponse(status_code=400, content={"error": "Invalid image"})

	logger.info(f"Processing image with shape: {img.shape}")
	dets: List[Detection] = _detector.predict(img)
	logger.info(f"Found {len(dets)} detections")
	
	for i, det in enumerate(dets):
		logger.info(f"Detection {i+1}: bbox={det.bbox_xyxy}, score={det.score:.3f}, class={det.class_name}")
	
	resp = [
		{
			"bbox": det.bbox_xyxy,
			"score": det.score,
			"class_id": det.class_id,
			"class_name": det.class_name,
		}
		for det in dets
	]
	return JSONResponse(content={"detections": resp})

# Include the API router
app.include_router(api_router)
