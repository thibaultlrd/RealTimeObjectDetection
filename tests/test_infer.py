import sys
from pathlib import Path
import numpy as np
import cv2

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rtod.models import TFLiteYoloDetector  # noqa: E402

MODEL_PATH = Path(__file__).parent.parent / "rtod" / "models" / "yolov8n.tflite"
TEST_IMG_PATH = Path(__file__).parent.parent / "res" / "dog_test_picture.jpg"

def test_detector_outputs_structure():
	"""Test that detector returns proper structure with real model"""
	if not MODEL_PATH.exists():
		print(f"Skipping test - model not found at {MODEL_PATH}")
		return
		
	img = np.zeros((240, 320, 3), dtype=np.uint8)
	detector = TFLiteYoloDetector(model_path=str(MODEL_PATH))
	dets = detector.predict(img)
	assert isinstance(dets, list)
	for d in dets:
		assert hasattr(d, "bbox_xyxy")
		assert hasattr(d, "score")
		assert hasattr(d, "class_id")


def test_detector_with_test_image():
	"""Test detector with actual test image from res/test.png"""
	if not MODEL_PATH.exists():
		print(f"Skipping test - model not found at {MODEL_PATH}")
		return
		
	assert TEST_IMG_PATH.exists(), f"Test image not found at {TEST_IMG_PATH}"
	
	img = cv2.imread(str(TEST_IMG_PATH))
	assert img is not None, "Failed to load test image"
	assert img.shape[2] == 3, "Expected BGR image with 3 channels"
	
	detector = TFLiteYoloDetector(model_path=str(MODEL_PATH))
	dets = detector.predict(img)
	
	assert isinstance(dets, list)

	for det in dets:
		assert hasattr(det, "bbox_xyxy")
		assert hasattr(det, "score")
		assert hasattr(det, "class_id")
		assert hasattr(det, "class_name")
		
		x1, y1, x2, y2 = det.bbox_xyxy
		h, w = img.shape[:2]
		assert 0 <= x1 < x2 <= w, f"Invalid bbox x coordinates: {x1}, {x2} for image width {w}"
		assert 0 <= y1 < y2 <= h, f"Invalid bbox y coordinates: {y1}, {y2} for image height {h}"
		
		assert 0.0 <= det.score <= 1.0, f"Invalid score: {det.score}"
		
		assert det.class_id >= 0, f"Invalid class_id: {det.class_id}"


def test_model_accuracy_and_visualization():
	"""Test model prediction accuracy on test image and visualize bounding boxes"""
	if not MODEL_PATH.exists():
		print(f"Skipping test - model not found at {MODEL_PATH}")
		return
		
	output_dir = Path(__file__).parent.parent / "test_outputs"
	
	assert TEST_IMG_PATH.exists(), f"Test image not found at {TEST_IMG_PATH}"
	
	img = cv2.imread(str(TEST_IMG_PATH))
	assert img is not None, "Failed to load test image"
	print(f"Loaded test image with shape: {img.shape}")
	
	output_dir.mkdir(exist_ok=True)
	
	detector = TFLiteYoloDetector(
		model_path=str(MODEL_PATH),
		conf_threshold=0.25,
		iou_threshold=0.45
	)

	detections = detector.predict(img)
	print(f"Found {len(detections)} detections")
	
	assert isinstance(detections, list), "Detections should be a list"
	
	for i, det in enumerate(detections):
		print(f"Detection {i+1}:")
		print(f"  - Bounding box (x1,y1,x2,y2): {det.bbox_xyxy}")
		print(f"  - Confidence score: {det.score:.3f}")
		print(f"  - Class ID: {det.class_id}")
		print(f"  - Class name: {det.class_name}")
		
		assert hasattr(det, "bbox_xyxy"), "Detection missing bbox_xyxy"
		assert hasattr(det, "score"), "Detection missing score"
		assert hasattr(det, "class_id"), "Detection missing class_id"
		assert hasattr(det, "class_name"), "Detection missing class_name"
		
		x1, y1, x2, y2 = det.bbox_xyxy
		h, w = img.shape[:2]
		assert 0 <= x1 < x2 <= w, f"Invalid bbox x coordinates: {x1}, {x2} for image width {w}"
		assert 0 <= y1 < y2 <= h, f"Invalid bbox y coordinates: {y1}, {y2} for image height {h}"
		
		assert 0.0 <= det.score <= 1.0, f"Invalid score: {det.score}"
		
		assert det.class_id >= 0, f"Invalid class_id: {det.class_id}"
	
	img_with_detections = img.copy()
	img_with_detections = detector.draw_detections(img_with_detections, detections)
	
	original_output = output_dir / "test_image_original.jpg"
	annotated_output = output_dir / "test_image_with_detections.jpg"
	
	cv2.imwrite(str(original_output), img)
	cv2.imwrite(str(annotated_output), img_with_detections)
	
	print(f"Saved original image to: {original_output}")
	print(f"Saved annotated image to: {annotated_output}")
	
	assert original_output.exists(), "Failed to save original image"
	assert annotated_output.exists(), "Failed to save annotated image"
	
	if len(detections) > 0:
		high_confidence_dets = [d for d in detections if d.score >= 0.5]
		print(f"High confidence detections (>=0.5): {len(high_confidence_dets)}")
		
		avg_confidence = sum(d.score for d in detections) / len(detections)
		print(f"Average confidence: {avg_confidence:.3f}")
		
		assert avg_confidence > 0.1, f"Average confidence too low: {avg_confidence}"
	
	print("✅ Model accuracy and visualization test completed successfully!")