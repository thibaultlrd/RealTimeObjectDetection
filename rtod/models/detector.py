from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from rtod.utils.postprocess import nms
from rtod.utils.logging_setup import logger
import tensorflow as tf

@dataclass
class Detection:
	bbox_xyxy: Tuple[int, int, int, int]
	score: float
	class_id: int
	class_name: str
	color: Tuple[int, int, int]


class TFLiteYoloDetector:
	def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45, class_names: Optional[List[str]] = None, color_map: Optional[List[Tuple[int, int, int]]] = None, class_starting_idx: Optional[int] = 4) -> None:
		self.conf_threshold = conf_threshold
		self.iou_threshold = iou_threshold

		if not Path(model_path).exists():
			raise FileNotFoundError(f"Model file not found: {model_path}")

		self.interpreter = tf.lite.Interpreter(model_path=model_path)
		logger.info(f"Loaded TFLite model from {model_path}")
		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()		
		self.class_starting_idx = class_starting_idx
		self.input_size = self._detect_input_size()
		self.num_classes = self._detect_num_classes()
		self.class_names = class_names
		self._set_color_map(color_map)
		if self.class_names is None:
			self.class_names = [f"class_{i}" for i in range(self.num_classes)]

	def _detect_input_size(self) -> Tuple[int, int]:
		"""Detect input size from the loaded TFLite model."""
		if self.input_details is None:
			raise RuntimeError("No input details available from model")
		
		# Get input shape from model (typically [batch, height, width, channels])
		input_shape = self.input_details[0]['shape']
		logger.debug(f"Model input shape: {input_shape}")
		
		if len(input_shape) == 4:
			# Standard format: [batch, height, width, channels]
			height, width = input_shape[1], input_shape[2]
		elif len(input_shape) == 3:
			# Format without batch dimension: [height, width, channels]
			height, width = input_shape[0], input_shape[1]
		else:
			raise ValueError(f"Unexpected input shape format: {input_shape}")
		
		# Validate and return input size
		if height > 0 and width > 0:
			logger.info(f"Detected model input size: ({height}, {width})")
			return (height, width)
		else:
			raise ValueError(f"Invalid detected input size: ({height}, {width})")

	def _detect_num_classes(self) -> int:
		"""Detect number of classes from the loaded TFLite model."""
		if self.output_details is None:
			raise RuntimeError("No output details available from model")
		num_classes = self.output_details[0]['shape'][1] - self.class_starting_idx
		logger.info(f"Detected number of classes: {num_classes}")
		return num_classes

	def _set_color_map(self, color_map: Optional[List[Tuple[int, int, int]]]) -> None:
		if color_map is None:
			self.color_map = self._generate_deterministic_colors(self.num_classes)
		else:
			self.color_map = color_map
	
	def _generate_deterministic_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
		"""Generate deterministic colors using HSV color space for better distribution."""
		colors = []
		
		# Use HSV color space for better color distribution
		# Hue values distributed evenly across the color wheel
		for i in range(num_classes):
			hue = (i * 360 // num_classes) % 360
			saturation = 255
			value = 255
			
			# Convert HSV to RGB
			import colorsys
			r, g, b = colorsys.hsv_to_rgb(hue / 360.0, saturation / 255.0, value / 255.0)
			colors.append((int(r * 255), int(g * 255), int(b * 255)))
		
		return colors

	def _infer_tflite(self, input_tensor: np.ndarray) -> List[np.ndarray]:
		assert self.interpreter is not None
		assert self.input_details is not None and self.output_details is not None

		# Set input
		input_index = self.input_details[0]["index"]
		self.interpreter.set_tensor(input_index, input_tensor)
		self.interpreter.invoke()

		# Collect outputs (assume single output or common YOLO export with boxes/classes)
		outputs: List[np.ndarray] = []
		for od in self.output_details:
			outputs.append(self.interpreter.get_tensor(od["index"]))
		return outputs

	def letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], float]:
		"""
		Resize and pad image while maintaining aspect ratio.
		
		Args:
			img: Input image
			new_shape: Target shape (height, width)
		
		Returns:
			Resized and padded image, padding offsets (top, left), and scale ratio
		"""
		new_shape = self.input_size
		shape = img.shape[:2]  # current shape [height, width]
		# Calculate scale ratio (new / old)
		r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
		# Calculate new unpadded dimensions
		new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
		# Calculate padding
		dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
		# Resize
		if shape[::-1] != new_unpad:  # if needed
			img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
		# Add padding
		top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
		left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
		img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
		
		return img, (top, left), r

	def preprocess(self, input_array: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], float]:
		"""
		preprocessing with better letterboxing.
		
		Args:
			input_array: Input image
		
		Returns:
			Preprocessed image, padding offsets (top, left), and scale factor
		"""
		# Convert greyscale image to RGB if needed
		if len(input_array.shape) == 2:
			input_array = cv2.cvtColor(input_array, cv2.COLOR_GRAY2RGB)
		
		# Resize and pad image using the dynamically detected input size
		img, pad_offsets, scale = self.letterbox(input_array)
		
		img = img.astype(np.float32) / 255.0
		# Add batch dimension and ensure contiguous memory
		if len(img.shape) == 3:
			img = np.expand_dims(img, axis=0)
		img = np.ascontiguousarray(img)
		
		return img, pad_offsets, scale


	def decode_yolo_output(self, outputs: List[np.ndarray], img_shape: Tuple[int, int], pad_offsets: Tuple[float, float], scale: float) -> List[Detection]:
		"""
		YOLO output decoding with proper coordinate transformation.
		
		Args:
			outputs: Model outputs
			img_shape: Original image shape (height, width)
			pad_offsets: Padding offsets (top, left) in pixels
			scale: Scale factor used in preprocessing
		
		Returns:
			List of Detection objects
		"""
		# Find the main prediction output
		pred = None
		for out in outputs:
			if out.ndim == 3 and out.shape[0] == 1:
				pred = out[0]  # Remove batch dimension
				break
		
		if pred is None:
			logger.warning("No valid prediction output found")
			return []
		
		logger.debug(f"Prediction shape: {pred.shape}")
		
		# Convert to tensor for easier processing
		pred_tensor = tf.constant(pred)
		
		# Transpose if needed (YOLO outputs are often [batch, features, detections])
		if pred_tensor.shape[1] > pred_tensor.shape[0]:
			pred_tensor = tf.transpose(pred_tensor)  # Shape: [detections, features]
		
		logger.debug(f"Transposed prediction shape: {pred_tensor.shape}")
		
		# Extract box coordinates and class predictions
		# Expected format: [x, y, w, h, class1_prob, class2_prob, ...]
		if pred_tensor.shape[1] != (4 + self.num_classes):
			logger.error(f"Unexpected prediction shape: {pred_tensor.shape}, expected {4 + self.num_classes} features")
			return []
		
		boxes_xywh = pred_tensor[:, :4]
		class_probs = pred_tensor[:, 4:]
		
		# Get the class with highest probability and its score
		class_scores = tf.reduce_max(class_probs, axis=1, keepdims=True)
		class_ids = tf.argmax(class_probs, axis=1)
		scores = tf.squeeze(class_scores)
		
		# Filter by confidence threshold
		valid_detections = scores > self.conf_threshold
		
		if not tf.reduce_any(valid_detections):
			logger.debug("No detections above confidence threshold")
			return []
		
		filtered_boxes = tf.boolean_mask(boxes_xywh, valid_detections)
		filtered_scores = tf.boolean_mask(scores, valid_detections)
		filtered_class_ids = tf.boolean_mask(class_ids, valid_detections)
		
		logger.debug(f"Filtered detections: {tf.shape(filtered_boxes)[0]}")
		
		# Convert boxes from center format to corner format and scale to image coordinates
		detections = []
		img_h, img_w = img_shape
		pad_top, pad_left = pad_offsets
		
		for i in range(tf.shape(filtered_boxes)[0]):
			box = filtered_boxes[i].numpy()
			score = filtered_scores[i].numpy()
			class_id = int(filtered_class_ids[i].numpy())
			
			# YOLO outputs are typically normalized [0, 1]
			x_center, y_center, width, height = box
			
			# Convert to pixel coordinates relative to input size
			x_center_px = x_center * self.input_size[1]  # width
			y_center_px = y_center * self.input_size[0]  # height
			width_px = width * self.input_size[1]
			height_px = height * self.input_size[0]
			
			# Remove padding offsets to get coordinates on the scaled image
			x_center_scaled = x_center_px - pad_left
			y_center_scaled = y_center_px - pad_top
			
			# Convert to corner coordinates on the scaled image
			x1_scaled = x_center_scaled - width_px / 2
			y1_scaled = y_center_scaled - height_px / 2
			x2_scaled = x_center_scaled + width_px / 2
			y2_scaled = y_center_scaled + height_px / 2
			
			# Scale back to original image coordinates
			x1 = int(x1_scaled / scale)
			y1 = int(y1_scaled / scale)
			x2 = int(x2_scaled / scale)
			y2 = int(y2_scaled / scale)
			
			# Clamp to image boundaries
			x1 = max(0, min(x1, img_w - 1))
			y1 = max(0, min(y1, img_h - 1))
			x2 = max(x1 + 1, min(x2, img_w))
			y2 = max(y1 + 1, min(y2, img_h))
			
			detections.append(Detection(
				bbox_xyxy=(x1, y1, x2, y2),
				score=float(score),
				class_id=class_id,
				class_name=self.class_names[class_id],
				color=self.color_map[class_id % len(self.color_map)]
			))
		
		# Apply NMS
		if len(detections) > 1:
			boxes_array = np.array([det.bbox_xyxy for det in detections], dtype=np.float32)
			scores_array = np.array([det.score for det in detections], dtype=np.float32)
			
			keep_indices = nms(boxes_array, scores_array, self.iou_threshold)
			detections = [detections[i] for i in keep_indices]
		
		logger.debug(f"Final detections after NMS: {len(detections)}")
		return detections

	def predict(self, image_bgr: np.ndarray) -> List[Detection]:
		logger.debug(f"Input image shape: {image_bgr.shape}")
		
		inp, pad_offsets, scale = self.preprocess(image_bgr)
		logger.debug(f"Preprocessed input shape: {inp.shape}, scale: {scale}, pad_offsets: {pad_offsets}")
		
		# Run inference
		outputs = self._infer_tflite(inp)
		logger.debug(f"Model outputs: {[out.shape for out in outputs]}")
		
		# Use decoding
		detections = self.decode_yolo_output(outputs, image_bgr.shape[:2], pad_offsets, scale)

		
		logger.info(f"Found {len(detections)} detections")
		return detections