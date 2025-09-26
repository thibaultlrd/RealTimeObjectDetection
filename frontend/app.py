import io
import os
import threading
import time
from typing import Optional, List, Dict, Any
from queue import Queue, Empty
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import av
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

from rtod.utils.logging_setup import logger

st.set_page_config(page_title="RealTimeObjectDetection", layout="wide")

# Prefer environment variable, then optional secrets, then default
BACKEND_URL = os.getenv("BACKEND_URL") or os.getenv("STREAMLIT_BACKEND_URL")
if not BACKEND_URL:
	try:
		BACKEND_URL = st.secrets["backend_url"]
	except Exception:
		BACKEND_URL = "http://localhost:8000"

st.title("Real-Time Object Detection")

logger.debug(f"Using backend URL: {BACKEND_URL}")

# Initialize session state variables (including queues)
if "detections" not in st.session_state:
	st.session_state.detections = []
if "last_detection_time" not in st.session_state:
	st.session_state.last_detection_time = 0
if "worker_thread" not in st.session_state:
	st.session_state.worker_thread = None
if "detection_queue" not in st.session_state:
	st.session_state.detection_queue = Queue()
if "result_queue" not in st.session_state:
	st.session_state.result_queue = Queue()
if "stop_worker" not in st.session_state:
	st.session_state.stop_worker = threading.Event()

# Create global references that can be accessed by VideoProcessor thread
# These will point to the same queue instances as in session_state
global_detection_queue = st.session_state.detection_queue
global_result_queue = st.session_state.result_queue
global_stop_worker = st.session_state.stop_worker

# WebRTC configuration for better connectivity
RTC_CONFIGURATION = RTCConfiguration(
	{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def draw_boxes(image_bgr: np.ndarray, detections: list) -> np.ndarray:
	"""Draw bounding boxes and labels on image"""
	for det in detections:
		x1, y1, x2, y2 = det["bbox"]
		cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
		label = f"{det['class_name']}:{det['score']:.2f}"
		cv2.putText(image_bgr, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
	return image_bgr


def detect_objects_async(frame: np.ndarray) -> Optional[List[Dict[str, Any]]]:
	"""Send frame to backend for detection (non-blocking)"""
	try:
		logger.debug(f"Sending request to {BACKEND_URL}/predict")
		_, enc = cv2.imencode('.jpg', frame)
		resp = requests.post(
			f"{BACKEND_URL}/predict",
			files={"image": ('frame.jpg', io.BytesIO(enc.tobytes()), 'image/jpeg')},
			timeout=5,  # Reduced timeout for real-time
		)
		logger.debug(f"Backend response: {resp.status_code}")
		if resp.ok:
			data = resp.json()
			detections = data.get("detections", [])
			logger.debug(f"Got {len(detections)} detections from backend")
			return detections
		else:
			logger.debug(f"Backend error: {resp.status_code}")
		return None
	except Exception as e:
		logger.debug(f"Exception in detect_objects_async: {e}")
		return None


def detection_worker(detection_queue, result_queue, stop_event):
	"""Background worker for processing detection requests"""
	logger.debug("Detection worker thread started")
	while not stop_event.is_set():
		try:
			frame = detection_queue.get(timeout=0.1)
			detections = detect_objects_async(frame)
			if detections is not None:
				result_queue.put({
					'detections': detections,
					'timestamp': time.time()
				})
				logger.debug(f"Worker puts {len(detections)} detections in result queue")
		except Empty:
			continue
		except Exception as e:
			logger.error(f"Exception in detection worker: {e}")
			continue


class VideoProcessor(VideoTransformerBase):
	"""Video processor for real-time object detection"""
	
	def __init__(self):
		self.last_frame_time = 0
		self.frame_interval = 1.0 / 5  # Process every 5 frames per second for detection
		self.current_detections = []
		
	def recv(self, frame):
		img = frame.to_ndarray(format="bgr24")
		current_time = time.time()
		
		# Send frame for detection at reduced rate to avoid overwhelming the backend
		if current_time - self.last_frame_time > self.frame_interval:
			if global_detection_queue.qsize() < 2:  # Limit queue size
				logger.debug(f"Adding frame to detection queue. Queue size: {global_detection_queue.qsize()}")  # Debug log
				global_detection_queue.put(img.copy())
			else:
				logger.debug("Detection queue is full, skipping frame")  # Debug log
			self.last_frame_time = current_time
		
		# Check for new detection results
		try:
			while True:
				result = global_result_queue.get_nowait()
				self.current_detections = result['detections']
		except Empty:
			pass
		
		# Always draw the latest detections on the current frame
		if self.current_detections:
			img_with_detections = draw_boxes(img, self.current_detections)
		else:
			img_with_detections = img
			
		return av.VideoFrame.from_ndarray(img_with_detections, format="bgr24")


def process_static_image(image_to_process: np.ndarray):
	"""Process a static image (upload or camera snapshot)"""
	_, enc = cv2.imencode('.jpg', image_to_process)
	try:
		resp = requests.post(
			f"{BACKEND_URL}/predict",
			files={"image": ('frame.jpg', io.BytesIO(enc.tobytes()), 'image/jpeg')},
			timeout=30,
		)
		if resp.ok:
			data = resp.json()
			detections = data.get("detections", [])
			st.write(f"Backend URL: {BACKEND_URL}")
			st.write(f"Found {len(detections)} detections:")
			for i, det in enumerate(detections):
				st.write(f"Detection {i+1}: bbox={det['bbox']}, score={det['score']:.3f}, class={det['class_name']}")
			img = draw_boxes(image_to_process.copy(), detections)
			st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detections")
		else:
			st.error(f"Backend error: {resp.status_code}")
	except requests.exceptions.RequestException as ex:
		st.error(f"Cannot reach backend at {BACKEND_URL}. Start the API (uvicorn) or use docker-compose.\nDetails: {ex}")


# Main UI
mode = st.radio("Mode", ["Upload Image", "Camera Snapshot", "Real-Time Camera"], horizontal=True)

image_to_process: Optional[np.ndarray] = None

if mode == "Upload Image":
	uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
	if uploaded:
		bytes_data = uploaded.read()
		arr = np.frombuffer(bytes_data, dtype=np.uint8)
		image_to_process = cv2.imdecode(arr, cv2.IMREAD_COLOR)

elif mode == "Camera Snapshot":
	camera = st.camera_input("Take a picture")
	if camera is not None:
		bytes_data = camera.getvalue()
		arr = np.frombuffer(bytes_data, dtype=np.uint8)
		image_to_process = cv2.imdecode(arr, cv2.IMREAD_COLOR)

elif mode == "Real-Time Camera":
	st.subheader("Real-Time Object Detection")
	
	# Start detection worker if not already running
	if st.session_state.worker_thread is None or not st.session_state.worker_thread.is_alive():
		st.session_state.stop_worker.clear()
		st.session_state.worker_thread = threading.Thread(
			target=detection_worker,
			args=(st.session_state.detection_queue, st.session_state.result_queue, st.session_state.stop_worker),
			daemon=True
		)
		st.session_state.worker_thread.start()
	
	# Update UI with latest results from the result queue
	# This runs in the main thread, so it's safe to access session state
	try:
		while True:
			result = st.session_state.result_queue.get_nowait()
			st.session_state.detections = result['detections']
			st.session_state.last_detection_time = result['timestamp']
	except Empty:
		pass
	
	# Display real-time stats
	col1, col2 = st.columns(2)
	with col1:
		st.metric("Active Detections", len(st.session_state.detections))
	with col2:
		if st.session_state.last_detection_time > 0:
			time_since_last = time.time() - st.session_state.last_detection_time
			st.metric("Last Detection", f"{time_since_last:.1f}s ago")
		else:
			st.metric("Last Detection", "None")
	
	# WebRTC video stream with real-time detection in a column layout
	col1, col2 = st.columns([2, 1])  # Camera takes 2/3, info takes 1/3
	
	with col1:
		webrtc_ctx = webrtc_streamer(
			key="object-detection",
			video_processor_factory=VideoProcessor,
			rtc_configuration=RTC_CONFIGURATION,
			media_stream_constraints={
				"video": {
					"width": {"ideal": 480, "max": 640},
					"height": {"ideal": 360, "max": 480}
				}, 
				"audio": False
			},
			async_processing=True,
		)
	
	with col2:
		# Display current detections info in the right column
		if st.session_state.detections:
			st.subheader("Current Detections")
			for i, det in enumerate(st.session_state.detections):
				st.write(f"Detection {i+1}: {det['class_name']} (confidence: {det['score']:.3f})")
	
	# Stop button to clean up
	if st.button("Stop Real-Time Detection"):
		st.session_state.stop_worker.set()
		st.session_state.detections = []
		st.session_state.last_detection_time = 0
		st.session_state.worker_thread = None
		st.rerun()

# Process static images
if image_to_process is not None:
	process_static_image(image_to_process)
elif mode in ["Upload Image", "Camera Snapshot"]:
	st.info("Upload an image or take a picture to run detection.")
