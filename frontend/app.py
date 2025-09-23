import io
import os
from typing import Optional

import cv2
import numpy as np
import requests
import streamlit as st

st.set_page_config(page_title="RealTimeObjectDetection", layout="wide")

# Prefer environment variable, then optional secrets, then default
BACKEND_URL = os.getenv("BACKEND_URL") or os.getenv("STREAMLIT_BACKEND_URL")
if not BACKEND_URL:
	try:
		BACKEND_URL = st.secrets["backend_url"]
	except Exception:
		BACKEND_URL = "http://localhost:8000"

st.title("Real-Time Object Detection")


def draw_boxes(image_bgr: np.ndarray, detections: list) -> np.ndarray:
	for det in detections:
		x1, y1, x2, y2 = det["bbox"]
		cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
		label = f"{det['class_name']}:{det['score']:.2f}"
		cv2.putText(image_bgr, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
	return image_bgr


mode = st.radio("Mode", ["Upload Image", "Camera"], horizontal=True)

image_to_process: Optional[np.ndarray] = None

if mode == "Upload Image":
	uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
	if uploaded:
		bytes_data = uploaded.read()
		arr = np.frombuffer(bytes_data, dtype=np.uint8)
		image_to_process = cv2.imdecode(arr, cv2.IMREAD_COLOR)
elif mode == "Camera":
	camera = st.camera_input("Take a picture")
	if camera is not None:
		bytes_data = camera.getvalue()
		arr = np.frombuffer(bytes_data, dtype=np.uint8)
		image_to_process = cv2.imdecode(arr, cv2.IMREAD_COLOR)


if image_to_process is not None:
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
else:
	st.info("Upload an image or take a picture to run detection.")
