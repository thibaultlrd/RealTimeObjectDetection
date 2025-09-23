# RealTimeObjectDetection

Object detection pipeline that runs inference on webcam/video input in real time with a lightweight model.

## Quickstart (local)

1. Create and activate venv, then install deps:

```bash
pip install -r requirements.txt
```

2. Place your model at `models/model_name.tflite`.

3. Run backend:

```bash
uvicorn backend.main:app --reload --port 8000
```

4. Run frontend (uses BACKEND_URL `http://localhost:8000` by default):

```bash
streamlit run frontend/app.py
```

## Docker

```bash
docker compose up --build
```
- Backend: http://localhost:8000/docs
- Frontend: http://localhost:8501

## Tests

```bash
pytest -q
```

## CI
- GitHub Actions: ruff lint, pytest (mock mode), Docker build for backend and frontend.

## Deployment (free-friendly)
- Backend: Google Cloud Run (container from `Dockerfile.backend`).
- Frontend: Streamlit Community Cloud (set secret `backend_url`).

## License
MIT
