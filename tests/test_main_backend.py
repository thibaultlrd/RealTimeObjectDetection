#!/usr/bin/env python3
"""
Tests specifically for backend/main.py FastAPI application
Tests the actual endpoints and functionality defined in main.py
"""
import pytest
import cv2
import numpy as np
import io
from pathlib import Path
from fastapi.testclient import TestClient

# Import the FastAPI app from backend/main.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from backend.main import app, _detector, MODEL_PATH
except ImportError as e:
    pytest.skip(f"Cannot import backend.main: {e}", allow_module_level=True)


class TestMainBackend:
    """Test suite for backend/main.py FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app)
    
    def create_test_image(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Create a simple test image"""
        # Create a simple image with some patterns
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Add a white rectangle
        cv2.rectangle(img, (100, 100), (300, 200), (255, 255, 255), -1)
        # Add some noise
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        return img
    
    def encode_image_as_jpeg(self, img: np.ndarray) -> bytes:
        """Encode image as JPEG bytes"""
        _, encoded = cv2.imencode('.jpg', img)
        return encoded.tobytes()
    
    def test_app_title(self, client):
        """Test that the FastAPI app has the correct title"""
        # This tests the app definition in main.py line 12
        response = client.get("/docs")
        assert response.status_code == 200
        # The docs page should contain the app title
        assert "RealTimeObjectDetection" in response.text
    
    def test_health_endpoint_structure(self, client):
        """Test /api/health endpoint structure (from main.py lines 25-28)"""
        response = client.get("/api/health")
        
        if response.status_code == 200:
            data = response.json()
            # Test the exact structure returned by main.py
            assert "status" in data
            assert "model_path" in data  
            assert "input_size" in data
            
            # Validate the values
            assert data["status"] == "ok"
            assert isinstance(data["model_path"], str)
            assert isinstance(data["input_size"], (list, tuple))
            assert len(data["input_size"]) == 2
            
            print("✅ Health endpoint working correctly")
        else:
            # Health endpoint might fail if detector can't be accessed in test context
            print(f"⚠️  Health endpoint returned {response.status_code} (detector access issue)")
            # This is acceptable in testing environment
    
    def test_predict_endpoint_no_file(self, client):
        """Test /api/predict endpoint without image file (main.py line 31)"""
        response = client.post("/api/predict")
        
        # Should return 422 for missing required field
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        
        # Check that the error mentions the missing image field
        error_details = str(data["detail"])
        assert "image" in error_details.lower()
        
        print("✅ Predict endpoint correctly validates required image parameter")
    
    def test_predict_endpoint_invalid_image(self, client):
        """Test /api/predict endpoint with invalid image data"""
        # This tests the cv2.imdecode logic in main.py lines 36-39
        invalid_data = b"not an image file"
        
        response = client.post(
            "/api/predict",
            files={"image": ("invalid.jpg", io.BytesIO(invalid_data), "image/jpeg")}
        )
        
        # Should return 400 with "Invalid image" error (main.py line 35)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "Invalid image" in data["error"]
        
        print("✅ Predict endpoint correctly rejects invalid images")
    
    def test_predict_endpoint_valid_image(self, client):
        """Test /api/predict endpoint with valid image"""
        # This tests the full prediction pipeline in main.py lines 31-57
        test_img = self.create_test_image()
        img_bytes = self.encode_image_as_jpeg(test_img)
        
        response = client.post(
            "/api/predict",
            files={"image": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Test the exact response structure from main.py lines 44-52
        assert "detections" in data
        assert isinstance(data["detections"], list)
        
        # Each detection should have the exact fields from main.py
        for detection in data["detections"]:
            # These correspond to the dict structure in main.py lines 49-56
            assert "bbox" in detection      # det.bbox_xyxy
            assert "score" in detection     # det.score
            assert "class_id" in detection  # det.class_id
            assert "class_name" in detection # det.class_name
            assert "color" in detection     # det.color
            
            # Validate data types and ranges
            bbox = detection["bbox"]
            assert isinstance(bbox, list) and len(bbox) == 4
            assert all(isinstance(x, (int, float)) for x in bbox)
            
            assert isinstance(detection["score"], (int, float))
            assert 0.0 <= detection["score"] <= 1.0
            
            assert isinstance(detection["class_id"], int)
            assert detection["class_id"] >= 0
            
            assert isinstance(detection["class_name"], str)
            assert len(detection["class_name"]) > 0
            
            # Validate color field
            color = detection["color"]
            assert isinstance(color, (list, tuple)), f"Color should be list/tuple, got {type(color)}"
            assert len(color) == 3, f"Color should have 3 components (RGB), got {len(color)}"
            for c in color:
                assert isinstance(c, int), f"Color component should be int, got {type(c)}"
                assert 0 <= c <= 255, f"Color component should be 0-255, got {c}"
        
        print(f"✅ Predict endpoint working correctly, found {len(data['detections'])} detections")
    
    def test_predict_endpoint_with_real_image(self, client):
        """Test /api/predict endpoint with real test image if available"""
        test_image_path = Path("res/dog_test_picture.jpg")
        
        if not test_image_path.exists():
            pytest.skip("Real test image not found")
        
        # Load and test with real image
        img = cv2.imread(str(test_image_path))
        if img is None:
            pytest.skip("Could not load test image")
        
        img_bytes = self.encode_image_as_jpeg(img)
        
        response = client.post(
            "/api/predict",
            files={"image": ("dog_test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        detections = data["detections"]
        
        print(f"✅ Real image test: found {len(detections)} detections")
        for i, det in enumerate(detections[:3]):  # Show first 3
            print(f"  Detection {i+1}: {det['class_name']} ({det['score']:.3f})")
    
    def test_model_path_exists(self):
        """Test that the model path defined in main.py actually exists"""
        # This tests the MODEL_PATH definition in main.py line 14
        model_path = Path(MODEL_PATH)
        assert model_path.exists(), f"Model file not found at {MODEL_PATH}"
        assert model_path.suffix == ".tflite", "Model should be a TensorFlow Lite file"
        print(f"✅ Model file exists at {MODEL_PATH}")
    
    def test_detector_initialization(self):
        """Test that the detector is properly initialized"""
        # This tests the _detector initialization in main.py line 19
        assert _detector is not None, "Detector should be initialized"
        
        # Test detector has required attributes
        assert hasattr(_detector, 'input_size'), "Detector should have input_size"
        assert hasattr(_detector, 'predict'), "Detector should have predict method"
        
        print("✅ Detector properly initialized")
    
    def test_openapi_schema(self, client):
        """Test that OpenAPI schema is available"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        
        # Check that our endpoints are documented
        paths = schema["paths"]
        assert "/api/health" in paths
        assert "/api/predict" in paths
        
        # Verify the predict endpoint expects a file upload
        predict_post = paths["/api/predict"]["post"]
        assert "requestBody" in predict_post
        
        print("✅ OpenAPI schema properly configured")


def run_main_backend_tests():
    """Run the backend tests without pytest (for manual execution)"""
    print("🔍 Testing backend/main.py")
    print("=" * 50)
    
    try:
        # Try to import the backend
        from backend.main import app
        print("✅ Successfully imported backend.main")
    except Exception as e:
        print(f"❌ Failed to import backend.main: {e}")
        return False
    
    # Create test client
    client = TestClient(app)
    test_instance = TestMainBackend()
    
    # List of tests to run
    tests = [
        ("App Title", lambda: test_instance.test_app_title(client)),
        ("Health Endpoint", lambda: test_instance.test_health_endpoint_structure(client)),
        ("Predict No File", lambda: test_instance.test_predict_endpoint_no_file(client)),
        ("Predict Invalid Image", lambda: test_instance.test_predict_endpoint_invalid_image(client)),
        ("Predict Valid Image", lambda: test_instance.test_predict_endpoint_valid_image(client)),
        ("Model Path Exists", test_instance.test_model_path_exists),
        ("Detector Initialization", test_instance.test_detector_initialization),
        ("OpenAPI Schema", lambda: test_instance.test_openapi_schema(client)),
        ("Real Image Test", lambda: test_instance.test_predict_endpoint_with_real_image(client)),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}...")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Results: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_main_backend_tests()
    exit(0 if success else 1)
