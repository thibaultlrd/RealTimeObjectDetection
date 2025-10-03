#!/usr/bin/env python3
"""
Tests for individual functions from frontend/app.py
This approach tests the functions in isolation without importing the full app
"""
import numpy as np
import cv2
import io
import threading
import time
from unittest.mock import Mock, patch
from queue import Queue
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFrontendFunctions:
    """Test suite for frontend/app.py functions"""
    
    def create_test_image(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Create a test image for testing"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Add a white rectangle
        cv2.rectangle(img, (100, 100), (300, 200), (255, 255, 255), -1)
        return img
    
    def create_test_detections(self) -> list:
        """Create sample detection data matching backend format"""
        return [
            {
                "bbox": [100, 100, 200, 150],
                "score": 0.85,
                "class_id": 0,
                "class_name": "person",
                "color": [255, 0, 0]  # Red
            },
            {
                "bbox": [250, 200, 350, 300],
                "score": 0.72,
                "class_id": 16,
                "class_name": "dog",
                "color": [0, 255, 0]  # Green
            }
        ]
    
    def test_draw_boxes_function(self):
        """Test the draw_boxes function logic"""
        # Import the function here to test it in isolation
        from frontend.app import draw_boxes
        
        # Create test image and detections
        img = self.create_test_image()
        detections = self.create_test_detections()
        original_img = img.copy()
        
        # Apply draw_boxes
        result_img = draw_boxes(img, detections)
        
        # Verify image was modified (should have drawn rectangles and text)
        assert not np.array_equal(original_img, result_img), "Image should be modified by draw_boxes"
        
        # Verify return type and shape
        assert isinstance(result_img, np.ndarray)
        assert result_img.shape == original_img.shape
        assert result_img.dtype == original_img.dtype
        
        print("✅ draw_boxes function working correctly")
    
    def test_draw_boxes_empty_detections(self):
        """Test draw_boxes with empty detections list"""
        from frontend.app import draw_boxes
        
        img = self.create_test_image()
        original_img = img.copy()
        
        result_img = draw_boxes(img, [])
        
        # With no detections, image should remain unchanged
        assert np.array_equal(original_img, result_img), "Image should be unchanged with empty detections"
        
        print("✅ draw_boxes handles empty detections correctly")
    
    @patch('requests.post')
    def test_detect_objects_async_success(self, mock_post):
        """Test detect_objects_async with successful response"""
        # Mock successful response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"detections": self.create_test_detections()}
        mock_post.return_value = mock_response
        
        # Import function and test
        from frontend.app import detect_objects_async
        
        # Test with sample image
        test_img = self.create_test_image()
        result = detect_objects_async(test_img)
        
        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['timeout'] == 5  # Reduced timeout for real-time
        assert 'files' in call_args[1]
        
        # Verify result
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]['class_name'] == 'person'
        assert result[1]['class_name'] == 'dog'
        
        print("✅ detect_objects_async handles successful response correctly")
    
    @patch('requests.post')
    def test_detect_objects_async_failure(self, mock_post):
        """Test detect_objects_async with failed response"""
        # Mock failed response
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        from frontend.app import detect_objects_async
        
        test_img = self.create_test_image()
        result = detect_objects_async(test_img)
        
        # Should return None on failure
        assert result is None
        
        print("✅ detect_objects_async handles failed response correctly")
    
    @patch('requests.post')
    def test_detect_objects_async_exception(self, mock_post):
        """Test detect_objects_async with request exception"""
        # Mock exception
        mock_post.side_effect = Exception("Network error")
        
        from frontend.app import detect_objects_async
        
        test_img = self.create_test_image()
        result = detect_objects_async(test_img)
        
        # Should return None on exception
        assert result is None
        
        print("✅ detect_objects_async handles exceptions correctly")
    
    def test_detection_worker_basic_functionality(self):
        """Test detection_worker function"""
        from frontend.app import detection_worker
        
        # Create test queues and event
        detection_queue = Queue()
        result_queue = Queue()
        stop_event = threading.Event()
        
        # Add test image to detection queue
        test_img = self.create_test_image()
        detection_queue.put(test_img)
        
        # Mock detect_objects_async to return test data
        with patch('frontend.app.detect_objects_async') as mock_detect:
            mock_detect.return_value = self.create_test_detections()
            
            # Start worker in thread
            worker_thread = threading.Thread(
                target=detection_worker,
                args=(detection_queue, result_queue, stop_event),
                daemon=True
            )
            worker_thread.start()
            
            # Wait a bit for processing
            time.sleep(0.2)
            
            # Stop the worker
            stop_event.set()
            worker_thread.join(timeout=1)
            
            # Check that result was added to result queue
            assert not result_queue.empty(), "Result should be in result queue"
            
            result = result_queue.get()
            assert 'detections' in result
            assert 'timestamp' in result
            assert len(result['detections']) == 2
            
        print("✅ detection_worker processes frames correctly")
    
    def test_detection_worker_stop_event(self):
        """Test that detection_worker respects stop event"""
        from frontend.app import detection_worker
        
        detection_queue = Queue()
        result_queue = Queue()
        stop_event = threading.Event()
        
        # Set stop event immediately
        stop_event.set()
        
        # Worker should exit immediately
        with patch('frontend.app.detect_objects_async') as mock_detect:
            detection_worker(detection_queue, result_queue, stop_event)
            
            # detect_objects_async should not be called
            mock_detect.assert_not_called()
        
        print("✅ detection_worker respects stop event")
    
    def test_image_encoding_pipeline(self):
        """Test that image encoding works correctly"""
        test_img = self.create_test_image()
        
        # Test cv2.imencode works
        _, encoded = cv2.imencode('.jpg', test_img)
        assert encoded is not None
        assert len(encoded) > 0
        
        # Test that encoded data can be converted to bytes
        img_bytes = encoded.tobytes()
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0
        
        # Test that bytes can be wrapped in BytesIO
        bio = io.BytesIO(img_bytes)
        assert bio.getvalue() == img_bytes
        
        print("✅ Image encoding pipeline works correctly")
    
    @patch('streamlit.write')
    @patch('streamlit.image')
    @patch('streamlit.error')
    @patch('requests.post')
    def test_process_static_image_success(self, mock_post, mock_st_error, mock_st_image, mock_st_write):
        """Test process_static_image function with successful response"""
        # Mock successful response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"detections": self.create_test_detections()}
        mock_post.return_value = mock_response
        
        # Mock cv2.cvtColor
        with patch('cv2.cvtColor') as mock_cvt_color:
            mock_cvt_color.return_value = self.create_test_image()
            
            from frontend.app import process_static_image
            
            test_img = self.create_test_image()
            process_static_image(test_img)
            
            # Verify request was made
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]['timeout'] == 30  # Static image timeout
            
            # Verify Streamlit components were called
            assert mock_st_write.call_count >= 3  # Backend URL, detection count, detection details
            mock_st_image.assert_called_once()
        
        print("✅ process_static_image handles successful response correctly")
    
    @patch('streamlit.error')
    @patch('requests.post')
    def test_process_static_image_failure(self, mock_post, mock_st_error):
        """Test process_static_image with failed response"""
        # Mock failed response
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        from frontend.app import process_static_image
        
        test_img = self.create_test_image()
        process_static_image(test_img)
        
        # Verify error was displayed
        mock_st_error.assert_called_once()
        error_call = mock_st_error.call_args[0][0]
        assert "Backend error: 500" in error_call
        
        print("✅ process_static_image handles failed response correctly")


def run_frontend_function_tests():
    """Run all frontend function tests manually (without pytest)"""
    print("🔍 Testing frontend/app.py Functions")
    print("=" * 50)
    
    test_instance = TestFrontendFunctions()
    
    # List of tests to run
    tests = [
        ("Draw Boxes Function", test_instance.test_draw_boxes_function),
        ("Draw Boxes Empty", test_instance.test_draw_boxes_empty_detections),
        ("Detect Objects Success", test_instance.test_detect_objects_async_success),
        ("Detect Objects Failure", test_instance.test_detect_objects_async_failure),
        ("Detect Objects Exception", test_instance.test_detect_objects_async_exception),
        ("Detection Worker Basic", test_instance.test_detection_worker_basic_functionality),
        ("Detection Worker Stop", test_instance.test_detection_worker_stop_event),
        ("Image Encoding", test_instance.test_image_encoding_pipeline),
        ("Process Static Success", test_instance.test_process_static_image_success),
        ("Process Static Failure", test_instance.test_process_static_image_failure),
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
    
    if failed == 0:
        print("🎉 All frontend function tests passed!")
    else:
        print(f"⚠️  {failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_frontend_function_tests()
    exit(0 if success else 1)
