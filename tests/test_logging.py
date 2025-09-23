#!/usr/bin/env python3
"""
Test script to verify logging configuration works correctly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Test the logging setup
    from rtod.utils.logging_setup import logger
    from rtod.models import TFLiteYoloDetector
    import numpy as np
    
    print("Testing logging configuration...")
    
    # Test basic logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test logging in the detector
    print("\nTesting logging in TFLiteYoloDetector...")
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    detector = TFLiteYoloDetector(model_path=None, mock=True)
    
    # This should trigger the logging in the detector
    detections = detector.predict(img)
    print(f"Mock detector returned {len(detections)} detections")
    
    # Test with real model path (if it exists)
    model_path = "rtod/models/B_72_foe_box_yolo8_256_2_v1_float32.tflite"
    if os.path.exists(model_path):
        print(f"\nTesting with real model: {model_path}")
        real_detector = TFLiteYoloDetector(model_path=model_path, mock=False)
        detections = real_detector.predict(img)
        print(f"Real detector returned {len(detections)} detections")
    
    print("\n✅ Logging test completed successfully!")
    print("Check the logs.log file for file output, and console for console output.")
    
except Exception as e:
    print(f"❌ Logging test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
