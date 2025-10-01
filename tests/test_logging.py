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
    
    print("Testing logging configuration...")
    
    # Test basic logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("\n✅ Logging test completed successfully!")
    print("Check the logs.log file for file output, and console for console output.")
    
except Exception as e:
    print(f"❌ Logging test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
