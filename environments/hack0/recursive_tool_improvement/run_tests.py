#!/usr/bin/env python3
"""
Test runner for the Recursive Tool Improvement Environment.

Run this script from the recursive_tool_improvement directory to execute all tests.
"""

import unittest
import os
import sys

if __name__ == "__main__":
    # Ensure we're running from the right directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # Add the current directory to the path so imports work
    sys.path.insert(0, current_dir)
    
    # Load all tests from the tests directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test_*.py")
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful())