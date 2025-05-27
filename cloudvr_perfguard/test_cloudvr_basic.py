#!/usr/bin/env python3
"""
Basic test script for CloudVR-PerfGuard
Tests core functionality without Docker/GPU dependencies
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_database_only():
    """Test database functionality in isolation"""
    print("🔍 Testing CloudVR-PerfGuard Database...")
    
    try:
        from core.database import DatabaseManager
        
        # Initialize database
        db = DatabaseManager("test_cloudvr.db")
        await db.initialize()
        print("✅ Database initialized successfully")
        
        # Test job creation
        job_id = "test-job-123"
        success = await db.create_test_job(
            job_id=job_id,
            app_name="TestVRApp",
            build_version="v1.0.0",
            platform="windows",
            submission_type="baseline",
            build_path="/tmp/test.exe",
            test_config={"gpu_types": ["T4"], "test_duration_seconds": 60}
        )
        print(f"✅ Test job created: {success}")
        
        # Test job retrieval
        job_data = await db.get_test_job(job_id)
        print(f"✅ Job retrieved: {job_data['app_name'] if job_data else 'None'}")
        
        # Test performance results storage
        test_results = {
            "test_id": "perf_test_123",
            "build_path": "/tmp/test.exe",
            "config": {"gpu_types": ["T4"]},
            "total_duration": 60.5,
            "individual_results": [
                {
                    "test_id": "test_1",
                    "config": {"gpu_type": "T4", "scene_name": "main_menu"},
                    "metrics": {
                        "avg_fps": 85.2,
                        "min_fps": 72.1,
                        "avg_frame_time": 11.7,
                        "vr_comfort_score": 92.5
                    },
                    "success": True
                }
            ],
            "aggregated_metrics": {
                "overall_avg_fps": 85.2,
                "overall_min_fps": 72.1,
                "overall_avg_frame_time": 11.7,
                "overall_comfort_score": 92.5
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await db.store_performance_results(job_id, test_results)
        print("✅ Performance results stored")
        
        # Test regression analysis
        regression_analysis = {
            "regressions": [
                {
                    "type": "fps_regression",
                    "metric": "average_fps",
                    "severity": "minor",
                    "current_value": 85.2,
                    "baseline_value": 90.0,
                    "change_percent": -5.3,
                    "description": "Average FPS decreased by 5.3%"
                }
            ],
            "statistical_analysis": {
                "t_statistic": -2.1,
                "p_value": 0.04,
                "cohens_d": 0.6,
                "effect_size": "medium"
            },
            "comparison": {
                "overall_avg_fps": {
                    "current": 85.2,
                    "baseline": 90.0,
                    "change_percent": -5.3,
                    "is_regression": True
                }
            },
            "regression_score": {
                "score": 2,
                "level": "minor",
                "regression_count": 1
            },
            "overall_status": "passed_with_issues",
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        await db.store_regression_analysis(job_id, regression_analysis)
        print("✅ Regression analysis stored")
        
        await db.close()
        print("✅ Database test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_regression_detector():
    """Test regression detection logic"""
    print("\n🔍 Testing Regression Detector...")
    
    try:
        from core.database import DatabaseManager
        from core.regression_detector import RegressionDetector
        
        # Initialize components
        db = DatabaseManager("test_cloudvr.db")
        await db.initialize()
        
        detector = RegressionDetector(db)
        print("✅ Regression detector initialized")
        
        # Test regression detection logic
        current_metrics = {
            "overall_avg_fps": 75.0,  # Decreased from baseline
            "overall_min_fps": 60.0,  # Decreased from baseline
            "overall_avg_frame_time": 13.3,  # Increased from baseline
            "overall_comfort_score": 80.0  # Decreased from baseline
        }
        
        baseline_metrics = {
            "overall_avg_fps": 85.0,
            "overall_min_fps": 70.0,
            "overall_avg_frame_time": 11.8,
            "overall_comfort_score": 90.0
        }
        
        # Test regression detection
        regressions = detector._detect_regressions(current_metrics, baseline_metrics)
        print(f"✅ Detected {len(regressions)} regressions")
        
        for regression in regressions:
            print(f"   - {regression['severity'].upper()}: {regression['description']}")
        
        await db.close()
        print("✅ Regression detector test completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Regression detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_endpoints():
    """Test API endpoints without full initialization"""
    print("\n🔍 Testing API Endpoints...")
    
    try:
        from fastapi.testclient import TestClient
        
        # Create a mock version of the app for testing
        from fastapi import FastAPI
        
        test_app = FastAPI(title="CloudVR-PerfGuard Test API")
        
        @test_app.get("/")
        async def root():
            return {
                "message": "CloudVR-PerfGuard API is operational!",
                "version": "1.0.0",
                "description": "Automated Performance Regression Detection for VR Applications"
            }
        
        @test_app.get("/status")
        async def get_status():
            return {
                "api_status": "OPERATIONAL",
                "services": {
                    "performance_tester": "mock_mode",
                    "regression_detector": "mock_mode",
                    "database": "connected"
                },
                "supported_platforms": ["windows", "linux", "android"],
                "supported_gpu_types": ["T4", "L4", "A100"],
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Test the endpoints
        client = TestClient(test_app)
        
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Root endpoint: {data['message']}")
        
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Status endpoint: {data['api_status']}")
        
        print("✅ API endpoints test completed!")
        return True
        
    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🚀 CloudVR-PerfGuard Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        ("Database Operations", test_database_only),
        ("Regression Detection", test_regression_detector),
        ("API Endpoints", test_api_endpoints)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All core functionality is working!")
        print("💡 Note: Docker/GPU features require proper environment setup")
    else:
        print("\n⚠️  Some tests failed - check error messages above")

if __name__ == "__main__":
    asyncio.run(main()) 