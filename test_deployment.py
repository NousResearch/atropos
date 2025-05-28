#!/usr/bin/env python3
"""
Deployment Test Script for CloudVR-PerfGuard AI Integration
Tests all endpoints after deployment
"""

import requests
import json
import time
import sys
import os
from datetime import datetime


def test_deployment(service_url: str):
    """Test the deployed AI integration service"""
    
    print("🧪 CloudVR-PerfGuard AI Integration Deployment Test")
    print("=" * 60)
    print(f"Service URL: {service_url}")
    print(f"Test Started: {datetime.now().isoformat()}")
    print()
    
    # Test data
    test_job_id = f"deploy_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Track test results
    results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "errors": []
    }
    
    # Test 1: Health Check
    print("1️⃣ Testing Health Check...")
    results["total_tests"] += 1
    try:
        response = requests.get(f"{service_url}/health")
        if response.status_code == 200:
            print("   ✅ Health check passed")
            results["passed"] += 1
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            results["failed"] += 1
            results["errors"].append(f"Health check returned {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        results["failed"] += 1
        results["errors"].append(f"Health check error: {str(e)}")
    print()
    
    # Test 2: Start Evolution
    print("2️⃣ Testing Evolution Start...")
    results["total_tests"] += 1
    try:
        response = requests.post(
            f"{service_url}/api/v1/research/evolution/start",
            json={
                "job_id": test_job_id,
                "evolution_type": "visual_cue_discovery"
            }
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("   ✅ Evolution started successfully")
                print(f"   📊 Functions evolved: {data.get('result', {}).get('total_functions_evolved', 0)}")
                results["passed"] += 1
            else:
                print(f"   ❌ Evolution failed: {data}")
                results["failed"] += 1
                results["errors"].append(f"Evolution failed: {data}")
        else:
            print(f"   ❌ Evolution endpoint error: {response.status_code}")
            results["failed"] += 1
            results["errors"].append(f"Evolution endpoint returned {response.status_code}")
    except Exception as e:
        print(f"   ❌ Evolution test error: {e}")
        results["failed"] += 1
        results["errors"].append(f"Evolution test error: {str(e)}")
    print()
    
    # Test 3: Check Status
    print("3️⃣ Testing Status Check...")
    results["total_tests"] += 1
    time.sleep(2)  # Wait for evolution to process
    try:
        response = requests.get(f"{service_url}/api/v1/research/status/{test_job_id}")
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("   ✅ Status check passed")
                status_data = data.get("research_status", {})
                print(f"   📊 Job status: {status_data.get('job_status', 'unknown')}")
                print(f"   📊 Functions: {status_data.get('evolved_functions_count', 0)}")
                results["passed"] += 1
            else:
                print(f"   ❌ Status check failed: {data}")
                results["failed"] += 1
                results["errors"].append(f"Status check failed: {data}")
        else:
            print(f"   ❌ Status endpoint error: {response.status_code}")
            results["failed"] += 1
            results["errors"].append(f"Status endpoint returned {response.status_code}")
    except Exception as e:
        print(f"   ❌ Status test error: {e}")
        results["failed"] += 1
        results["errors"].append(f"Status test error: {str(e)}")
    print()
    
    # Test 4: Get Evolved Functions
    print("4️⃣ Testing Get Functions...")
    results["total_tests"] += 1
    try:
        response = requests.get(f"{service_url}/api/v1/research/functions/{test_job_id}")
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("   ✅ Get functions passed")
                print(f"   📊 Functions retrieved: {data.get('count', 0)}")
                results["passed"] += 1
            else:
                print(f"   ❌ Get functions failed: {data}")
                results["failed"] += 1
                results["errors"].append(f"Get functions failed: {data}")
        else:
            print(f"   ❌ Functions endpoint error: {response.status_code}")
            results["failed"] += 1
            results["errors"].append(f"Functions endpoint returned {response.status_code}")
    except Exception as e:
        print(f"   ❌ Functions test error: {e}")
        results["failed"] += 1
        results["errors"].append(f"Functions test error: {str(e)}")
    print()
    
    # Test 5: Generate Paper (Quick Test)
    print("5️⃣ Testing Paper Generation...")
    results["total_tests"] += 1
    try:
        response = requests.post(
            f"{service_url}/api/v1/research/paper/generate",
            json={
                "job_id": test_job_id,
                "paper_type": "vr_affordance_discovery",
                "custom_focus": "Deployment Test"
            }
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("   ✅ Paper generation started")
                result = data.get("result", {})
                print(f"   📊 Paper ID: {result.get('paper_id', 'unknown')}")
                print(f"   📊 Cost: ${result.get('generation_cost', 0):.2f}")
                results["passed"] += 1
            else:
                print(f"   ❌ Paper generation failed: {data}")
                results["failed"] += 1
                results["errors"].append(f"Paper generation failed: {data}")
        else:
            print(f"   ❌ Paper endpoint error: {response.status_code}")
            results["failed"] += 1
            results["errors"].append(f"Paper endpoint returned {response.status_code}")
    except Exception as e:
        print(f"   ❌ Paper test error: {e}")
        results["failed"] += 1
        results["errors"].append(f"Paper test error: {str(e)}")
    print()
    
    # Test 6: Get Papers
    print("6️⃣ Testing Get Papers...")
    results["total_tests"] += 1
    time.sleep(2)  # Wait for paper generation
    try:
        response = requests.get(f"{service_url}/api/v1/research/papers/{test_job_id}")
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("   ✅ Get papers passed")
                print(f"   📊 Papers retrieved: {data.get('count', 0)}")
                results["passed"] += 1
            else:
                print(f"   ❌ Get papers failed: {data}")
                results["failed"] += 1
                results["errors"].append(f"Get papers failed: {data}")
        else:
            print(f"   ❌ Papers endpoint error: {response.status_code}")
            results["failed"] += 1
            results["errors"].append(f"Papers endpoint returned {response.status_code}")
    except Exception as e:
        print(f"   ❌ Papers test error: {e}")
        results["failed"] += 1
        results["errors"].append(f"Papers test error: {str(e)}")
    print()
    
    # Test 7: Complete Pipeline (Quick)
    print("7️⃣ Testing Complete Pipeline...")
    results["total_tests"] += 1
    pipeline_job_id = f"pipeline_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        response = requests.post(
            f"{service_url}/api/v1/research/pipeline/run",
            json={
                "job_id": pipeline_job_id,
                "pipeline_type": "quick_discovery",
                "custom_focus": "Deployment Pipeline Test"
            }
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("   ✅ Pipeline started successfully")
                result = data.get("result", {})
                print(f"   📊 Pipeline type: {result.get('pipeline_type', 'unknown')}")
                print(f"   📊 Total cost: ${result.get('total_cost', 0):.2f}")
                results["passed"] += 1
            else:
                print(f"   ❌ Pipeline failed: {data}")
                results["failed"] += 1
                results["errors"].append(f"Pipeline failed: {data}")
        else:
            print(f"   ❌ Pipeline endpoint error: {response.status_code}")
            results["failed"] += 1
            results["errors"].append(f"Pipeline endpoint returned {response.status_code}")
    except Exception as e:
        print(f"   ❌ Pipeline test error: {e}")
        results["failed"] += 1
        results["errors"].append(f"Pipeline test error: {str(e)}")
    print()
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {results['total_tests']}")
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed'] / results['total_tests'] * 100):.1f}%")
    print()
    
    if results["errors"]:
        print("⚠️ ERRORS:")
        for error in results["errors"]:
            print(f"  - {error}")
        print()
    
    # Performance Test
    print("⚡ PERFORMANCE TEST")
    print("-" * 40)
    
    # Test response times
    endpoints = [
        ("GET", f"/api/v1/research/status/{test_job_id}", None),
        ("GET", f"/api/v1/research/functions/{test_job_id}", None),
        ("GET", f"/api/v1/research/papers/{test_job_id}", None)
    ]
    
    for method, endpoint, data in endpoints:
        try:
            start_time = time.time()
            if method == "GET":
                response = requests.get(f"{service_url}{endpoint}")
            else:
                response = requests.post(f"{service_url}{endpoint}", json=data)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # Convert to ms
            print(f"{method} {endpoint}: {response_time:.0f}ms")
        except Exception as e:
            print(f"{method} {endpoint}: ERROR - {e}")
    
    print()
    print("=" * 60)
    
    if results["failed"] == 0:
        print("🎉 ALL TESTS PASSED! Deployment is successful!")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1


def main():
    """Main function"""
    
    # Get service URL from environment or command line
    if len(sys.argv) > 1:
        service_url = sys.argv[1]
    else:
        service_url = os.environ.get("SERVICE_URL")
        
        if not service_url:
            print("❌ Error: No service URL provided")
            print("Usage: python test_deployment.py <service_url>")
            print("   or: export SERVICE_URL=<url> && python test_deployment.py")
            sys.exit(1)
    
    # Ensure URL doesn't have trailing slash
    service_url = service_url.rstrip("/")
    
    # Run tests
    exit_code = test_deployment(service_url)
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 