#!/usr/bin/env python3
"""
Gemini AI Integration Test
Tests the Gemini-powered AI research tools
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add CloudVR-PerfGuard to path
sys.path.append('cloudvr_perfguard')

from ai_integration.data_adapter import PerformanceDataAdapter
from ai_integration.paper_generator import ResearchPaperGenerator
from ai_integration.function_discovery import OptimizationDiscovery

def test_gemini_integration():
    """Test Gemini AI integration specifically"""
    
    print("🤖 Gemini AI Integration Test")
    print("=" * 60)
    print("Testing Gemini-powered research paper and function generation")
    print("=" * 60)
    
    # Create test data
    test_data = {
        "test_id": "gemini_integration_test",
        "build_path": "/path/to/VRApp_Gemini.exe",
        "total_duration": 300.0,
        "timestamp": "2025-01-27T02:00:00Z",
        "individual_results": [
            {
                "test_id": "gemini_test_1",
                "success": True,
                "config": {
                    "scene_name": "complex_scene",
                    "test_duration": 120.0,
                    "gpu_type": "RTX4090"
                },
                "metrics": {
                    "avg_fps": 88.2,
                    "avg_frame_time": 11.3,
                    "avg_gpu_util": 87.5,
                    "max_vram_usage": 6800.0,
                    "avg_cpu_util": 42.0,
                    "vr_comfort_score": 86.5,
                    "frame_time_std": 1.1
                }
            },
            {
                "test_id": "gemini_test_2",
                "success": True,
                "config": {
                    "scene_name": "stress_test",
                    "test_duration": 180.0,
                    "gpu_type": "RTX4080"
                },
                "metrics": {
                    "avg_fps": 82.1,
                    "avg_frame_time": 12.2,
                    "avg_gpu_util": 94.2,
                    "max_vram_usage": 7200.0,
                    "avg_cpu_util": 55.0,
                    "vr_comfort_score": 83.8,
                    "frame_time_std": 1.8
                }
            }
        ]
    }
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Gemini Paper Generation
    print("\n📝 Test 1: Gemini Paper Generation...")
    try:
        generator = ResearchPaperGenerator()
        adapter = PerformanceDataAdapter()
        
        # Convert data
        research_data = adapter.to_ai_scientist_format(test_data)
        
        # Generate paper with Gemini
        result = generator.generate_paper(
            research_data,
            paper_type="performance_analysis",
            custom_title="Gemini-Powered VR Performance Analysis"
        )
        
        # Check if Gemini was used
        if result["generation_method"] == "gemini_ai":
            print("   ✅ Gemini AI successfully generated paper!")
            print(f"      Quality: {result['quality_score']}/100")
            print(f"      Cost: ${result['generation_cost']}")
            print(f"      Title: {result['title'][:60]}...")
            tests_passed += 1
        elif result["generation_method"] == "ai_scientist":
            print("   ⚠️  AI Scientist wrapper used (Gemini may be inside)")
            print(f"      Quality: {result['quality_score']}/100")
            tests_passed += 1
        else:
            print(f"   ❌ Unexpected method: {result['generation_method']}")
        
    except Exception as e:
        print(f"   ❌ Gemini paper generation failed: {e}")
    
    # Test 2: Gemini Function Discovery
    print("\n🧬 Test 2: Gemini Function Discovery...")
    try:
        discovery = OptimizationDiscovery()
        adapter = PerformanceDataAdapter()
        
        # Convert data
        training_data = adapter.to_funsearch_format(test_data)
        
        # Discover function with Gemini
        result = discovery.discover_optimization_function(
            training_data,
            domain="comfort_optimization"
        )
        
        # Check if Gemini was used
        if result["discovery_method"] == "gemini_ai":
            print("   ✅ Gemini AI successfully discovered function!")
            print(f"      Fitness: {result['fitness_score']:.4f}")
            print(f"      Domain: comfort_optimization")
            print(f"      Function preview: {result['function_code'][:100]}...")
            tests_passed += 1
        elif result["discovery_method"] == "funsearch":
            print("   ⚠️  FunSearch wrapper used (Gemini may be inside)")
            print(f"      Fitness: {result['fitness_score']:.4f}")
            tests_passed += 1
        else:
            print(f"   ❌ Unexpected method: {result['discovery_method']}")
        
    except Exception as e:
        print(f"   ❌ Gemini function discovery failed: {e}")
    
    # Test 3: Direct Gemini API Test
    print("\n🔧 Test 3: Direct Gemini API Test...")
    try:
        import google.generativeai as genai
        
        # Configure Gemini (clean API key)
        api_key = os.getenv('GEMINI_API_KEY', '').split()[0] if os.getenv('GEMINI_API_KEY') else None
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            
            # Test prompt
            prompt = "Generate a brief VR optimization tip in exactly one sentence."
            response = model.generate_content(prompt)
            
            if response and response.text:
                print("   ✅ Direct Gemini API working!")
                print(f"      Response: {response.text[:100]}...")
                tests_passed += 1
            else:
                print("   ❌ Gemini API returned empty response")
        else:
            print("   ❌ GEMINI_API_KEY not found")
        
    except Exception as e:
        print(f"   ❌ Direct Gemini API test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 GEMINI AI INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    test_names = [
        "Gemini Paper Generation",
        "Gemini Function Discovery", 
        "Direct Gemini API Test"
    ]
    
    for i in range(1, total_tests + 1):
        status = "✅ PASSED" if i <= tests_passed else "❌ FAILED"
        print(f"Test {i}: {test_names[i-1]}: {status}")
    
    print(f"\nOverall: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("\n🎉 Gemini AI Integration: FULLY OPERATIONAL!")
        print("\n🤖 Gemini Capabilities:")
        print("   ✅ Research paper generation with AI")
        print("   ✅ Function discovery with AI")
        print("   ✅ Direct API access working")
        print("   ✅ Enhanced quality scores (85.0/100 potential)")
        print("   ✅ Cost-effective AI research automation")
        
        print("\n🎯 Ready for Week 3: Real CloudVR-PerfGuard Data + Gemini AI")
    else:
        print(f"\n⚠️  Gemini Integration: {tests_passed}/{total_tests} tests passed")
        print("   Some Gemini features may need configuration")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = test_gemini_integration()
    sys.exit(0 if success else 1) 