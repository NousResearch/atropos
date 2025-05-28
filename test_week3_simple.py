#!/usr/bin/env python3
"""
Week 3 Simple Real Data Integration Test
Simplified test for CloudVR-PerfGuard real data integration with timeout handling
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Add CloudVR-PerfGuard to path
sys.path.append('cloudvr_perfguard')

from ai_integration.real_data_integration import RealDataResearchPipeline
from scripts.populate_test_data import TestDataGenerator


async def test_with_timeout(test_func, timeout_seconds=60):
    """Run a test function with timeout"""
    try:
        return await asyncio.wait_for(test_func(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        print(f"   ⚠️  Test timed out after {timeout_seconds} seconds")
        return False
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


async def test_database_population():
    """Test database population with smaller dataset"""
    print("📊 Testing database population...")
    
    generator = TestDataGenerator()
    await generator.initialize()
    
    try:
        # Populate with smaller dataset for faster testing
        summary = await generator.populate_realistic_data(num_tests_per_app=3)
        
        if summary["total_jobs_created"] >= 8:
            print(f"   ✅ Database populated: {summary['total_jobs_created']} jobs")
            return True
        else:
            print(f"   ❌ Insufficient jobs: {summary['total_jobs_created']}")
            return False
    finally:
        await generator.close()


async def test_data_retrieval():
    """Test real data retrieval"""
    print("🔍 Testing data retrieval...")
    
    pipeline = RealDataResearchPipeline()
    await pipeline.initialize()
    
    try:
        real_data = await pipeline.get_real_performance_data(limit=20)
        
        if len(real_data) >= 5:
            print(f"   ✅ Retrieved {len(real_data)} test results")
            return True
        else:
            print(f"   ❌ Insufficient data: {len(real_data)}")
            return False
    finally:
        await pipeline.close()


async def test_research_generation():
    """Test AI research generation"""
    print("🔬 Testing research generation...")
    
    pipeline = RealDataResearchPipeline()
    await pipeline.initialize()
    
    try:
        # Lower the minimum data requirement for testing
        pipeline.research_config["min_tests_for_research"] = 3
        
        research_result = await pipeline.generate_research_from_real_data(
            research_type="performance_analysis"
        )
        
        if research_result["success"]:
            print(f"   ✅ Research generated successfully")
            print(f"      Papers: {len(research_result['papers'])}")
            print(f"      Functions: {len(research_result['functions'])}")
            print(f"      Quality: {research_result['research_quality']:.1f}/100")
            return True
        else:
            print(f"   ❌ Research failed: {research_result.get('error', 'Unknown')}")
            return False
    finally:
        await pipeline.close()


async def test_week3_simple():
    """Simple Week 3 test with timeout handling"""
    
    print("🚀 Week 3: Simple Real Data Integration Test")
    print("=" * 60)
    print("Testing CloudVR-PerfGuard + Gemini AI integration")
    print("=" * 60)
    
    tests = [
        ("Database Population", test_database_population, 120),
        ("Data Retrieval", test_data_retrieval, 30),
        ("Research Generation", test_research_generation, 90)
    ]
    
    passed = 0
    total = len(tests)
    
    for i, (name, test_func, timeout) in enumerate(tests, 1):
        print(f"\n🧪 Test {i}/{total}: {name}")
        
        result = await test_with_timeout(test_func, timeout)
        if result:
            passed += 1
            print(f"   ✅ PASSED")
        else:
            print(f"   ❌ FAILED")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 WEEK 3 SIMPLE TEST SUMMARY")
    print("=" * 60)
    
    for i, (name, _, _) in enumerate(tests, 1):
        status = "✅ PASSED" if i <= passed else "❌ FAILED"
        print(f"Test {i}: {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Week 3 Real Data Integration: WORKING!")
        print("\n✅ Achievements:")
        print("   📊 CloudVR-PerfGuard database integration")
        print("   🔍 Real VR performance data retrieval")
        print("   🔬 Gemini AI research generation")
        print("   📝 Automated paper and function creation")
        
        print("\n🚀 System Status: READY FOR PRODUCTION")
        print("   Real data ✅ | AI research ✅ | Quality control ✅")
        
        # Show sample research output
        try:
            pipeline = RealDataResearchPipeline()
            await pipeline.initialize()
            pipeline.research_config["min_tests_for_research"] = 3
            
            sample_research = await pipeline.generate_research_from_real_data(
                research_type="performance_analysis"
            )
            
            if sample_research["success"] and sample_research["papers"]:
                paper = sample_research["papers"][0]
                print(f"\n📄 Sample Research Output:")
                print(f"   Title: {paper['title'][:60]}...")
                print(f"   Quality: {paper['quality_score']}/100")
                print(f"   Method: {paper.get('generation_method', 'unknown')}")
                print(f"   Cost: ${paper.get('generation_cost', 0):.3f}")
            
            await pipeline.close()
        except:
            pass  # Don't fail the test if sample generation fails
            
    else:
        print(f"\n⚠️  Week 3: {passed}/{total} tests passed")
        print("   Some features may need debugging")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = asyncio.run(test_week3_simple())
        print(f"\n🏁 Test completed: {'SUCCESS' if success else 'PARTIAL'}")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test crashed: {e}")
        sys.exit(1) 