#!/usr/bin/env python3
"""
Week 3 Real Data Integration Test
Tests the integration of Gemini-powered AI research with real CloudVR-PerfGuard data
"""

import asyncio
import sys
import os
from pathlib import Path

# Add CloudVR-PerfGuard to path
sys.path.append('cloudvr_perfguard')

from ai_integration.real_data_integration import RealDataResearchPipeline
from scripts.populate_test_data import TestDataGenerator


async def test_week3_real_data_integration():
    """Test Week 3 real data integration"""
    
    print("🚀 Week 3: Real CloudVR-PerfGuard Data Integration Test")
    print("=" * 70)
    print("Testing Gemini AI research generation with real VR performance data")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Database Population
    print("\n📊 Test 1: Database Population with Realistic VR Data...")
    try:
        generator = TestDataGenerator()
        await generator.initialize()
        
        # Populate with realistic test data
        population_summary = await generator.populate_realistic_data(num_tests_per_app=8)
        
        if population_summary["total_jobs_created"] >= 20:
            print(f"   ✅ Database populated successfully!")
            print(f"      Jobs created: {population_summary['total_jobs_created']}")
            print(f"      Apps: {', '.join(population_summary['apps_populated'])}")
            tests_passed += 1
        else:
            print(f"   ❌ Insufficient test data created: {population_summary['total_jobs_created']}")
        
        await generator.close()
        
    except Exception as e:
        print(f"   ❌ Database population failed: {e}")
    
    # Test 2: Real Data Retrieval
    print("\n🔍 Test 2: Real Data Retrieval from CloudVR-PerfGuard...")
    try:
        pipeline = RealDataResearchPipeline()
        await pipeline.initialize()
        
        # Retrieve real performance data
        real_data = await pipeline.get_real_performance_data(limit=50)
        
        if len(real_data) >= 10:
            print(f"   ✅ Real data retrieved successfully!")
            print(f"      Test results: {len(real_data)}")
            
            # Check data structure
            sample_data = real_data[0]
            required_fields = ["job_metadata", "individual_results", "aggregated_metrics"]
            
            if all(field in sample_data for field in required_fields):
                print(f"      Data structure: Valid")
                print(f"      Sample app: {sample_data['job_metadata']['app_name']}")
                print(f"      Individual tests: {len(sample_data['individual_results'])}")
                tests_passed += 1
            else:
                print(f"   ❌ Invalid data structure")
        else:
            print(f"   ❌ Insufficient real data: {len(real_data)} (minimum 10)")
        
        await pipeline.close()
        
    except Exception as e:
        print(f"   ❌ Real data retrieval failed: {e}")
    
    # Test 3: Single App Research Generation
    print("\n📝 Test 3: Single App Research Generation...")
    try:
        pipeline = RealDataResearchPipeline()
        await pipeline.initialize()
        
        # Get available apps
        real_data = await pipeline.get_real_performance_data(limit=20)
        if real_data:
            test_app = real_data[0]["job_metadata"]["app_name"]
            
            # Generate research for specific app
            research_result = await pipeline.generate_research_from_real_data(
                app_name=test_app,
                research_type="performance_analysis"
            )
            
            if research_result["success"]:
                print(f"   ✅ Single app research generated!")
                print(f"      App analyzed: {test_app}")
                print(f"      Data count: {research_result['data_count']}")
                print(f"      Papers: {len(research_result['papers'])}")
                print(f"      Functions: {len(research_result['functions'])}")
                print(f"      Quality: {research_result['research_quality']:.1f}/100")
                print(f"      Cost: ${research_result['total_cost']:.2f}")
                tests_passed += 1
            else:
                print(f"   ❌ Research generation failed: {research_result.get('error', 'Unknown')}")
        else:
            print(f"   ❌ No real data available for testing")
        
        await pipeline.close()
        
    except Exception as e:
        print(f"   ❌ Single app research failed: {e}")
    
    # Test 4: Comprehensive Multi-App Research
    print("\n🔬 Test 4: Comprehensive Multi-App Research...")
    try:
        pipeline = RealDataResearchPipeline()
        await pipeline.initialize()
        
        # Generate comprehensive research across all apps
        research_result = await pipeline.generate_research_from_real_data(
            app_name=None,  # All apps
            research_type="comprehensive"
        )
        
        if research_result["success"]:
            print(f"   ✅ Comprehensive research generated!")
            print(f"      Apps analyzed: {len(research_result['apps_analyzed'])}")
            print(f"      App names: {', '.join(research_result['apps_analyzed'][:3])}")
            print(f"      Total data: {research_result['data_count']} tests")
            print(f"      Papers generated: {len(research_result['papers'])}")
            print(f"      Functions discovered: {len(research_result['functions'])}")
            print(f"      Research quality: {research_result['research_quality']:.1f}/100")
            print(f"      Total cost: ${research_result['total_cost']:.2f}")
            
            # Validate research content
            if research_result["papers"]:
                sample_paper = research_result["papers"][0]
                print(f"      Sample paper: {sample_paper['title'][:50]}...")
                print(f"      Paper quality: {sample_paper['quality_score']}/100")
            
            if research_result["functions"]:
                sample_function = research_result["functions"][0]
                print(f"      Sample function: {sample_function['domain']}")
                print(f"      Function fitness: {sample_function['fitness_score']:.3f}")
            
            tests_passed += 1
        else:
            print(f"   ❌ Comprehensive research failed: {research_result.get('error', 'Unknown')}")
        
        await pipeline.close()
        
    except Exception as e:
        print(f"   ❌ Comprehensive research failed: {e}")
    
    # Test 5: Research Quality Validation
    print("\n🎯 Test 5: Research Quality Validation...")
    try:
        pipeline = RealDataResearchPipeline()
        await pipeline.initialize()
        
        # Generate research and validate quality
        research_result = await pipeline.generate_research_from_real_data(
            research_type="comprehensive"
        )
        
        if research_result["success"]:
            quality_score = research_result["research_quality"]
            cost = research_result["total_cost"]
            
            # Quality thresholds
            quality_passed = quality_score >= 70.0
            cost_passed = cost <= 30.0
            content_passed = len(research_result["papers"]) >= 1 and len(research_result["functions"]) >= 1
            
            if quality_passed and cost_passed and content_passed:
                print(f"   ✅ Research quality validation passed!")
                print(f"      Quality score: {quality_score:.1f}/100 (≥70 required)")
                print(f"      Cost efficiency: ${cost:.2f} (≤$30 required)")
                print(f"      Content completeness: ✅")
                
                # Check for real data context
                if research_result["papers"]:
                    paper = research_result["papers"][0]
                    if "real_data_context" in paper:
                        print(f"      Real data context: ✅")
                        print(f"      Data source: {paper['real_data_context']['data_source']}")
                
                tests_passed += 1
            else:
                print(f"   ❌ Quality validation failed:")
                print(f"      Quality: {quality_score:.1f}/100 {'✅' if quality_passed else '❌'}")
                print(f"      Cost: ${cost:.2f} {'✅' if cost_passed else '❌'}")
                print(f"      Content: {'✅' if content_passed else '❌'}")
        else:
            print(f"   ❌ Research generation failed for quality validation")
        
        await pipeline.close()
        
    except Exception as e:
        print(f"   ❌ Research quality validation failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 WEEK 3 REAL DATA INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    test_names = [
        "Database Population with Realistic VR Data",
        "Real Data Retrieval from CloudVR-PerfGuard",
        "Single App Research Generation",
        "Comprehensive Multi-App Research",
        "Research Quality Validation"
    ]
    
    for i in range(1, total_tests + 1):
        status = "✅ PASSED" if i <= tests_passed else "❌ FAILED"
        print(f"Test {i}: {test_names[i-1]}: {status}")
    
    print(f"\nOverall: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("\n🎉 Week 3 Real Data Integration: FULLY OPERATIONAL!")
        print("\n🔬 Real Data Research Capabilities:")
        print("   ✅ Real CloudVR-PerfGuard database integration")
        print("   ✅ Realistic VR performance data generation")
        print("   ✅ Single app performance analysis")
        print("   ✅ Multi-app comparative research")
        print("   ✅ Gemini AI-powered paper generation")
        print("   ✅ Gemini AI-powered function discovery")
        print("   ✅ Quality validation and cost control")
        print("   ✅ Real data context preservation")
        
        print("\n🎯 Week 3 Achievements:")
        print("   📊 Real VR performance data integration")
        print("   📝 AI research from production data")
        print("   🧬 Optimization functions from real metrics")
        print("   💰 Cost-effective research generation")
        print("   🎨 High-quality scientific outputs")
        
        print("\n🚀 Ready for Week 4: Production Scaling & Automation")
        print("   Next: Continuous research pipeline")
        print("   Next: Multi-application scaling")
        print("   Next: Research quality optimization")
        print("   Next: CI/CD integration")
    else:
        print(f"\n⚠️  Week 3 Integration: {tests_passed}/{total_tests} tests passed")
        print("   Some real data features may need configuration")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = asyncio.run(test_week3_real_data_integration())
    sys.exit(0 if success else 1) 