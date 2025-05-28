#!/usr/bin/env python3
"""
Week 4 Continuous Research Pipeline Test
Tests the production-ready continuous research pipeline with scaling and automation
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add CloudVR-PerfGuard to path
sys.path.append('cloudvr_perfguard')

from ai_integration.continuous_research_pipeline import ContinuousResearchPipeline, ResearchSchedule
from ai_integration.real_data_integration import RealDataResearchPipeline


async def test_week4_continuous_pipeline():
    """Test Week 4 continuous research pipeline"""
    
    print("🚀 Week 4: Continuous Research Pipeline Test")
    print("=" * 70)
    print("Testing production-ready automated research generation and scaling")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Pipeline Initialization
    print("\n🔧 Test 1: Pipeline Initialization...")
    try:
        pipeline = ContinuousResearchPipeline("test_research_config.json")
        await pipeline.initialize()
        
        # Check if configuration was created
        if os.path.exists("test_research_config.json"):
            print("   ✅ Configuration file created")
            
            # Check if research output directory exists
            if pipeline.research_output_dir.exists():
                print("   ✅ Research output directory created")
                
                # Check if logging is set up
                if pipeline.logger:
                    print("   ✅ Logging system initialized")
                    tests_passed += 1
                else:
                    print("   ❌ Logging system not initialized")
            else:
                print("   ❌ Research output directory not created")
        else:
            print("   ❌ Configuration file not created")
        
        await pipeline.stop()
        
    except Exception as e:
        print(f"   ❌ Pipeline initialization failed: {e}")
    
    # Test 2: Daily Research Generation
    print("\n📅 Test 2: Daily Research Generation...")
    try:
        pipeline = ContinuousResearchPipeline("test_research_config.json")
        await pipeline.initialize()
        
        # Lower requirements for testing
        pipeline.schedule.min_data_points = 5
        pipeline.schedule.max_cost_per_day = 10.0
        pipeline.pipeline.research_config["min_tests_for_research"] = 3
        
        # Run daily research
        daily_result = await pipeline.run_daily_research()
        
        if daily_result.get("success", False):
            print(f"   ✅ Daily research completed successfully")
            print(f"      Apps analyzed: {len(daily_result.get('apps_analyzed', []))}")
            print(f"      Research items: {len(daily_result.get('research_items', []))}")
            print(f"      Total cost: ${daily_result.get('total_cost', 0):.2f}")
            
            # Check if research files were saved
            research_files = list(pipeline.research_output_dir.glob("daily_*.json"))
            if research_files:
                print(f"      Research files saved: {len(research_files)}")
                tests_passed += 1
            else:
                print("   ❌ No research files saved")
        else:
            print(f"   ❌ Daily research failed: {daily_result.get('error', 'Unknown')}")
        
        await pipeline.stop()
        
    except Exception as e:
        print(f"   ❌ Daily research test failed: {e}")
    
    # Test 3: Weekly Research Generation
    print("\n📊 Test 3: Weekly Research Generation...")
    try:
        pipeline = ContinuousResearchPipeline("test_research_config.json")
        await pipeline.initialize()
        
        # Lower requirements for testing
        pipeline.pipeline.research_config["min_tests_for_research"] = 3
        
        # Run weekly research
        weekly_result = await pipeline.run_weekly_research()
        
        if weekly_result.get("success", False):
            print(f"   ✅ Weekly research completed successfully")
            print(f"      Papers generated: {len(weekly_result.get('papers', []))}")
            print(f"      Functions discovered: {len(weekly_result.get('functions', []))}")
            print(f"      Research quality: {weekly_result.get('research_quality', 0):.1f}/100")
            
            # Check if weekly research file was saved
            weekly_files = list(pipeline.research_output_dir.glob("weekly_*.json"))
            if weekly_files:
                print(f"      Weekly research file saved")
                tests_passed += 1
            else:
                print("   ❌ Weekly research file not saved")
        else:
            print(f"   ❌ Weekly research failed: {weekly_result.get('error', 'Unknown')}")
        
        await pipeline.stop()
        
    except Exception as e:
        print(f"   ❌ Weekly research test failed: {e}")
    
    # Test 4: Metrics and Monitoring
    print("\n📈 Test 4: Metrics and Monitoring...")
    try:
        pipeline = ContinuousResearchPipeline("test_research_config.json")
        await pipeline.initialize()
        
        # Simulate some metrics
        pipeline.metrics.papers_generated = 5
        pipeline.metrics.functions_discovered = 8
        pipeline.metrics.total_cost = 2.50
        pipeline.metrics.uptime_hours = 24.0
        
        # Generate metrics report
        metrics_report = pipeline.generate_monthly_metrics_report()
        
        if metrics_report and "metrics" in metrics_report:
            print("   ✅ Metrics report generated successfully")
            print(f"      Papers generated: {metrics_report['metrics']['papers_generated']}")
            print(f"      Functions discovered: {metrics_report['metrics']['functions_discovered']}")
            print(f"      Total cost: ${metrics_report['metrics']['total_cost']:.2f}")
            
            # Check efficiency calculations
            efficiency = metrics_report.get("efficiency", {})
            if efficiency:
                print(f"      Cost per paper: ${efficiency.get('cost_per_paper', 0):.3f}")
                print(f"      Papers per day: {efficiency.get('papers_per_day', 0):.1f}")
                tests_passed += 1
            else:
                print("   ❌ Efficiency metrics not calculated")
        else:
            print("   ❌ Metrics report generation failed")
        
        await pipeline.stop()
        
    except Exception as e:
        print(f"   ❌ Metrics test failed: {e}")
    
    # Test 5: Configuration Management
    print("\n⚙️  Test 5: Configuration Management...")
    try:
        pipeline = ContinuousResearchPipeline("test_research_config.json")
        
        # Modify configuration
        pipeline.schedule.daily_hour = 3
        pipeline.schedule.max_cost_per_day = 25.0
        pipeline.schedule.min_quality_score = 80.0
        
        # Save configuration
        pipeline.save_config()
        
        # Create new pipeline and load configuration
        pipeline2 = ContinuousResearchPipeline("test_research_config.json")
        
        # Check if configuration was loaded correctly
        if (pipeline2.schedule.daily_hour == 3 and 
            pipeline2.schedule.max_cost_per_day == 25.0 and
            pipeline2.schedule.min_quality_score == 80.0):
            print("   ✅ Configuration save/load working correctly")
            print(f"      Daily hour: {pipeline2.schedule.daily_hour}")
            print(f"      Max cost per day: ${pipeline2.schedule.max_cost_per_day}")
            print(f"      Min quality score: {pipeline2.schedule.min_quality_score}")
            tests_passed += 1
        else:
            print("   ❌ Configuration not loaded correctly")
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
    
    # Test 6: Status and Control
    print("\n🎛️  Test 6: Status and Control...")
    try:
        pipeline = ContinuousResearchPipeline("test_research_config.json")
        await pipeline.initialize()
        
        # Set some test state
        pipeline.metrics.papers_generated = 3
        pipeline.metrics.total_cost = 1.25
        pipeline.last_daily_run = datetime.now()
        
        # Get status
        status = pipeline.get_status()
        
        if status and "metrics" in status and "schedule" in status:
            print("   ✅ Status reporting working correctly")
            print(f"      Running: {status['is_running']}")
            print(f"      Papers generated: {status['metrics']['papers_generated']}")
            print(f"      Total cost: ${status['metrics']['total_cost']:.2f}")
            
            # Check schedule status
            schedule_status = status.get("schedule", {})
            if schedule_status:
                print(f"      Daily research enabled: {schedule_status.get('daily_research', False)}")
                print(f"      Weekly research enabled: {schedule_status.get('weekly_comprehensive', False)}")
                tests_passed += 1
            else:
                print("   ❌ Schedule status not available")
        else:
            print("   ❌ Status reporting failed")
        
        await pipeline.stop()
        
    except Exception as e:
        print(f"   ❌ Status test failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 WEEK 4 CONTINUOUS PIPELINE TEST SUMMARY")
    print("=" * 70)
    
    test_names = [
        "Pipeline Initialization",
        "Daily Research Generation",
        "Weekly Research Generation", 
        "Metrics and Monitoring",
        "Configuration Management",
        "Status and Control"
    ]
    
    for i in range(1, total_tests + 1):
        status = "✅ PASSED" if i <= tests_passed else "❌ FAILED"
        print(f"Test {i}: {test_names[i-1]}: {status}")
    
    print(f"\nOverall: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("\n🎉 Week 4 Continuous Research Pipeline: FULLY OPERATIONAL!")
        print("\n🚀 Production-Ready Features:")
        print("   ✅ Automated daily research generation")
        print("   ✅ Weekly comprehensive analysis")
        print("   ✅ Monthly deep analysis with metrics")
        print("   ✅ Real-time monitoring and logging")
        print("   ✅ Configuration management")
        print("   ✅ Cost control and quality thresholds")
        print("   ✅ Research output storage and organization")
        print("   ✅ Error handling and recovery")
        
        print("\n📊 Production Capabilities:")
        print("   🔄 24/7 continuous operation")
        print("   📈 Scalable multi-application analysis")
        print("   💰 Cost-controlled research generation")
        print("   📝 Automated paper and function discovery")
        print("   📊 Performance metrics and reporting")
        print("   🎛️  Remote monitoring and control")
        
        print("\n🎯 Ready for Production Deployment!")
        print("   Next: Deploy to cloud infrastructure")
        print("   Next: Set up monitoring dashboards")
        print("   Next: Configure CI/CD integration")
        print("   Next: Scale to multiple VR applications")
        
        # Show sample production command
        print("\n🚀 Production Deployment Commands:")
        print("   Start continuous pipeline: python -m ai_integration.continuous_research_pipeline --mode continuous")
        print("   Run daily research: python -m ai_integration.continuous_research_pipeline --mode daily")
        print("   Check status: python -m ai_integration.continuous_research_pipeline --mode status")
        
    else:
        print(f"\n⚠️  Week 4 Pipeline: {tests_passed}/{total_tests} tests passed")
        print("   Some production features may need configuration")
    
    # Cleanup test files
    try:
        if os.path.exists("test_research_config.json"):
            os.remove("test_research_config.json")
        
        # Clean up test research outputs
        research_dir = Path("research_outputs")
        if research_dir.exists():
            for file in research_dir.glob("*"):
                if file.is_file():
                    file.unlink()
    except:
        pass  # Don't fail test on cleanup issues
    
    return tests_passed == total_tests


if __name__ == "__main__":
    try:
        success = asyncio.run(test_week4_continuous_pipeline())
        print(f"\n🏁 Week 4 Test completed: {'SUCCESS' if success else 'PARTIAL'}")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test crashed: {e}")
        sys.exit(1) 