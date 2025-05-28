#!/usr/bin/env python3
"""
AMIEN Complete Integration Test Suite
Comprehensive testing for continuous integration and system validation
"""

import asyncio
import sys
import os
import json
import time
import unittest
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import subprocess

# Add paths
sys.path.append('.')

class AMIENIntegrationTestSuite:
    """Complete integration test suite for AMIEN system"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        
        print("🧪 AMIEN Complete Integration Test Suite")
        print("=" * 60)
        print(f"   Start Time: {self.start_time.isoformat()}")
        print(f"   Test Data Directory: {self.test_data_dir}")
    
    async def test_core_integration_manager(self):
        """Test the core AMIEN integration manager"""
        print("\n🔬 Testing Core Integration Manager...")
        
        try:
            from restart_amien_integration import AMIENIntegrationManager
            
            # Initialize manager
            manager = AMIENIntegrationManager()
            
            # Test synthetic data generation
            test_data = await manager.generate_synthetic_vr_data(10)
            assert len(test_data) == 10, "Should generate 10 experiments"
            assert all("user_id" in exp for exp in test_data), "All experiments should have user_id"
            
            # Test AI Scientist integration
            paper_result = await manager.run_ai_scientist_integration(test_data[:5])
            assert "quality_score" in paper_result, "Should return quality score"
            assert paper_result["quality_score"] > 0, "Quality score should be positive"
            
            # Test FunSearch integration
            function_result = await manager.run_funsearch_integration(test_data[:5])
            assert "fitness" in function_result, "Should return fitness score"
            assert function_result["fitness"] >= 0, "Fitness should be non-negative"
            
            self.test_results["core_integration"] = {
                "status": "PASS",
                "data_generation": len(test_data),
                "paper_quality": paper_result.get("quality_score", 0),
                "function_fitness": function_result.get("fitness", 0)
            }
            print("   ✅ Core Integration Manager: PASS")
            
        except Exception as e:
            self.test_results["core_integration"] = {
                "status": "FAIL",
                "error": str(e)
            }
            print(f"   ❌ Core Integration Manager: FAIL - {e}")
    
    async def test_massive_scale_runner(self):
        """Test the massive scale experiment runner"""
        print("\n📊 Testing Massive Scale Runner...")
        
        try:
            from scale_to_production import MassiveScaleExperimentRunner
            
            # Initialize with small scale for testing
            runner = MassiveScaleExperimentRunner(num_users=50, num_environments=5)
            
            # Test user generation
            users = await runner.generate_synthetic_users()
            assert len(users) == 50, "Should generate 50 users"
            assert all("user_id" in user for user in users), "All users should have user_id"
            
            # Test environment generation
            environments = await runner.generate_vr_environments()
            assert len(environments) == 5, "Should generate 5 environments"
            assert all("environment_id" in env for env in environments), "All environments should have environment_id"
            
            # Test experiment execution
            results = await runner.run_massive_scale_experiments(sample_size=20)
            assert len(results) == 20, "Should run 20 experiments"
            assert all("success" in result for result in results), "All results should have success flag"
            
            success_rate = sum(1 for r in results if r["success"]) / len(results)
            
            self.test_results["massive_scale"] = {
                "status": "PASS",
                "users_generated": len(users),
                "environments_generated": len(environments),
                "experiments_run": len(results),
                "success_rate": success_rate
            }
            print("   ✅ Massive Scale Runner: PASS")
            
        except Exception as e:
            self.test_results["massive_scale"] = {
                "status": "FAIL",
                "error": str(e)
            }
            print(f"   ❌ Massive Scale Runner: FAIL - {e}")
    
    async def test_ai_scientist_manager(self):
        """Test the AI Scientist manager"""
        print("\n🤖 Testing AI Scientist Manager...")
        
        try:
            # Test with the main integration manager instead
            from restart_amien_integration import AMIENIntegrationManager
            
            # Initialize manager
            manager = AMIENIntegrationManager()
            
            # Test paper generation
            test_experiments = [
                {"user_id": f"test_user_{i}", "comfort_score": 0.5 + i * 0.1, "fps": 60 + i * 5}
                for i in range(5)
            ]
            
            paper_result = await manager.run_ai_scientist_integration(test_experiments)
            assert "quality_score" in paper_result, "Should return quality score"
            assert paper_result["quality_score"] > 0, "Quality score should be positive"
            
            self.test_results["ai_scientist"] = {
                "status": "PASS",
                "quality_score": paper_result.get("quality_score", 0),
                "cost": paper_result.get("cost", 0)
            }
            print("   ✅ AI Scientist Manager: PASS")
            
        except Exception as e:
            self.test_results["ai_scientist"] = {
                "status": "FAIL",
                "error": str(e)
            }
            print(f"   ❌ AI Scientist Manager: FAIL - {e}")
    
    async def test_funsearch_manager(self):
        """Test the FunSearch manager"""
        print("\n🔍 Testing FunSearch Manager...")
        
        try:
            # Test with the main integration manager instead
            from restart_amien_integration import AMIENIntegrationManager
            
            # Initialize manager
            manager = AMIENIntegrationManager()
            
            # Test function evolution
            test_experiments = [
                {"user_id": f"test_user_{i}", "fps": 60 + i * 5, "comfort_score": 0.5 + i * 0.1}
                for i in range(5)
            ]
            
            function_result = await manager.run_funsearch_integration(test_experiments)
            assert "fitness" in function_result, "Should return fitness score"
            assert function_result["fitness"] >= 0, "Fitness should be non-negative"
            
            self.test_results["funsearch"] = {
                "status": "PASS",
                "fitness_score": function_result.get("fitness", 0),
                "cost": function_result.get("cost", 0)
            }
            print("   ✅ FunSearch Manager: PASS")
            
        except Exception as e:
            self.test_results["funsearch"] = {
                "status": "FAIL",
                "error": str(e)
            }
            print(f"   ❌ FunSearch Manager: FAIL - {e}")
    
    async def test_advanced_analytics(self):
        """Test the advanced analytics system"""
        print("\n📈 Testing Advanced Analytics...")
        
        try:
            from advanced_analytics_system import AMIENAdvancedAnalytics
            
            # Initialize analytics
            analytics = AMIENAdvancedAnalytics()
            
            # Test data loading
            await analytics.load_historical_data()
            assert len(analytics.experiment_history) > 0, "Should load experiment history"
            
            # Test dashboard generation
            dashboard = await analytics.generate_performance_dashboard()
            assert "system_health" in dashboard, "Should include system health"
            assert "research_productivity" in dashboard, "Should include research productivity"
            
            # Test recommendations
            recommendations = await analytics.generate_optimization_recommendations()
            assert "system_optimizations" in recommendations, "Should include system optimizations"
            
            self.test_results["analytics"] = {
                "status": "PASS",
                "experiments_loaded": len(analytics.experiment_history),
                "papers_loaded": len(analytics.paper_quality_history),
                "health_score": dashboard["system_health"].get("health_score", 0)
            }
            print("   ✅ Advanced Analytics: PASS")
            
        except Exception as e:
            self.test_results["analytics"] = {
                "status": "FAIL",
                "error": str(e)
            }
            print(f"   ❌ Advanced Analytics: FAIL - {e}")
    
    def test_deployment_readiness(self):
        """Test deployment readiness"""
        print("\n☁️ Testing Deployment Readiness...")
        
        try:
            # Check required files
            required_files = [
                "final_production_deployment.py",
                "deploy_ai_integration.sh",
                "advanced_analytics_system.py",
                "restart_amien_integration.py",
                "scale_to_production.py",
                "ai_scientist_manager.py",
                "funsearch_manager.py"
            ]
            
            missing_files = [f for f in required_files if not Path(f).exists()]
            
            # Check GCP deployment files
            gcp_files = [
                "gcp_deployment/main.tf",
                "gcp_deployment/main_api.py",
                "gcp_deployment/Dockerfile.api"
            ]
            
            missing_gcp_files = [f for f in gcp_files if not Path(f).exists()]
            
            # Check configuration files
            config_files = [
                "production.env",
                "requirements_production.txt",
                "DEPLOYMENT_CHECKLIST.md"
            ]
            
            missing_config_files = [f for f in config_files if not Path(f).exists()]
            
            all_files_present = not (missing_files or missing_gcp_files or missing_config_files)
            
            self.test_results["deployment_readiness"] = {
                "status": "PASS" if all_files_present else "FAIL",
                "core_files_present": len(required_files) - len(missing_files),
                "gcp_files_present": len(gcp_files) - len(missing_gcp_files),
                "config_files_present": len(config_files) - len(missing_config_files),
                "missing_files": missing_files + missing_gcp_files + missing_config_files
            }
            
            if all_files_present:
                print("   ✅ Deployment Readiness: PASS")
            else:
                print(f"   ❌ Deployment Readiness: FAIL - Missing files: {missing_files + missing_gcp_files + missing_config_files}")
                
        except Exception as e:
            self.test_results["deployment_readiness"] = {
                "status": "FAIL",
                "error": str(e)
            }
            print(f"   ❌ Deployment Readiness: FAIL - {e}")
    
    def test_environment_configuration(self):
        """Test environment configuration"""
        print("\n🔧 Testing Environment Configuration...")
        
        try:
            # Check environment variables
            required_env_vars = ["GEMINI_API_KEY"]
            optional_env_vars = ["OPENAI_API_KEY", "PERPLEXITY_API_KEY", "GCP_PROJECT_ID"]
            
            required_present = [var for var in required_env_vars if os.getenv(var)]
            optional_present = [var for var in optional_env_vars if os.getenv(var)]
            
            # Check Python dependencies
            try:
                import google.generativeai
                import fastapi
                import uvicorn
                import numpy
                import pandas
                import matplotlib
                dependencies_ok = True
            except ImportError as e:
                dependencies_ok = False
                dependency_error = str(e)
            
            self.test_results["environment"] = {
                "status": "PASS" if len(required_present) == len(required_env_vars) and dependencies_ok else "WARN",
                "required_env_vars": required_present,
                "optional_env_vars": optional_present,
                "dependencies_ok": dependencies_ok,
                "dependency_error": dependency_error if not dependencies_ok else None
            }
            
            if len(required_present) == len(required_env_vars) and dependencies_ok:
                print("   ✅ Environment Configuration: PASS")
            else:
                print("   ⚠️ Environment Configuration: WARN - Some optional components missing")
                
        except Exception as e:
            self.test_results["environment"] = {
                "status": "FAIL",
                "error": str(e)
            }
            print(f"   ❌ Environment Configuration: FAIL - {e}")
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        print("\n⚡ Testing Performance Benchmarks...")
        
        try:
            # Test data generation speed
            start_time = time.time()
            from restart_amien_integration import AMIENIntegrationManager
            manager = AMIENIntegrationManager()
            test_data = await manager.generate_synthetic_vr_data(100)
            data_gen_time = time.time() - start_time
            
            # Test processing speed
            start_time = time.time()
            paper_result = await manager.run_ai_scientist_integration(test_data[:10])
            processing_time = time.time() - start_time
            
            # Performance thresholds
            data_gen_ok = data_gen_time < 5.0  # Should generate 100 experiments in < 5 seconds
            processing_ok = processing_time < 30.0  # Should process 10 experiments in < 30 seconds
            
            self.test_results["performance"] = {
                "status": "PASS" if data_gen_ok and processing_ok else "WARN",
                "data_generation_time": data_gen_time,
                "processing_time": processing_time,
                "data_generation_rate": len(test_data) / data_gen_time,
                "processing_rate": 10 / processing_time
            }
            
            if data_gen_ok and processing_ok:
                print("   ✅ Performance Benchmarks: PASS")
            else:
                print("   ⚠️ Performance Benchmarks: WARN - Performance below optimal")
                
        except Exception as e:
            self.test_results["performance"] = {
                "status": "FAIL",
                "error": str(e)
            }
            print(f"   ❌ Performance Benchmarks: FAIL - {e}")
    
    async def run_complete_test_suite(self):
        """Run the complete test suite"""
        print("🧪 Running Complete AMIEN Integration Test Suite")
        print("=" * 60)
        
        # Run all tests
        await self.test_core_integration_manager()
        await self.test_massive_scale_runner()
        await self.test_ai_scientist_manager()
        await self.test_funsearch_manager()
        await self.test_advanced_analytics()
        self.test_deployment_readiness()
        self.test_environment_configuration()
        await self.test_performance_benchmarks()
        
        # Calculate overall results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASS")
        warned_tests = sum(1 for result in self.test_results.values() if result["status"] == "WARN")
        failed_tests = sum(1 for result in self.test_results.values() if result["status"] == "FAIL")
        
        # Determine overall status
        if failed_tests == 0 and warned_tests == 0:
            overall_status = "EXCELLENT"
        elif failed_tests == 0:
            overall_status = "GOOD"
        elif failed_tests <= total_tests * 0.2:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "NEEDS_ATTENTION"
        
        # Create test summary
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        summary = {
            "test_timestamp": self.start_time.isoformat(),
            "completion_timestamp": end_time.isoformat(),
            "duration_seconds": duration,
            "overall_status": overall_status,
            "test_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "warned": warned_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests) * 100
            },
            "detailed_results": self.test_results,
            "recommendations": self._generate_test_recommendations()
        }
        
        # Save test results
        test_file = self.test_data_dir / f"integration_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(test_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n🎉 Test Suite Complete!")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Overall Status: {overall_status}")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {(passed_tests / total_tests) * 100:.1f}%")
        print(f"   Results saved: {test_file}")
        
        # Print recommendations
        recommendations = summary["recommendations"]
        if recommendations:
            print(f"\n📋 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        return summary
    
    def _generate_test_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for failed tests
        failed_tests = [name for name, result in self.test_results.items() if result["status"] == "FAIL"]
        if failed_tests:
            recommendations.append(f"Fix failed tests: {', '.join(failed_tests)}")
        
        # Check for performance issues
        if "performance" in self.test_results:
            perf = self.test_results["performance"]
            if perf.get("data_generation_time", 0) > 3:
                recommendations.append("Optimize data generation performance")
            if perf.get("processing_time", 0) > 20:
                recommendations.append("Optimize AI processing performance")
        
        # Check for missing environment variables
        if "environment" in self.test_results:
            env = self.test_results["environment"]
            if len(env.get("required_env_vars", [])) == 0:
                recommendations.append("Set required environment variables (GEMINI_API_KEY)")
        
        # Check for deployment readiness
        if "deployment_readiness" in self.test_results:
            deploy = self.test_results["deployment_readiness"]
            if deploy.get("missing_files"):
                recommendations.append("Ensure all deployment files are present")
        
        return recommendations

async def main():
    """Main test function"""
    test_suite = AMIENIntegrationTestSuite()
    await test_suite.run_complete_test_suite()

if __name__ == "__main__":
    asyncio.run(main()) 