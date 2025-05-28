#!/usr/bin/env python3
"""
Populate CloudVR-PerfGuard Database with Realistic Test Data
Creates realistic VR performance test data for Week 3 real data integration testing
"""

import asyncio
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.database import DatabaseManager


class TestDataGenerator:
    """Generate realistic VR performance test data"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        
        # VR Applications to simulate
        self.vr_apps = [
            {
                "name": "VRExplorer",
                "versions": ["1.0.0", "1.0.1", "1.1.0", "1.1.1"],
                "complexity": "medium",
                "base_fps": 85
            },
            {
                "name": "SpatialWorkshop", 
                "versions": ["2.3.0", "2.3.1", "2.4.0"],
                "complexity": "high",
                "base_fps": 75
            },
            {
                "name": "VRTraining",
                "versions": ["0.9.0", "1.0.0", "1.0.1"],
                "complexity": "low",
                "base_fps": 90
            },
            {
                "name": "MetaverseClient",
                "versions": ["3.1.0", "3.1.1", "3.2.0"],
                "complexity": "very_high", 
                "base_fps": 70
            }
        ]
        
        # GPU types and their characteristics
        self.gpu_types = {
            "RTX4090": {"vram": 24000, "performance_multiplier": 1.2},
            "RTX4080": {"vram": 16000, "performance_multiplier": 1.0},
            "RTX4070": {"vram": 12000, "performance_multiplier": 0.85},
            "RTX3080": {"vram": 10000, "performance_multiplier": 0.9}
        }
        
        # Scene types and complexity
        self.scene_types = [
            {"name": "simple_room", "complexity": 0.3, "objects": 50},
            {"name": "office_environment", "complexity": 0.5, "objects": 150},
            {"name": "outdoor_landscape", "complexity": 0.7, "objects": 300},
            {"name": "complex_city", "complexity": 0.9, "objects": 500},
            {"name": "stress_test", "complexity": 1.0, "objects": 1000}
        ]
    
    async def initialize(self):
        """Initialize the database"""
        await self.db_manager.initialize()
        print("✅ Test data generator initialized")
    
    def generate_realistic_metrics(
        self, 
        app: Dict[str, Any], 
        gpu_type: str, 
        scene: Dict[str, Any],
        test_duration: float
    ) -> Dict[str, Any]:
        """Generate realistic VR performance metrics"""
        
        gpu_info = self.gpu_types[gpu_type]
        base_fps = app["base_fps"] * gpu_info["performance_multiplier"]
        
        # Scene complexity affects performance
        complexity_factor = 1.0 - (scene["complexity"] * 0.3)
        target_fps = base_fps * complexity_factor
        
        # Add some realistic variance
        fps_variance = random.uniform(0.85, 1.15)
        avg_fps = target_fps * fps_variance
        
        # Frame time calculation
        avg_frame_time = 1000.0 / avg_fps if avg_fps > 0 else 16.67
        
        # GPU utilization based on scene complexity and target FPS
        base_gpu_util = 60 + (scene["complexity"] * 30)
        gpu_util_variance = random.uniform(0.9, 1.1)
        avg_gpu_util = min(95, base_gpu_util * gpu_util_variance)
        
        # VRAM usage based on scene objects and GPU capacity
        base_vram = 4000 + (scene["objects"] * 8)
        vram_variance = random.uniform(0.8, 1.2)
        max_vram_usage = min(gpu_info["vram"] * 0.9, base_vram * vram_variance)
        
        # CPU utilization (VR apps are typically GPU-bound)
        avg_cpu_util = random.uniform(25, 60)
        
        # VR comfort score (higher FPS and lower frame time variance = better comfort)
        frame_time_consistency = 1.0 / (1.0 + (avg_frame_time * 0.01))
        fps_consistency = min(1.0, avg_fps / 90.0)  # 90 FPS is ideal for VR
        vr_comfort_score = (frame_time_consistency + fps_consistency) * 50
        
        # Frame time standard deviation (consistency metric)
        frame_time_std = random.uniform(0.5, 3.0) * (1.0 + scene["complexity"])
        
        return {
            "avg_fps": round(avg_fps, 1),
            "min_fps": round(avg_fps * random.uniform(0.7, 0.9), 1),
            "max_fps": round(avg_fps * random.uniform(1.1, 1.3), 1),
            "avg_frame_time": round(avg_frame_time, 2),
            "min_frame_time": round(avg_frame_time * random.uniform(0.7, 0.9), 2),
            "max_frame_time": round(avg_frame_time * random.uniform(1.1, 1.5), 2),
            "frame_time_std": round(frame_time_std, 2),
            "avg_gpu_util": round(avg_gpu_util, 1),
            "max_gpu_util": round(min(100, avg_gpu_util * 1.2), 1),
            "max_vram_usage": round(max_vram_usage, 1),
            "avg_cpu_util": round(avg_cpu_util, 1),
            "max_cpu_util": round(avg_cpu_util * 1.4, 1),
            "vr_comfort_score": round(vr_comfort_score, 1),
            "test_duration": test_duration,
            "scene_complexity": scene["complexity"],
            "scene_objects": scene["objects"]
        }
    
    async def create_test_job_with_results(
        self,
        app: Dict[str, Any],
        version: str,
        submission_type: str = "performance_test",
        baseline_version: str = None,
        num_individual_tests: int = 5
    ) -> str:
        """Create a complete test job with realistic results"""
        
        job_id = str(uuid.uuid4())
        
        # Create test job
        test_config = {
            "test_type": "vr_performance",
            "duration_per_test": 120,
            "num_tests": num_individual_tests,
            "gpu_monitoring": True,
            "comfort_analysis": True
        }
        
        build_path = f"/builds/{app['name']}/{version}/{app['name']}_v{version}.exe"
        
        success = await self.db_manager.create_test_job(
            job_id=job_id,
            app_name=app["name"],
            build_version=version,
            platform="Windows",
            submission_type=submission_type,
            build_path=build_path,
            test_config=test_config,
            baseline_version=baseline_version
        )
        
        if not success:
            print(f"❌ Failed to create test job {job_id}")
            return None
        
        # Update to running status
        await self.db_manager.update_job_status(job_id, "running", progress=0)
        
        # Generate individual test results
        individual_results = []
        total_duration = 0
        
        for i in range(num_individual_tests):
            # Random GPU and scene for each test
            gpu_type = random.choice(list(self.gpu_types.keys()))
            scene = random.choice(self.scene_types)
            test_duration = random.uniform(100, 140)  # 100-140 seconds
            
            metrics = self.generate_realistic_metrics(app, gpu_type, scene, test_duration)
            
            individual_result = {
                "test_id": f"{job_id}_test_{i+1}",
                "success": True,
                "config": {
                    "scene_name": scene["name"],
                    "test_duration": test_duration,
                    "gpu_type": gpu_type,
                    "resolution": "2880x1700",  # Typical VR resolution
                    "refresh_rate": 90
                },
                "metrics": metrics,
                "timestamp": (datetime.utcnow() - timedelta(minutes=random.randint(1, 60))).isoformat()
            }
            
            individual_results.append(individual_result)
            total_duration += test_duration
        
        # Calculate aggregated metrics
        aggregated_metrics = self._calculate_aggregated_metrics(individual_results)
        
        # Store performance results
        performance_results = {
            "test_id": job_id,
            "build_path": build_path,
            "config": test_config,
            "total_duration": total_duration,
            "individual_results": individual_results,
            "aggregated_metrics": aggregated_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.db_manager.store_performance_results(job_id, performance_results)
        
        # Update to completed status
        await self.db_manager.update_job_status(job_id, "completed", progress=100)
        
        print(f"✅ Created test job {job_id} for {app['name']} v{version}")
        return job_id
    
    def _calculate_aggregated_metrics(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregated metrics from individual test results"""
        
        if not individual_results:
            return {}
        
        # Extract all metrics
        all_fps = [r["metrics"]["avg_fps"] for r in individual_results]
        all_frame_times = [r["metrics"]["avg_frame_time"] for r in individual_results]
        all_gpu_utils = [r["metrics"]["avg_gpu_util"] for r in individual_results]
        all_comfort_scores = [r["metrics"]["vr_comfort_score"] for r in individual_results]
        
        return {
            "overall_avg_fps": round(sum(all_fps) / len(all_fps), 1),
            "overall_min_fps": round(min(all_fps), 1),
            "overall_max_fps": round(max(all_fps), 1),
            "overall_avg_frame_time": round(sum(all_frame_times) / len(all_frame_times), 2),
            "overall_avg_gpu_util": round(sum(all_gpu_utils) / len(all_gpu_utils), 1),
            "overall_comfort_score": round(sum(all_comfort_scores) / len(all_comfort_scores), 1),
            "test_count": len(individual_results),
            "success_rate": 1.0,  # All tests successful in this simulation
            "total_test_time": sum(r["config"]["test_duration"] for r in individual_results)
        }
    
    async def populate_realistic_data(self, num_tests_per_app: int = 15) -> Dict[str, Any]:
        """Populate database with realistic test data"""
        
        print(f"🔄 Populating database with realistic VR test data...")
        print(f"   Apps: {len(self.vr_apps)}")
        print(f"   Tests per app: {num_tests_per_app}")
        
        created_jobs = []
        
        for app in self.vr_apps:
            print(f"\n📱 Creating tests for {app['name']}...")
            
            # Create baseline tests for each version
            for version in app["versions"]:
                # Create baseline
                job_id = await self.create_test_job_with_results(
                    app=app,
                    version=version,
                    submission_type="baseline",
                    num_individual_tests=random.randint(3, 7)
                )
                if job_id:
                    created_jobs.append(job_id)
                
                # Create performance tests
                for _ in range(num_tests_per_app // len(app["versions"])):
                    job_id = await self.create_test_job_with_results(
                        app=app,
                        version=version,
                        submission_type="performance_test",
                        baseline_version=app["versions"][0],  # Use first version as baseline
                        num_individual_tests=random.randint(4, 8)
                    )
                    if job_id:
                        created_jobs.append(job_id)
        
        # Create some recent tests (last 7 days)
        print(f"\n🕒 Creating recent tests...")
        for app in self.vr_apps[:2]:  # Just first 2 apps for recent tests
            latest_version = app["versions"][-1]
            for _ in range(5):
                job_id = await self.create_test_job_with_results(
                    app=app,
                    version=latest_version,
                    submission_type="performance_test",
                    num_individual_tests=random.randint(5, 10)
                )
                if job_id:
                    created_jobs.append(job_id)
        
        summary = {
            "total_jobs_created": len(created_jobs),
            "apps_populated": [app["name"] for app in self.vr_apps],
            "job_ids": created_jobs,
            "population_time": datetime.utcnow().isoformat()
        }
        
        print(f"\n✅ Database population completed!")
        print(f"   Total jobs created: {len(created_jobs)}")
        print(f"   Apps with data: {len(self.vr_apps)}")
        
        return summary
    
    async def close(self):
        """Close the database connection"""
        await self.db_manager.close()


async def main():
    """Main function to populate test data"""
    
    print("🚀 CloudVR-PerfGuard Test Data Population")
    print("=" * 60)
    
    generator = TestDataGenerator()
    await generator.initialize()
    
    try:
        # Populate with realistic data
        summary = await generator.populate_realistic_data(num_tests_per_app=12)
        
        print(f"\n📊 Population Summary:")
        print(f"   Jobs created: {summary['total_jobs_created']}")
        print(f"   Apps: {', '.join(summary['apps_populated'])}")
        print(f"   Ready for Week 3 real data testing!")
        
        return summary
        
    finally:
        await generator.close()


if __name__ == "__main__":
    asyncio.run(main()) 