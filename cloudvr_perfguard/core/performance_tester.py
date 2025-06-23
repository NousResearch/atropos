"""
VR Performance Tester - Core module for running VR performance tests
Adapted from AMIEN's parallel execution patterns
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import psutil

from .container_manager import ContainerManager
from .gpu_monitor import GPUMonitor

# PerformanceTestConfig will be imported from api.main when needed


class VRPerformanceTester:
    """
    Core VR performance testing engine
    Adapted from AMIEN's run_1000_experiments.py patterns
    """

    def __init__(self):
        self.gpu_monitor = None
        self.container_manager = None
        self.supported_gpu_types = ["T4", "L4", "A100"]
        self.test_scenes = {
            "main_menu": {"complexity": "low", "duration": 30},
            "gameplay_scene": {"complexity": "medium", "duration": 60},
            "stress_test": {"complexity": "high", "duration": 120},
        }

    async def initialize(self):
        """Initialize performance testing components"""
        try:
            self.gpu_monitor = GPUMonitor()
            await self.gpu_monitor.initialize()

            self.container_manager = ContainerManager()
            await self.container_manager.initialize()

            print("INFO: VRPerformanceTester initialized successfully")

        except Exception as e:
            print(f"ERROR: Failed to initialize VRPerformanceTester: {e}")
            raise

    async def run_test(self, build_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run performance test on VR build
        Adapted from run_1000_experiments.py parallel execution pattern
        """

        print(f"🔬 Starting VR performance test for build: {build_path}")
        test_start_time = time.time()

        # Generate test configurations (like target variations in AMIEN)
        test_configs = self._generate_test_configurations(config)

        # Run tests in parallel (adapted from AMIEN's batch execution)
        results = await self._run_parallel_tests(build_path, test_configs)

        # Aggregate results (like AMIEN's analysis functions)
        aggregated_results = self._aggregate_test_results(results)

        total_time = time.time() - test_start_time

        print(f"✅ Performance test completed in {total_time:.1f}s")
        print(f"   Tests run: {len(results)}")
        print(f"   Success rate: {aggregated_results['success_rate']*100:.1f}%")

        return {
            "test_id": f"perf_test_{int(time.time())}",
            "build_path": build_path,
            "config": config.dict(),
            "total_duration": total_time,
            "individual_results": results,
            "aggregated_metrics": aggregated_results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _generate_test_configurations(
        self, config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate test configurations for parallel execution
        Similar to AMIEN's target position variations
        """

        test_configs = []

        for gpu_type in config.get("gpu_types", ["T4"]):
            for scene_name in config.get("test_scenes", ["main_menu"]):
                if scene_name not in self.test_scenes:
                    print("WARNING: Unknown test scene "{scene_name}', skipping")
                    continue

                scene_config = self.test_scenes[scene_name]

                test_config = {
                    "gpu_type": gpu_type,
                    "scene_name": scene_name,
                    "scene_complexity": scene_config["complexity"],
                    "test_duration": min(
                        config.get("test_duration_seconds", 60),
                        scene_config["duration"],
                    ),
                    "resolution": "2160x1200",  # Standard VR resolution
                    "refresh_rate": 90,  # Standard VR refresh rate
                }

                test_configs.append(test_config)

        print(f"📊 Generated {len(test_configs)} test configurations")
        return test_configs

    async def _run_parallel_tests(
        self, build_path: str, test_configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run performance tests in parallel
        Adapted from AMIEN's run_batch pattern
        """

        print(f"🚀 Running {len(test_configs)} performance tests in parallel")

        # Create semaphore to limit concurrent tests (avoid overwhelming GPU)
        semaphore = asyncio.Semaphore(2)  # Max 2 concurrent GPU tests

        # Create tasks for parallel execution
        tasks = []
        for i, test_config in enumerate(test_configs):
            task = self._run_single_performance_test(
                semaphore, build_path, test_config, i + 1
            )
            tasks.append(task)

        # Execute all tests in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Test {i+1} failed: {result}")
                processed_results.append(
                    {
                        "test_id": f"test_{i+1}",
                        "config": test_configs[i],
                        "success": False,
                        "error": str(result),
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _run_single_performance_test(
        self,
        semaphore: asyncio.Semaphore,
        build_path: str,
        test_config: Dict[str, Any],
        test_num: int,
    ) -> Dict[str, Any]:
        """
        Run single performance test
        Adapted from AMIEN's run_single_experiment pattern
        """

        async with semaphore:
            print(
                f"  🧪 Test {test_num}: {test_config['gpu_type']} + {test_config['scene_name']}"
            )

            try:
                # Setup test environment (like AMIEN's setup_environment)
                container_id = await self._setup_test_environment(
                    build_path, test_config
                )

                # Run performance measurement (like AMIEN's execute_action)
                performance_data = await self._measure_performance(
                    container_id, test_config
                )

                # Cleanup
                await self._cleanup_test_environment(container_id)

                # Calculate metrics (like AMIEN's calculate_performance_metrics)
                metrics = self._calculate_performance_metrics(
                    performance_data, test_config
                )

                result = {
                    "test_id": f"test_{test_num}",
                    "config": test_config,
                    "performance_data": performance_data,
                    "metrics": metrics,
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                print(f"    ✅ Test {test_num}: {metrics.get('avg_fps', 0):.1f} FPS")
                return result

            except Exception as e:
                print(f"    ❌ Test {test_num} failed: {e}")
                return {
                    "test_id": f"test_{test_num}",
                    "config": test_config,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }

    async def _setup_test_environment(
        self, build_path: str, test_config: Dict[str, Any]
    ) -> str:
        """Setup containerized test environment for VR app"""

        # Create container with specific GPU type
        container_config = {
            "gpu_type": test_config["gpu_type"],
            "platform": "windows",  # TODO: Make configurable
            "vr_runtime": "steamvr",  # TODO: Make configurable
            "build_path": build_path,
        }

        container_id = await self.container_manager.create_container(container_config)

        # Wait for container to be ready
        await self.container_manager.wait_for_ready(container_id)

        return container_id

    async def _measure_performance(
        self, container_id: str, test_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Measure VR performance metrics
        Similar to AMIEN's physics data collection
        """

        duration = test_config["test_duration"]
        scene_name = test_config["scene_name"]

        print(f"    📊 Measuring performance for {duration}s on scene '{scene_name}'")

        # Start performance monitoring
        monitor_task = asyncio.create_task(
            self.gpu_monitor.monitor_performance(container_id, duration)
        )

        # Launch VR app in container
        app_task = asyncio.create_task(
            self.container_manager.run_vr_app(container_id, scene_name, duration)
        )

        # Wait for both to complete
        monitor_data, app_data = await asyncio.gather(monitor_task, app_task)

        return {
            "gpu_metrics": monitor_data,
            "app_metrics": app_data,
            "test_duration": duration,
            "scene_name": scene_name,
        }

    async def _cleanup_test_environment(self, container_id: str):
        """Cleanup test environment"""
        await self.container_manager.destroy_container(container_id)

    def _calculate_performance_metrics(
        self, performance_data: Dict[str, Any], test_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics from raw data
        Adapted from AMIEN's calculate_performance_metrics pattern
        """

        gpu_metrics = performance_data.get("gpu_metrics", {})
        app_metrics = performance_data.get("app_metrics", {})

        # Extract frame timing data
        frame_times = app_metrics.get("frame_times", [])
        if not frame_times:
            frame_times = [16.67] * 60  # Fallback: 60 FPS for 1 second

        # Calculate FPS metrics
        fps_values = [1000.0 / ft for ft in frame_times if ft > 0]

        metrics = {
            # Frame rate metrics
            "avg_fps": np.mean(fps_values) if fps_values else 0,
            "min_fps": np.min(fps_values) if fps_values else 0,
            "max_fps": np.max(fps_values) if fps_values else 0,
            "fps_std": np.std(fps_values) if fps_values else 0,
            "fps_p1": np.percentile(fps_values, 1) if fps_values else 0,
            "fps_p99": np.percentile(fps_values, 99) if fps_values else 0,
            # Frame time metrics (critical for VR)
            "avg_frame_time": np.mean(frame_times) if frame_times else 0,
            "p99_frame_time": np.percentile(frame_times, 99) if frame_times else 0,
            "frame_time_std": np.std(frame_times) if frame_times else 0,
            # GPU metrics
            "avg_gpu_util": np.mean(gpu_metrics.get("gpu_utilization", [0])),
            "max_gpu_util": np.max(gpu_metrics.get("gpu_utilization", [0])),
            "avg_vram_usage": np.mean(gpu_metrics.get("vram_usage_mb", [0])),
            "max_vram_usage": np.max(gpu_metrics.get("vram_usage_mb", [0])),
            # CPU metrics
            "avg_cpu_util": np.mean(gpu_metrics.get("cpu_utilization", [0])),
            "max_cpu_util": np.max(gpu_metrics.get("cpu_utilization", [0])),
            # VR-specific metrics
            "dropped_frames": app_metrics.get("dropped_frames", 0),
            "reprojected_frames": app_metrics.get("reprojected_frames", 0),
            "motion_to_photon_latency": app_metrics.get("motion_to_photon_ms", 0),
            # Quality metrics
            "vr_comfort_score": self._calculate_vr_comfort_score(frame_times),
            "performance_grade": self._calculate_performance_grade(fps_values),
        }

        return metrics

    def _calculate_vr_comfort_score(self, frame_times: List[float]) -> float:
        """Calculate VR comfort score based on frame consistency"""
        if not frame_times:
            return 0.0

        # VR comfort depends on consistent frame times
        target_frame_time = 11.11  # 90 FPS target
        deviations = [abs(ft - target_frame_time) for ft in frame_times]
        avg_deviation = np.mean(deviations)

        # Score from 0-100, where 100 is perfect consistency
        comfort_score = max(0, 100 - (avg_deviation * 10))
        return comfort_score

    def _calculate_performance_grade(self, fps_values: List[float]) -> str:
        """Calculate performance grade (A-F) based on VR standards"""
        if not fps_values:
            return "F"

        avg_fps = np.mean(fps_values)
        min_fps = np.min(fps_values)

        # VR performance grading
        if avg_fps >= 90 and min_fps >= 80:
            return "A"  # Excellent VR performance
        elif avg_fps >= 80 and min_fps >= 70:
            return "B"  # Good VR performance
        elif avg_fps >= 70 and min_fps >= 60:
            return "C"  # Acceptable VR performance
        elif avg_fps >= 60 and min_fps >= 45:
            return "D"  # Poor VR performance
        else:
            return "F"  # Unacceptable VR performance

    def _aggregate_test_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple tests
        Adapted from AMIEN's analyze_progress pattern
        """

        successful_results = [r for r in results if r.get("success", False)]
        success_rate = len(successful_results) / len(results) if results else 0

        if not successful_results:
            return {
                "success_rate": success_rate,
                "total_tests": len(results),
                "successful_tests": 0,
                "error": "No successful tests to aggregate",
            }

        # Aggregate metrics across all successful tests
        all_metrics = [r["metrics"] for r in successful_results]

        aggregated = {
            "success_rate": success_rate,
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            # FPS aggregation
            "overall_avg_fps": np.mean([m["avg_fps"] for m in all_metrics]),
            "overall_min_fps": np.min([m["min_fps"] for m in all_metrics]),
            "overall_max_fps": np.max([m["max_fps"] for m in all_metrics]),
            # Frame time aggregation
            "overall_avg_frame_time": np.mean(
                [m["avg_frame_time"] for m in all_metrics]
            ),
            "overall_p99_frame_time": np.max(
                [m["p99_frame_time"] for m in all_metrics]
            ),
            # GPU utilization
            "overall_avg_gpu_util": np.mean([m["avg_gpu_util"] for m in all_metrics]),
            "overall_max_vram": np.max([m["max_vram_usage"] for m in all_metrics]),
            # VR quality
            "overall_comfort_score": np.mean(
                [m["vr_comfort_score"] for m in all_metrics]
            ),
            "performance_grades": [m["performance_grade"] for m in all_metrics],
            # Per-GPU breakdown
            "gpu_breakdown": self._create_gpu_breakdown(successful_results),
        }

        return aggregated

    def _create_gpu_breakdown(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create per-GPU performance breakdown"""

        gpu_breakdown = {}

        for result in results:
            gpu_type = result["config"]["gpu_type"]
            if gpu_type not in gpu_breakdown:
                gpu_breakdown[gpu_type] = []

            gpu_breakdown[gpu_type].append(result["metrics"])

        # Aggregate per GPU type
        for gpu_type, metrics_list in gpu_breakdown.items():
            gpu_breakdown[gpu_type] = {
                "test_count": len(metrics_list),
                "avg_fps": np.mean([m["avg_fps"] for m in metrics_list]),
                "avg_frame_time": np.mean([m["avg_frame_time"] for m in metrics_list]),
                "avg_gpu_util": np.mean([m["avg_gpu_util"] for m in metrics_list]),
                "comfort_score": np.mean([m["vr_comfort_score"] for m in metrics_list]),
            }

        return gpu_breakdown

    async def cleanup(self):
        """Cleanup resources"""
        if self.gpu_monitor:
            await self.gpu_monitor.cleanup()
        if self.container_manager:
            await self.container_manager.cleanup()
        print("INFO: VRPerformanceTester cleanup complete")
