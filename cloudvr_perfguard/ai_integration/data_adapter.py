"""
Data Adapter for AI Research Integration
Converts CloudVR-PerfGuard performance data into formats suitable for AI tools
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime


class PerformanceDataAdapter:
    """
    Adapts CloudVR performance data for AI research tools
    """
    
    def __init__(self):
        self.supported_formats = ["ai_scientist", "funsearch", "csv", "json"]
    
    def to_ai_scientist_format(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert performance data to AI Scientist input format
        
        Args:
            performance_data: Raw performance test results
            
        Returns:
            Formatted data for AI Scientist paper generation
        """
        
        # Extract key metrics for research analysis
        individual_results = performance_data.get("individual_results", [])
        
        # Calculate statistical summaries
        fps_values = [r.get("metrics", {}).get("avg_fps", 0) for r in individual_results if r.get("success")]
        frame_times = [r.get("metrics", {}).get("avg_frame_time", 0) for r in individual_results if r.get("success")]
        comfort_scores = [r.get("metrics", {}).get("vr_comfort_score", 0) for r in individual_results if r.get("success")]
        
        # Prepare research data structure
        research_data = {
            "experiment_metadata": {
                "test_id": performance_data.get("test_id", "unknown"),
                "app_name": self._extract_app_name(performance_data.get("build_path", "")),
                "test_count": len(individual_results),
                "success_rate": len(fps_values) / len(individual_results) if individual_results else 0,
                "test_duration": performance_data.get("total_duration", 0),
                "timestamp": performance_data.get("timestamp", datetime.utcnow().isoformat())
            },
            "performance_metrics": {
                "fps_statistics": self._calculate_statistics(fps_values),
                "frame_time_statistics": self._calculate_statistics(frame_times),
                "comfort_score_statistics": self._calculate_statistics(comfort_scores),
                "gpu_breakdown": self._analyze_gpu_performance(individual_results)
            },
            "research_context": {
                "methodology": "automated_vr_performance_testing",
                "tools_used": ["CloudVR-PerfGuard"],
                "metrics_collected": ["fps", "frame_time", "gpu_utilization", "vr_comfort_score"],
                "test_environments": self._extract_test_environments(individual_results)
            },
            "raw_data": individual_results[:10]  # Include sample of raw data
        }
        
        return research_data
    
    def to_funsearch_format(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert performance data to FunSearch training format
        
        Args:
            performance_data: Raw performance test results
            
        Returns:
            Training data for function discovery
        """
        
        individual_results = performance_data.get("individual_results", [])
        
        # Prepare feature matrix (X) and target vector (y)
        features = []
        targets = []
        
        for result in individual_results:
            if not result.get("success"):
                continue
                
            metrics = result.get("metrics", {})
            config = result.get("config", {})
            
            # Extract input features
            feature_vector = [
                metrics.get("avg_gpu_util", 50.0),
                metrics.get("max_vram_usage", 1000.0) / 1000.0,  # Normalize to GB
                metrics.get("avg_cpu_util", 30.0),
                self._encode_scene_complexity(config.get("scene_name", "main_menu")),
                config.get("test_duration", 60.0) / 60.0,  # Normalize to minutes
                self._encode_gpu_type(config.get("gpu_type", "unknown"))
            ]
            
            # Extract target values for different optimization goals
            target_values = {
                "frame_time_consistency": 1.0 / (1.0 + metrics.get("frame_time_std", 1.0)),
                "comfort_score": metrics.get("vr_comfort_score", 50.0) / 100.0,
                "performance_efficiency": metrics.get("avg_fps", 60.0) / metrics.get("avg_gpu_util", 50.0)
            }
            
            features.append(feature_vector)
            targets.append(target_values)
        
        return {
            "features": np.array(features),
            "targets": targets,
            "feature_names": [
                "gpu_utilization", "vram_usage_gb", "cpu_utilization", 
                "scene_complexity", "test_duration_min", "gpu_type_encoded"
            ],
            "target_names": list(targets[0].keys()) if targets else [],
            "sample_count": len(features),
            "data_quality": {
                "completeness": len(features) / len(individual_results) if individual_results else 0,
                "feature_ranges": self._calculate_feature_ranges(features)
            }
        }
    
    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of values"""
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
        
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "count": len(values)
        }
    
    def _analyze_gpu_performance(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance breakdown by GPU type"""
        gpu_data = {}
        
        for result in individual_results:
            if not result.get("success"):
                continue
                
            gpu_type = result.get("config", {}).get("gpu_type", "unknown")
            metrics = result.get("metrics", {})
            
            if gpu_type not in gpu_data:
                gpu_data[gpu_type] = {"fps_values": [], "comfort_scores": []}
            
            gpu_data[gpu_type]["fps_values"].append(metrics.get("avg_fps", 0))
            gpu_data[gpu_type]["comfort_scores"].append(metrics.get("vr_comfort_score", 0))
        
        # Calculate statistics for each GPU type
        gpu_analysis = {}
        for gpu_type, data in gpu_data.items():
            gpu_analysis[gpu_type] = {
                "test_count": len(data["fps_values"]),
                "avg_fps": float(np.mean(data["fps_values"])) if data["fps_values"] else 0,
                "avg_comfort": float(np.mean(data["comfort_scores"])) if data["comfort_scores"] else 0
            }
        
        return gpu_analysis
    
    def _extract_app_name(self, build_path: str) -> str:
        """Extract application name from build path"""
        if not build_path:
            return "Unknown VR Application"
        
        # Extract filename without extension
        filename = build_path.split("/")[-1].split("\\")[-1]
        return filename.split(".")[0] if "." in filename else filename
    
    def _extract_test_environments(self, individual_results: List[Dict[str, Any]]) -> List[str]:
        """Extract unique test environments from results"""
        environments = set()
        
        for result in individual_results:
            config = result.get("config", {})
            scene = config.get("scene_name", "unknown")
            gpu = config.get("gpu_type", "unknown")
            environments.add(f"{scene}_{gpu}")
        
        return list(environments)
    
    def _encode_scene_complexity(self, scene_name: str) -> float:
        """Encode scene complexity as a numeric value"""
        complexity_map = {
            "main_menu": 1.0,
            "simple_scene": 2.0,
            "gameplay_scene": 3.0,
            "complex_scene": 4.0,
            "stress_test": 5.0
        }
        return complexity_map.get(scene_name.lower(), 2.5)
    
    def _encode_gpu_type(self, gpu_type: str) -> float:
        """Encode GPU type as a numeric value"""
        gpu_map = {
            "t4": 1.0,
            "l4": 2.0,
            "a100": 3.0,
            "v100": 2.5,
            "rtx3080": 2.2,
            "rtx4090": 3.5
        }
        return gpu_map.get(gpu_type.lower(), 2.0)
    
    def _calculate_feature_ranges(self, features: List[List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate ranges for feature normalization"""
        if not features:
            return {}
        
        features_array = np.array(features)
        feature_names = [
            "gpu_utilization", "vram_usage_gb", "cpu_utilization", 
            "scene_complexity", "test_duration_min", "gpu_type_encoded"
        ]
        
        ranges = {}
        for i, name in enumerate(feature_names):
            if i < features_array.shape[1]:
                column = features_array[:, i]
                ranges[name] = {
                    "min": float(np.min(column)),
                    "max": float(np.max(column)),
                    "mean": float(np.mean(column)),
                    "std": float(np.std(column))
                }
        
        return ranges
    
    def export_to_csv(self, performance_data: Dict[str, Any], filename: str) -> str:
        """Export performance data to CSV format"""
        import csv
        
        individual_results = performance_data.get("individual_results", [])
        
        with open(filename, 'w', newline='') as csvfile:
            if not individual_results:
                return filename
            
            # Get all possible field names
            fieldnames = set()
            for result in individual_results:
                metrics = result.get("metrics", {})
                config = result.get("config", {})
                fieldnames.update(metrics.keys())
                fieldnames.update(f"config_{k}" for k in config.keys())
                fieldnames.add("success")
            
            fieldnames = sorted(list(fieldnames))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in individual_results:
                row = {}
                row.update(result.get("metrics", {}))
                row.update({f"config_{k}": v for k, v in result.get("config", {}).items()})
                row["success"] = result.get("success", False)
                writer.writerow(row)
        
        return filename
    
    def validate_data_quality(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality for AI research"""
        
        individual_results = performance_data.get("individual_results", [])
        
        quality_metrics = {
            "total_samples": len(individual_results),
            "successful_samples": len([r for r in individual_results if r.get("success")]),
            "success_rate": 0,
            "missing_metrics": [],
            "data_completeness": 0,
            "recommended_for_ai": False
        }
        
        if individual_results:
            quality_metrics["success_rate"] = quality_metrics["successful_samples"] / quality_metrics["total_samples"]
        
        # Check for missing critical metrics
        required_metrics = ["avg_fps", "avg_frame_time", "avg_gpu_util"]
        missing_count = 0
        
        for result in individual_results:
            if result.get("success"):
                metrics = result.get("metrics", {})
                for metric in required_metrics:
                    if metric not in metrics:
                        missing_count += 1
        
        if quality_metrics["successful_samples"] > 0:
            quality_metrics["data_completeness"] = 1.0 - (missing_count / (quality_metrics["successful_samples"] * len(required_metrics)))
        
        # Recommendation for AI research
        quality_metrics["recommended_for_ai"] = (
            quality_metrics["successful_samples"] >= 10 and
            quality_metrics["success_rate"] >= 0.7 and
            quality_metrics["data_completeness"] >= 0.8
        )
        
        return quality_metrics 