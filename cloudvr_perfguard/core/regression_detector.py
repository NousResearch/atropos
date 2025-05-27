"""
Regression Detector - Analyzes performance differences between builds
Core logic for detecting VR performance regressions
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

class RegressionDetector:
    """
    Detects performance regressions between VR builds
    Implements statistical analysis and VR-specific regression criteria
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
        # Regression thresholds (configurable)
        self.regression_thresholds = {
            # FPS regressions (critical for VR)
            "avg_fps_threshold": 0.05,  # 5% decrease
            "min_fps_threshold": 0.10,  # 10% decrease in minimum FPS
            "fps_p1_threshold": 0.15,   # 15% decrease in 1st percentile
            
            # Frame time regressions (critical for VR comfort)
            "avg_frame_time_threshold": 0.05,  # 5% increase
            "p99_frame_time_threshold": 0.10,  # 10% increase in 99th percentile
            
            # GPU utilization
            "gpu_util_threshold": 0.20,  # 20% increase
            "vram_threshold": 0.15,      # 15% increase
            
            # VR-specific metrics
            "comfort_score_threshold": 0.10,  # 10% decrease in comfort
            "dropped_frames_threshold": 2.0,   # 2x increase in dropped frames
            
            # Statistical significance
            "min_effect_size": 0.5,     # Cohen's d
            "confidence_level": 0.95    # 95% confidence
        }
        
        # Severity levels
        self.severity_levels = {
            "critical": {"color": "#FF0000", "priority": 1},
            "major": {"color": "#FF8000", "priority": 2},
            "minor": {"color": "#FFFF00", "priority": 3},
            "info": {"color": "#00FF00", "priority": 4}
        }
    
    async def analyze_regression(self, job_id: str) -> Dict[str, Any]:
        """
        Analyze regression for a test job
        Main entry point for regression analysis
        """
        
        print(f"🔍 Analyzing regression for job {job_id}")
        
        try:
            # Get job data
            job_data = await self.db_manager.get_test_job(job_id)
            if not job_data:
                raise ValueError(f"Job {job_id} not found")
            
            if job_data["submission_type"] != "regression_test":
                raise ValueError(f"Job {job_id} is not a regression test")
            
            # Get current and baseline performance data
            current_results = await self.db_manager.get_performance_results(job_id)
            baseline_results = await self._get_baseline_results(
                job_data["app_name"], 
                job_data["baseline_version"]
            )
            
            if not baseline_results:
                raise ValueError(f"No baseline data found for {job_data['app_name']} v{job_data['baseline_version']}")
            
            # Perform regression analysis
            regression_analysis = self._perform_regression_analysis(
                current_results, baseline_results
            )
            
            # Store regression results
            await self.db_manager.store_regression_analysis(job_id, regression_analysis)
            
            print(f"✅ Regression analysis complete for job {job_id}")
            print(f"   Regressions detected: {len(regression_analysis['regressions'])}")
            
            return regression_analysis
            
        except Exception as e:
            print(f"❌ Regression analysis failed for job {job_id}: {e}")
            raise
    
    async def generate_report(self, job_id: str) -> Dict[str, Any]:
        """Generate comprehensive regression report"""
        
        try:
            # Get job and regression data
            job_data = await self.db_manager.get_test_job(job_id)
            regression_data = await self.db_manager.get_regression_analysis(job_id)
            current_results = await self.db_manager.get_performance_results(job_id)
            
            # Generate report
            report = {
                "job_id": job_id,
                "app_name": job_data["app_name"],
                "current_version": job_data["build_version"],
                "baseline_version": job_data["baseline_version"],
                "test_timestamp": job_data["created_at"],
                "report_generated": datetime.utcnow().isoformat(),
                
                # Summary
                "summary": self._generate_summary(regression_data),
                
                # Detailed regressions
                "regressions": regression_data.get("regressions", []),
                
                # Performance comparison
                "performance_comparison": regression_data.get("comparison", {}),
                
                # Recommendations
                "recommendations": self._generate_recommendations(regression_data),
                
                # Raw data
                "current_performance": current_results,
                "regression_analysis": regression_data
            }
            
            return report
            
        except Exception as e:
            print(f"ERROR generating report for job {job_id}: {e}")
            raise
    
    async def _get_baseline_results(self, app_name: str, baseline_version: str) -> Optional[Dict[str, Any]]:
        """Get baseline performance results for comparison"""
        
        # Find baseline job
        baseline_job = await self.db_manager.get_baseline_job(app_name, baseline_version)
        if not baseline_job:
            return None
        
        # Get baseline performance data
        baseline_results = await self.db_manager.get_performance_results(baseline_job["job_id"])
        return baseline_results
    
    def _perform_regression_analysis(
        self, 
        current_results: Dict[str, Any], 
        baseline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform detailed regression analysis between current and baseline results
        """
        
        print("📊 Performing regression analysis...")
        
        # Extract aggregated metrics for comparison
        current_metrics = current_results.get("aggregated_metrics", {})
        baseline_metrics = baseline_results.get("aggregated_metrics", {})
        
        # Detect regressions
        regressions = self._detect_regressions(current_metrics, baseline_metrics)
        
        # Perform statistical analysis
        statistical_analysis = self._perform_statistical_analysis(
            current_results.get("individual_results", []),
            baseline_results.get("individual_results", [])
        )
        
        # Create detailed comparison
        comparison = self._create_detailed_comparison(current_metrics, baseline_metrics)
        
        # Calculate overall regression score
        regression_score = self._calculate_regression_score(regressions)
        
        analysis = {
            "regressions": regressions,
            "statistical_analysis": statistical_analysis,
            "comparison": comparison,
            "regression_score": regression_score,
            "overall_status": self._determine_overall_status(regressions),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        return analysis
    
    def _detect_regressions(
        self, 
        current_metrics: Dict[str, Any], 
        baseline_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect specific performance regressions"""
        
        regressions = []
        
        # FPS regressions
        regressions.extend(self._check_fps_regressions(current_metrics, baseline_metrics))
        
        # Frame time regressions
        regressions.extend(self._check_frame_time_regressions(current_metrics, baseline_metrics))
        
        # GPU utilization regressions
        regressions.extend(self._check_gpu_regressions(current_metrics, baseline_metrics))
        
        # VR-specific regressions
        regressions.extend(self._check_vr_regressions(current_metrics, baseline_metrics))
        
        # Sort by severity
        regressions.sort(key=lambda x: self.severity_levels[x["severity"]]["priority"])
        
        return regressions
    
    def _check_fps_regressions(
        self, 
        current: Dict[str, Any], 
        baseline: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for FPS-related regressions"""
        
        regressions = []
        
        # Average FPS regression
        current_fps = current.get("overall_avg_fps", 0)
        baseline_fps = baseline.get("overall_avg_fps", 0)
        
        if baseline_fps > 0:
            fps_change = (current_fps - baseline_fps) / baseline_fps
            
            if fps_change < -self.regression_thresholds["avg_fps_threshold"]:
                severity = self._determine_fps_severity(fps_change, current_fps)
                
                regressions.append({
                    "type": "fps_regression",
                    "metric": "average_fps",
                    "severity": severity,
                    "current_value": current_fps,
                    "baseline_value": baseline_fps,
                    "change_percent": fps_change * 100,
                    "description": f"Average FPS decreased by {abs(fps_change)*100:.1f}% ({baseline_fps:.1f} → {current_fps:.1f})",
                    "impact": "VR experience may feel less smooth"
                })
        
        # Minimum FPS regression (critical for VR)
        current_min_fps = current.get("overall_min_fps", 0)
        baseline_min_fps = baseline.get("overall_min_fps", 0)
        
        if baseline_min_fps > 0:
            min_fps_change = (current_min_fps - baseline_min_fps) / baseline_min_fps
            
            if min_fps_change < -self.regression_thresholds["min_fps_threshold"]:
                severity = "critical" if current_min_fps < 45 else "major"
                
                regressions.append({
                    "type": "fps_regression",
                    "metric": "minimum_fps",
                    "severity": severity,
                    "current_value": current_min_fps,
                    "baseline_value": baseline_min_fps,
                    "change_percent": min_fps_change * 100,
                    "description": f"Minimum FPS decreased by {abs(min_fps_change)*100:.1f}% ({baseline_min_fps:.1f} → {current_min_fps:.1f})",
                    "impact": "Users may experience stuttering and motion sickness"
                })
        
        return regressions
    
    def _check_frame_time_regressions(
        self, 
        current: Dict[str, Any], 
        baseline: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for frame time regressions"""
        
        regressions = []
        
        # P99 frame time regression (critical for VR comfort)
        current_p99 = current.get("overall_p99_frame_time", 0)
        baseline_p99 = baseline.get("overall_p99_frame_time", 0)
        
        if baseline_p99 > 0:
            p99_change = (current_p99 - baseline_p99) / baseline_p99
            
            if p99_change > self.regression_thresholds["p99_frame_time_threshold"]:
                severity = "critical" if current_p99 > 20 else "major"  # 20ms = 50 FPS
                
                regressions.append({
                    "type": "frame_time_regression",
                    "metric": "p99_frame_time",
                    "severity": severity,
                    "current_value": current_p99,
                    "baseline_value": baseline_p99,
                    "change_percent": p99_change * 100,
                    "description": f"99th percentile frame time increased by {p99_change*100:.1f}% ({baseline_p99:.1f}ms → {current_p99:.1f}ms)",
                    "impact": "Occasional frame drops may cause discomfort"
                })
        
        return regressions
    
    def _check_gpu_regressions(
        self, 
        current: Dict[str, Any], 
        baseline: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for GPU utilization regressions"""
        
        regressions = []
        
        # VRAM usage regression
        current_vram = current.get("overall_max_vram", 0)
        baseline_vram = baseline.get("overall_max_vram", 0)
        
        if baseline_vram > 0:
            vram_change = (current_vram - baseline_vram) / baseline_vram
            
            if vram_change > self.regression_thresholds["vram_threshold"]:
                severity = "major" if vram_change > 0.3 else "minor"
                
                regressions.append({
                    "type": "resource_regression",
                    "metric": "vram_usage",
                    "severity": severity,
                    "current_value": current_vram,
                    "baseline_value": baseline_vram,
                    "change_percent": vram_change * 100,
                    "description": f"VRAM usage increased by {vram_change*100:.1f}% ({baseline_vram:.0f}MB → {current_vram:.0f}MB)",
                    "impact": "May cause issues on lower-end GPUs"
                })
        
        return regressions
    
    def _check_vr_regressions(
        self, 
        current: Dict[str, Any], 
        baseline: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for VR-specific regressions"""
        
        regressions = []
        
        # VR comfort score regression
        current_comfort = current.get("overall_comfort_score", 0)
        baseline_comfort = baseline.get("overall_comfort_score", 0)
        
        if baseline_comfort > 0:
            comfort_change = (current_comfort - baseline_comfort) / baseline_comfort
            
            if comfort_change < -self.regression_thresholds["comfort_score_threshold"]:
                severity = "major" if current_comfort < 70 else "minor"
                
                regressions.append({
                    "type": "vr_regression",
                    "metric": "comfort_score",
                    "severity": severity,
                    "current_value": current_comfort,
                    "baseline_value": baseline_comfort,
                    "change_percent": comfort_change * 100,
                    "description": f"VR comfort score decreased by {abs(comfort_change)*100:.1f}% ({baseline_comfort:.1f} → {current_comfort:.1f})",
                    "impact": "Users may experience increased motion sickness"
                })
        
        return regressions
    
    def _determine_fps_severity(self, fps_change: float, current_fps: float) -> str:
        """Determine severity of FPS regression"""
        
        if current_fps < 45:  # Below minimum VR threshold
            return "critical"
        elif fps_change < -0.15:  # >15% decrease
            return "major"
        elif fps_change < -0.10:  # >10% decrease
            return "minor"
        else:
            return "info"
    
    def _perform_statistical_analysis(
        self, 
        current_results: List[Dict[str, Any]], 
        baseline_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform statistical significance testing"""
        
        # Extract FPS values for statistical testing
        current_fps = []
        baseline_fps = []
        
        for result in current_results:
            if result.get("success") and "metrics" in result:
                current_fps.append(result["metrics"].get("avg_fps", 0))
        
        for result in baseline_results:
            if result.get("success") and "metrics" in result:
                baseline_fps.append(result["metrics"].get("avg_fps", 0))
        
        if len(current_fps) < 2 or len(baseline_fps) < 2:
            return {"error": "Insufficient data for statistical analysis"}
        
        # Perform t-test
        from scipy import stats
        
        t_stat, p_value = stats.ttest_ind(current_fps, baseline_fps)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(current_fps) - 1) * np.var(current_fps, ddof=1) + 
                             (len(baseline_fps) - 1) * np.var(baseline_fps, ddof=1)) / 
                            (len(current_fps) + len(baseline_fps) - 2))
        
        cohens_d = (np.mean(current_fps) - np.mean(baseline_fps)) / pooled_std if pooled_std > 0 else 0
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "effect_size": self._interpret_effect_size(abs(cohens_d)),
            "statistically_significant": p_value < (1 - self.regression_thresholds["confidence_level"]),
            "sample_sizes": {"current": len(current_fps), "baseline": len(baseline_fps)}
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _create_detailed_comparison(
        self, 
        current: Dict[str, Any], 
        baseline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create detailed metric-by-metric comparison"""
        
        comparison = {}
        
        metrics_to_compare = [
            "overall_avg_fps", "overall_min_fps", "overall_max_fps",
            "overall_avg_frame_time", "overall_p99_frame_time",
            "overall_avg_gpu_util", "overall_max_vram",
            "overall_comfort_score"
        ]
        
        for metric in metrics_to_compare:
            current_val = current.get(metric, 0)
            baseline_val = baseline.get(metric, 0)
            
            if baseline_val > 0:
                change_percent = ((current_val - baseline_val) / baseline_val) * 100
                change_direction = "increase" if change_percent > 0 else "decrease"
                
                comparison[metric] = {
                    "current": current_val,
                    "baseline": baseline_val,
                    "change_absolute": current_val - baseline_val,
                    "change_percent": change_percent,
                    "change_direction": change_direction,
                    "is_regression": self._is_metric_regression(metric, change_percent)
                }
        
        return comparison
    
    def _is_metric_regression(self, metric: str, change_percent: float) -> bool:
        """Determine if a metric change constitutes a regression"""
        
        # Metrics where increase is bad
        bad_increase_metrics = [
            "overall_avg_frame_time", "overall_p99_frame_time",
            "overall_avg_gpu_util", "overall_max_vram"
        ]
        
        # Metrics where decrease is bad
        bad_decrease_metrics = [
            "overall_avg_fps", "overall_min_fps", "overall_max_fps",
            "overall_comfort_score"
        ]
        
        if metric in bad_increase_metrics:
            return change_percent > 5  # 5% increase threshold
        elif metric in bad_decrease_metrics:
            return change_percent < -5  # 5% decrease threshold
        
        return False
    
    def _calculate_regression_score(self, regressions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall regression severity score"""
        
        if not regressions:
            return {"score": 0, "level": "none", "description": "No regressions detected"}
        
        # Weight regressions by severity
        severity_weights = {"critical": 10, "major": 5, "minor": 2, "info": 1}
        
        total_score = sum(severity_weights.get(r["severity"], 0) for r in regressions)
        
        # Determine overall level
        if total_score >= 20:
            level = "critical"
        elif total_score >= 10:
            level = "major"
        elif total_score >= 5:
            level = "minor"
        else:
            level = "info"
        
        return {
            "score": total_score,
            "level": level,
            "regression_count": len(regressions),
            "description": f"{len(regressions)} regressions detected with {level} overall severity"
        }
    
    def _determine_overall_status(self, regressions: List[Dict[str, Any]]) -> str:
        """Determine overall test status"""
        
        if not regressions:
            return "passed"
        
        critical_count = sum(1 for r in regressions if r["severity"] == "critical")
        major_count = sum(1 for r in regressions if r["severity"] == "major")
        
        if critical_count > 0:
            return "failed"
        elif major_count > 0:
            return "warning"
        else:
            return "passed_with_issues"
    
    def _generate_summary(self, regression_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of regression analysis"""
        
        regressions = regression_data.get("regressions", [])
        regression_score = regression_data.get("regression_score", {})
        
        summary = {
            "overall_status": regression_data.get("overall_status", "unknown"),
            "regression_count": len(regressions),
            "severity_breakdown": {},
            "key_issues": [],
            "recommendation": ""
        }
        
        # Count by severity
        for severity in ["critical", "major", "minor", "info"]:
            count = sum(1 for r in regressions if r["severity"] == severity)
            summary["severity_breakdown"][severity] = count
        
        # Identify key issues
        critical_regressions = [r for r in regressions if r["severity"] == "critical"]
        major_regressions = [r for r in regressions if r["severity"] == "major"]
        
        summary["key_issues"] = critical_regressions[:3] + major_regressions[:2]  # Top 5 issues
        
        # Generate recommendation
        if critical_regressions:
            summary["recommendation"] = "Do not deploy - critical performance regressions detected"
        elif major_regressions:
            summary["recommendation"] = "Review required - significant performance impact detected"
        elif regressions:
            summary["recommendation"] = "Monitor closely - minor performance changes detected"
        else:
            summary["recommendation"] = "Safe to deploy - no performance regressions detected"
        
        return summary
    
    def _generate_recommendations(self, regression_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on regression analysis"""
        
        recommendations = []
        regressions = regression_data.get("regressions", [])
        
        # FPS-related recommendations
        fps_regressions = [r for r in regressions if r["type"] == "fps_regression"]
        if fps_regressions:
            recommendations.append({
                "category": "performance",
                "priority": "high",
                "title": "Investigate FPS Degradation",
                "description": "Frame rate has decreased significantly. Check for new rendering features, increased scene complexity, or inefficient code changes.",
                "actions": [
                    "Profile GPU usage with RenderDoc or NSight Graphics",
                    "Review recent changes to rendering pipeline",
                    "Check for new high-poly assets or complex shaders",
                    "Verify LOD system is working correctly"
                ]
            })
        
        # VRAM recommendations
        vram_regressions = [r for r in regressions if r["metric"] == "vram_usage"]
        if vram_regressions:
            recommendations.append({
                "category": "memory",
                "priority": "medium",
                "title": "Optimize Memory Usage",
                "description": "VRAM usage has increased. This may cause issues on lower-end GPUs.",
                "actions": [
                    "Review texture compression settings",
                    "Check for memory leaks in asset loading",
                    "Optimize texture streaming",
                    "Consider reducing texture resolution for distant objects"
                ]
            })
        
        # VR comfort recommendations
        comfort_regressions = [r for r in regressions if r["metric"] == "comfort_score"]
        if comfort_regressions:
            recommendations.append({
                "category": "vr_comfort",
                "priority": "high",
                "title": "Address VR Comfort Issues",
                "description": "Frame consistency has degraded, which may cause motion sickness.",
                "actions": [
                    "Investigate frame time spikes",
                    "Review asynchronous loading systems",
                    "Check for blocking operations on main thread",
                    "Optimize garbage collection patterns"
                ]
            })
        
        return recommendations 