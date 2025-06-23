"""
Real Data Integration for CloudVR-PerfGuard AI Research
Connects to the CloudVR-PerfGuard database and pulls real performance data for AI analysis
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ai_integration.data_adapter import PerformanceDataAdapter
from ai_integration.function_discovery import OptimizationDiscovery
from ai_integration.paper_generator import ResearchPaperGenerator
from core.database import DatabaseManager


class RealDataResearchPipeline:
    """
    Integrates real CloudVR-PerfGuard data with AI research pipeline
    """

    def __init__(self, db_path: str = "cloudvr_perfguard.db"):
        self.db_manager = DatabaseManager(db_path)
        self.data_adapter = PerformanceDataAdapter()
        self.paper_generator = ResearchPaperGenerator()
        self.function_discovery = OptimizationDiscovery()

        # Research configuration
        self.research_config = {
            "min_tests_for_research": 10,
            "max_tests_per_research": 100,
            "research_quality_threshold": 80.0,
            "max_cost_per_research": 25.0,
        }

    async def initialize(self):
        """Initialize the research pipeline"""
        await self.db_manager.initialize()
        print("✅ Real Data Research Pipeline initialized")

    async def get_real_performance_data(
        self, app_name: Optional[str] = None, limit: int = 50, days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Pull real performance data from CloudVR-PerfGuard database

        Args:
            app_name: Specific app to analyze (None for all apps)
            limit: Maximum number of test results to retrieve
            days_back: How many days back to look for data

        Returns:
            List of performance test results
        """

        try:
            # Get recent completed jobs
            recent_jobs = await self.db_manager.get_recent_jobs(limit=limit * 2)

            # Filter for completed jobs within time range
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            performance_data = []

            for job in recent_jobs:
                # Skip if not completed
                if job["status"] != "completed":
                    continue

                # Skip if outside time range
                job_date = datetime.fromisoformat(
                    job["created_at"].replace("Z", "+00:00")
                )
                if job_date < cutoff_date:
                    continue

                # Skip if app filter doesn't match
                if app_name and job["app_name"] != app_name:
                    continue

                # Get performance results for this job
                results = await self.db_manager.get_performance_results(job["job_id"])
                if results:
                    # Add job metadata to results
                    results["job_metadata"] = {
                        "job_id": job["job_id"],
                        "app_name": job["app_name"],
                        "build_version": job["build_version"],
                        "submission_type": job["submission_type"],
                        "created_at": job["created_at"],
                    }
                    performance_data.append(results)

                # Stop if we have enough data
                if len(performance_data) >= limit:
                    break

            print(f"📊 Retrieved {len(performance_data)} real performance test results")
            return performance_data

        except Exception as e:
            print(f"❌ Error retrieving real performance data: {e}")
            return []

    async def generate_research_from_real_data(
        self, app_name: Optional[str] = None, research_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate AI research from real CloudVR-PerfGuard data

        Args:
            app_name: Specific app to analyze
            research_type: Type of research ('comprehensive', 'performance_analysis', 'optimization_study')

        Returns:
            Research results including papers and functions
        """

        print("🔬 Starting real data research generation...")
        print(f"   App: {app_name or 'All apps'}")
        print(f"   Type: {research_type}")

        # Pull real data
        real_data = await self.get_real_performance_data(app_name=app_name, limit=100)

        if len(real_data) < self.research_config["min_tests_for_research"]:
            return {
                "success": False,
                "error": f"Insufficient data: {len(real_data)} tests (minimum {self.research_config['min_tests_for_research']})",
                "data_count": len(real_data),
            }

        # Convert to research format
        research_data = self._convert_real_data_to_research_format(real_data)

        # Generate research based on type
        results = {
            "success": True,
            "research_type": research_type,
            "data_count": len(real_data),
            "apps_analyzed": list(
                set(d["job_metadata"]["app_name"] for d in real_data)
            ),
            "time_range": self._get_data_time_range(real_data),
            "papers": [],
            "functions": [],
            "total_cost": 0.0,
            "generation_time": datetime.utcnow().isoformat(),
        }

        try:
            if research_type in ["comprehensive", "performance_analysis"]:
                # Generate performance analysis paper
                paper_result = await self._generate_performance_paper(
                    research_data, real_data
                )
                if paper_result:
                    results["papers"].append(paper_result)
                    results["total_cost"] += paper_result.get("generation_cost", 0)

            if research_type in ["comprehensive", "optimization_study"]:
                # Generate optimization functions
                function_results = await self._generate_optimization_functions(
                    research_data
                )
                results["functions"].extend(function_results)

            if research_type == "comprehensive":
                # Generate comparative analysis if multiple apps
                if len(results["apps_analyzed"]) > 1:
                    comparison_paper = await self._generate_comparison_paper(
                        research_data, real_data
                    )
                    if comparison_paper:
                        results["papers"].append(comparison_paper)
                        results["total_cost"] += comparison_paper.get(
                            "generation_cost", 0
                        )

            # Calculate research quality
            results["research_quality"] = self._calculate_research_quality(results)

            print("✅ Real data research completed!")
            print(f"   Papers generated: {len(results['papers'])}")
            print(f"   Functions discovered: {len(results['functions'])}")
            print(f"   Total cost: ${results['total_cost']:.2f}")
            print(f"   Research quality: {results['research_quality']:.1f}/100")

            return results

        except Exception as e:
            print(f"❌ Error generating research: {e}")
            results["success"] = False
            results["error"] = str(e)
            return results

    def _convert_real_data_to_research_format(
        self, real_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert real CloudVR-PerfGuard data to AI research format"""

        # Aggregate all individual results
        all_individual_results = []
        apps_tested = set()

        for test_result in real_data:
            job_meta = test_result["job_metadata"]
            apps_tested.add(job_meta["app_name"])

            # Add job metadata to each individual result
            for individual in test_result["individual_results"]:
                individual["job_metadata"] = job_meta
                all_individual_results.append(individual)

        # Create aggregated research data
        research_data = {
            "test_id": f"real_data_research_{int(datetime.utcnow().timestamp())}",
            "build_path": f"Real CloudVR-PerfGuard Data ({len(apps_tested)} apps)",
            "total_duration": sum(r.get("total_duration", 0) for r in real_data),
            "timestamp": datetime.utcnow().isoformat(),
            "individual_results": all_individual_results,
            "metadata": {
                "source": "real_cloudvr_perfguard_data",
                "apps_count": len(apps_tested),
                "apps_tested": list(apps_tested),
                "tests_count": len(real_data),
                "individual_results_count": len(all_individual_results),
                "data_collection_period": self._get_data_time_range(real_data),
            },
        }

        return research_data

    def _get_data_time_range(self, real_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Get the time range of the data"""

        if not real_data:
            return {"start": "unknown", "end": "unknown"}

        timestamps = [d["job_metadata"]["created_at"] for d in real_data]
        return {
            "start": min(timestamps),
            "end": max(timestamps),
            "span_days": (
                datetime.fromisoformat(max(timestamps).replace("Z", "+00:00"))
                - datetime.fromisoformat(min(timestamps).replace("Z", "+00:00"))
            ).days,
        }

    async def _generate_performance_paper(
        self, research_data: Dict[str, Any], real_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate performance analysis paper from real data"""

        try:
            # Convert to AI Scientist format
            ai_scientist_data = self.data_adapter.to_ai_scientist_format(research_data)

            # Create custom title with real data context
            apps_list = ", ".join(research_data["metadata"]["apps_tested"][:3])
            if len(research_data["metadata"]["apps_tested"]) > 3:
                apps_list += (
                    f" and {len(research_data['metadata']['apps_tested']) - 3} others"
                )

            title = f"Real-World VR Performance Analysis: {apps_list} Study"

            # Generate paper
            paper_result = self.paper_generator.generate_paper(
                ai_scientist_data, paper_type="performance_analysis", custom_title=title
            )

            # Add real data context
            paper_result["real_data_context"] = {
                "apps_analyzed": research_data["metadata"]["apps_tested"],
                "test_count": research_data["metadata"]["tests_count"],
                "data_source": "CloudVR-PerfGuard Production Database",
                "time_range": research_data["metadata"]["data_collection_period"],
            }

            return paper_result

        except Exception as e:
            print(f"❌ Error generating performance paper: {e}")
            return None

    async def _generate_optimization_functions(
        self, research_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate optimization functions from real data"""

        functions = []

        try:
            # Convert to FunSearch format
            funsearch_data = self.data_adapter.to_funsearch_format(research_data)

            # Generate functions for different optimization domains
            domains = [
                "frame_time_consistency",
                "comfort_optimization",
                "performance_efficiency",
            ]

            for domain in domains:
                try:
                    function_result = (
                        self.function_discovery.discover_optimization_function(
                            funsearch_data, domain=domain
                        )
                    )

                    # Add real data context
                    function_result["real_data_context"] = {
                        "training_samples": len(funsearch_data["features"]),
                        "apps_trained_on": research_data["metadata"]["apps_tested"],
                        "data_source": "CloudVR-PerfGuard Production Database",
                    }

                    functions.append(function_result)

                except Exception as e:
                    print(f"⚠️  Error generating {domain} function: {e}")

            return functions

        except Exception as e:
            print(f"❌ Error generating optimization functions: {e}")
            return []

    async def _generate_comparison_paper(
        self, research_data: Dict[str, Any], real_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate comparative analysis paper for multiple apps"""

        try:
            # Convert to AI Scientist format
            ai_scientist_data = self.data_adapter.to_ai_scientist_format(research_data)

            # Create comparative title
            apps = research_data["metadata"]["apps_tested"]
            title = f"Comparative VR Performance Analysis: {' vs '.join(apps[:3])}"
            if len(apps) > 3:
                title += f" and {len(apps) - 3} Other Applications"

            # Generate comparative paper
            paper_result = self.paper_generator.generate_paper(
                ai_scientist_data, paper_type="comparative_study", custom_title=title
            )

            # Add comparative context
            paper_result["comparative_context"] = {
                "apps_compared": apps,
                "comparison_metrics": [
                    "fps",
                    "frame_time",
                    "gpu_utilization",
                    "comfort_score",
                ],
                "data_source": "CloudVR-PerfGuard Production Database",
            }

            return paper_result

        except Exception as e:
            print(f"❌ Error generating comparison paper: {e}")
            return None

    def _calculate_research_quality(self, results: Dict[str, Any]) -> float:
        """Calculate overall research quality score"""

        quality_factors = []

        # Data quality (30%)
        data_count = results["data_count"]
        if data_count >= 50:
            quality_factors.append(30.0)
        elif data_count >= 20:
            quality_factors.append(25.0)
        else:
            quality_factors.append(15.0)

        # Paper quality (40%)
        if results["papers"]:
            avg_paper_quality = sum(
                p.get("quality_score", 0) for p in results["papers"]
            ) / len(results["papers"])
            quality_factors.append(avg_paper_quality * 0.4)
        else:
            quality_factors.append(0)

        # Function quality (20%)
        if results["functions"]:
            avg_function_fitness = sum(
                f.get("fitness_score", 0) for f in results["functions"]
            ) / len(results["functions"])
            quality_factors.append(avg_function_fitness * 20.0)
        else:
            quality_factors.append(0)

        # Diversity bonus (10%)
        app_diversity = min(len(results["apps_analyzed"]) * 2.5, 10.0)
        quality_factors.append(app_diversity)

        return sum(quality_factors)

    async def run_daily_research(self) -> Dict[str, Any]:
        """Run daily automated research generation"""

        print("🔄 Running daily automated research...")

        # Get all apps with recent data
        recent_jobs = await self.db_manager.get_recent_jobs(limit=100)
        apps_with_data = set(
            job["app_name"] for job in recent_jobs if job["status"] == "completed"
        )

        daily_results = {
            "date": datetime.utcnow().date().isoformat(),
            "apps_analyzed": list(apps_with_data),
            "research_generated": [],
            "total_cost": 0.0,
            "success": True,
        }

        try:
            # Generate comprehensive research for each app with sufficient data
            for app_name in apps_with_data:
                app_research = await self.generate_research_from_real_data(
                    app_name=app_name, research_type="performance_analysis"
                )

                if app_research["success"]:
                    daily_results["research_generated"].append(app_research)
                    daily_results["total_cost"] += app_research.get("total_cost", 0)

            # Generate cross-app comparison if multiple apps
            if len(apps_with_data) > 1:
                comparison_research = await self.generate_research_from_real_data(
                    app_name=None, research_type="comprehensive"
                )

                if comparison_research["success"]:
                    daily_results["research_generated"].append(comparison_research)
                    daily_results["total_cost"] += comparison_research.get(
                        "total_cost", 0
                    )

            print("✅ Daily research completed!")
            print(f"   Apps analyzed: {len(apps_with_data)}")
            print(f"   Research items: {len(daily_results['research_generated'])}")
            print(f"   Total cost: ${daily_results['total_cost']:.2f}")

            return daily_results

        except Exception as e:
            print(f"❌ Daily research failed: {e}")
            daily_results["success"] = False
            daily_results["error"] = str(e)
            return daily_results

    async def close(self):
        """Close the research pipeline"""
        await self.db_manager.close()
        print("✅ Real Data Research Pipeline closed")


# Convenience functions for direct usage
async def generate_research_from_real_data(
    app_name: Optional[str] = None,
    research_type: str = "comprehensive",
    db_path: str = "cloudvr_perfguard.db",
) -> Dict[str, Any]:
    """
    Convenience function to generate research from real data
    """

    pipeline = RealDataResearchPipeline(db_path)
    await pipeline.initialize()

    try:
        results = await pipeline.generate_research_from_real_data(
            app_name, research_type
        )
        return results
    finally:
        await pipeline.close()


async def run_daily_research(db_path: str = "cloudvr_perfguard.db") -> Dict[str, Any]:
    """
    Convenience function to run daily research
    """

    pipeline = RealDataResearchPipeline(db_path)
    await pipeline.initialize()

    try:
        results = await pipeline.run_daily_research()
        return results
    finally:
        await pipeline.close()


if __name__ == "__main__":
    # Example usage
    async def main():
        print("🚀 CloudVR-PerfGuard Real Data Research Pipeline")
        print("=" * 60)

        # Run comprehensive research on real data
        results = await generate_research_from_real_data(research_type="comprehensive")

        if results["success"]:
            print("\n📊 Research Results:")
            print(f"   Apps analyzed: {results['apps_analyzed']}")
            print(f"   Papers generated: {len(results['papers'])}")
            print(f"   Functions discovered: {len(results['functions'])}")
            print(f"   Research quality: {results['research_quality']:.1f}/100")
            print(f"   Total cost: ${results['total_cost']:.2f}")
        else:
            print(f"\n❌ Research failed: {results.get('error', 'Unknown error')}")

    asyncio.run(main())
