"""
Research Orchestrator for CloudVR-PerfGuard AI Integration
Coordinates FunSearch evolution and AI Scientist paper generation
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.database import DatabaseManager
from .ai_scientist_integration import AIScientistIntegration
from .funsearch_integration import FunSearchIntegration


class ResearchOrchestrator:
    """
    Orchestrates the complete AI research pipeline:
    1. VR performance testing
    2. FunSearch function evolution
    3. AI Scientist paper generation
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.funsearch = FunSearchIntegration(db_manager)
        self.ai_scientist = AIScientistIntegration(db_manager)

        # Research pipeline configurations
        self.pipeline_configs = {
            "quick_discovery": {
                "evolution_types": ["visual_cue_discovery"],
                "paper_types": ["vr_affordance_discovery"],
                "max_evolution_time": 300,  # 5 minutes
                "expected_duration": 600,  # 10 minutes total
            },
            "comprehensive_analysis": {
                "evolution_types": [
                    "visual_cue_discovery",
                    "affordance_scoring",
                    "interaction_predictor",
                ],
                "paper_types": ["vr_affordance_discovery", "performance_optimization"],
                "max_evolution_time": 1800,  # 30 minutes
                "expected_duration": 3600,  # 1 hour total
            },
            "full_research_suite": {
                "evolution_types": [
                    "visual_cue_discovery",
                    "affordance_scoring",
                    "interaction_predictor",
                ],
                "paper_types": [
                    "vr_affordance_discovery",
                    "performance_optimization",
                    "user_experience_analysis",
                ],
                "max_evolution_time": 3600,  # 1 hour
                "expected_duration": 7200,  # 2 hours total
            },
        }

    async def run_complete_research_pipeline(
        self,
        job_id: str,
        vr_performance_data: Dict[str, Any],
        pipeline_type: str = "comprehensive_analysis",
        custom_focus: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete AI research pipeline for a VR performance test job
        """
        try:
            print(
                f"INFO: Starting complete research pipeline for job {job_id}, type: {pipeline_type}"
            )

            # Get pipeline configuration
            config = self.pipeline_configs.get(
                pipeline_type, self.pipeline_configs["comprehensive_analysis"]
            )

            # Update job status
            await self.db_manager.update_job_status(
                job_id=job_id, status="ai_research_running", progress=10
            )

            # Phase 1: Function Evolution
            print(f"INFO: Phase 1 - Function Evolution for job {job_id}")
            evolution_results = await self._run_evolution_phase(
                job_id=job_id,
                vr_data=vr_performance_data,
                evolution_types=config["evolution_types"],
                max_time=config["max_evolution_time"],
            )

            await self.db_manager.update_job_status(
                job_id=job_id, status="ai_research_running", progress=60
            )

            # Phase 2: Paper Generation
            print(f"INFO: Phase 2 - Paper Generation for job {job_id}")
            paper_results = await self._run_paper_generation_phase(
                job_id=job_id,
                vr_data=vr_performance_data,
                evolved_functions=evolution_results["evolved_functions"],
                paper_types=config["paper_types"],
                custom_focus=custom_focus,
            )

            await self.db_manager.update_job_status(
                job_id=job_id, status="ai_research_running", progress=90
            )

            # Phase 3: Results Compilation
            print(f"INFO: Phase 3 - Results Compilation for job {job_id}")
            final_results = await self._compile_research_results(
                job_id=job_id,
                evolution_results=evolution_results,
                paper_results=paper_results,
                pipeline_type=pipeline_type,
            )

            # Update job status to completed
            await self.db_manager.update_job_status(
                job_id=job_id, status="ai_research_completed", progress=100
            )

            return {
                "status": "success",
                "job_id": job_id,
                "pipeline_type": pipeline_type,
                "results": final_results,
                "total_cost": final_results.get("total_cost", 0),
                "completion_time": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            print(f"ERROR: Research pipeline failed for job {job_id}: {e}")

            await self.db_manager.update_job_status(
                job_id=job_id, status="ai_research_failed", error=str(e)
            )

            return {"status": "error", "job_id": job_id, "error": str(e)}

    async def _run_evolution_phase(
        self,
        job_id: str,
        vr_data: Dict[str, Any],
        evolution_types: List[str],
        max_time: int,
    ) -> Dict[str, Any]:
        """Run the function evolution phase"""

        evolved_functions = []
        evolution_summaries = {}
        total_functions = 0

        start_time = datetime.utcnow()

        for i, evolution_type in enumerate(evolution_types):
            print(
                f"INFO: Running evolution type {evolution_type} ({i+1}/{len(evolution_types)})"
            )

            # Check time limit
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > max_time:
                print(
                    "WARNING: Evolution phase time limit reached, skipping remaining types"
                )
                break

            try:
                # Run FunSearch evolution
                result = await self.funsearch.evolve_visual_cue_function(
                    job_id=job_id,
                    vr_performance_data=vr_data,
                    evolution_type=evolution_type,
                )

                if result["status"] == "success":
                    # Get evolved functions for this type
                    type_functions = await self.funsearch.get_evolved_functions_for_job(
                        job_id
                    )
                    type_functions = [
                        f
                        for f in type_functions
                        if f.get("metadata", {}).get("evolution_type") == evolution_type
                    ]

                    evolved_functions.extend(type_functions)
                    total_functions += len(type_functions)

                    evolution_summaries[evolution_type] = {
                        "status": "success",
                        "functions_evolved": len(type_functions),
                        "best_score": result.get("best_score", 0),
                        "best_function": result.get("best_function"),
                    }
                else:
                    evolution_summaries[evolution_type] = {
                        "status": "error",
                        "error": result.get("error", "Unknown error"),
                    }

            except Exception as e:
                print(f"ERROR: Evolution type {evolution_type} failed: {e}")
                evolution_summaries[evolution_type] = {
                    "status": "error",
                    "error": str(e),
                }

        return {
            "evolved_functions": evolved_functions,
            "evolution_summaries": evolution_summaries,
            "total_functions": total_functions,
            "evolution_duration": (datetime.utcnow() - start_time).total_seconds(),
        }

    async def _run_paper_generation_phase(
        self,
        job_id: str,
        vr_data: Dict[str, Any],
        evolved_functions: List[Dict[str, Any]],
        paper_types: List[str],
        custom_focus: Optional[str],
    ) -> Dict[str, Any]:
        """Run the paper generation phase"""

        generated_papers = []
        paper_summaries = {}
        total_cost = 0.0

        start_time = datetime.utcnow()

        for i, paper_type in enumerate(paper_types):
            print(
                f"INFO: Generating paper type {paper_type} ({i+1}/{len(paper_types)})"
            )

            try:
                # Generate research paper
                result = await self.ai_scientist.generate_research_paper(
                    job_id=job_id,
                    vr_performance_data=vr_data,
                    evolved_functions=evolved_functions,
                    paper_type=paper_type,
                    custom_focus=custom_focus,
                )

                if result["status"] == "success":
                    generated_papers.append(result)
                    total_cost += result.get("generation_cost", 0)

                    paper_summaries[paper_type] = {
                        "status": "success",
                        "paper_id": result["paper_id"],
                        "title": result["title"],
                        "cost": result.get("generation_cost", 0),
                        "abstract_preview": (
                            result.get("abstract", "")[:200] + "..."
                            if result.get("abstract")
                            else ""
                        ),
                    }
                else:
                    paper_summaries[paper_type] = {
                        "status": "error",
                        "error": result.get("error", "Unknown error"),
                    }

            except Exception as e:
                print(f"ERROR: Paper type {paper_type} generation failed: {e}")
                paper_summaries[paper_type] = {"status": "error", "error": str(e)}

        return {
            "generated_papers": generated_papers,
            "paper_summaries": paper_summaries,
            "total_papers": len(generated_papers),
            "total_cost": total_cost,
            "generation_duration": (datetime.utcnow() - start_time).total_seconds(),
        }

    async def _compile_research_results(
        self,
        job_id: str,
        evolution_results: Dict[str, Any],
        paper_results: Dict[str, Any],
        pipeline_type: str,
    ) -> Dict[str, Any]:
        """Compile final research results"""

        # Calculate overall statistics
        total_functions = evolution_results["total_functions"]
        total_papers = paper_results["total_papers"]
        total_cost = paper_results["total_cost"]

        # Identify best discoveries
        best_functions = []
        for evo_type, summary in evolution_results["evolution_summaries"].items():
            if summary["status"] == "success" and summary.get("best_function"):
                best_functions.append(
                    {
                        "evolution_type": evo_type,
                        "score": summary["best_score"],
                        "function": summary["best_function"],
                    }
                )

        # Sort by score
        best_functions.sort(key=lambda x: x["score"], reverse=True)

        # Generate research summary
        research_summary = self._generate_research_summary(
            job_id=job_id,
            evolution_results=evolution_results,
            paper_results=paper_results,
            best_functions=best_functions[:3],  # Top 3
        )

        return {
            "job_id": job_id,
            "pipeline_type": pipeline_type,
            "statistics": {
                "total_functions_evolved": total_functions,
                "total_papers_generated": total_papers,
                "total_cost": total_cost,
                "evolution_duration": evolution_results["evolution_duration"],
                "paper_generation_duration": paper_results["generation_duration"],
            },
            "best_discoveries": best_functions[:5],  # Top 5
            "evolution_summaries": evolution_results["evolution_summaries"],
            "paper_summaries": paper_results["paper_summaries"],
            "research_summary": research_summary,
            "generated_papers": paper_results["generated_papers"],
            "total_cost": total_cost,
        }

    def _generate_research_summary(
        self,
        job_id: str,
        evolution_results: Dict[str, Any],
        paper_results: Dict[str, Any],
        best_functions: List[Dict[str, Any]],
    ) -> str:
        """Generate a human-readable research summary"""

        summary_parts = [
            f"# AI Research Pipeline Results for Job {job_id}",
            f"Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Overview",
            f"- **Functions Evolved**: {evolution_results['total_functions']}",
            f"- **Papers Generated**: {paper_results['total_papers']}",
            f"- **Total Cost**: ${paper_results['total_cost']:.2f}",
            f"- **Evolution Time**: {evolution_results['evolution_duration']:.1f} seconds",
            f"- **Paper Generation Time**: {paper_results['generation_duration']:.1f} seconds",
            "",
            "## Top Discoveries",
        ]

        for i, func in enumerate(best_functions, 1):
            summary_parts.extend(
                [
                    f"### {i}. {func['evolution_type'].replace('_', ' ').title()}",
                    f"- **Score**: {func['score']:.3f}",
                    f"- **Optimization Focus**: {func['function'].get('optimization_target', 'General VR affordance')}",
                ]
            )

        summary_parts.extend(["", "## Evolution Results"])

        for evo_type, summary in evolution_results["evolution_summaries"].items():
            status_icon = "✅" if summary["status"] == "success" else "❌"
            summary_parts.append(
                f"- **{evo_type.replace('_', ' ').title()}**: {status_icon} {summary['status']}"
            )
            if summary["status"] == "success":
                summary_parts.append(
                    f"  - Functions: {summary['functions_evolved']}, Best Score: {summary['best_score']:.3f}"
                )

        summary_parts.extend(["", "## Generated Papers"])

        for paper_type, summary in paper_results["paper_summaries"].items():
            status_icon = "✅" if summary["status"] == "success" else "❌"
            summary_parts.append(
                f"- **{paper_type.replace('_', ' ').title()}**: {status_icon} {summary['status']}"
            )
            if summary["status"] == "success":
                summary_parts.extend(
                    [
                        f"  - Title: {summary['title']}",
                        f"  - Cost: ${summary['cost']:.2f}",
                        f"  - Paper ID: {summary['paper_id']}",
                    ]
                )

        summary_parts.extend(
            [
                "",
                "## Next Steps",
                "1. Review evolved functions for implementation in VR applications",
                "2. Analyze generated papers for research insights and publication opportunities",
                "3. Consider running additional evolution cycles with refined parameters",
                "4. Integrate best-performing functions into production VR systems",
            ]
        )

        return "\n".join(summary_parts)

    async def get_research_status(self, job_id: str) -> Dict[str, Any]:
        """Get current research status for a job"""

        # Get job status
        job_data = await self.db_manager.get_test_job(job_id)
        if not job_data:
            return {"status": "not_found", "job_id": job_id}

        # Get evolved functions
        evolved_functions = await self.funsearch.get_evolved_functions_for_job(job_id)

        # Get generated papers
        generated_papers = await self.ai_scientist.get_papers_for_job(job_id)

        return {
            "status": "found",
            "job_id": job_id,
            "job_status": job_data.get("status"),
            "job_progress": job_data.get("progress", 0),
            "evolved_functions_count": len(evolved_functions),
            "generated_papers_count": len(generated_papers),
            "evolved_functions": evolved_functions,
            "generated_papers": generated_papers,
        }

    async def run_quick_evolution_only(
        self,
        job_id: str,
        vr_performance_data: Dict[str, Any],
        evolution_type: str = "visual_cue_discovery",
    ) -> Dict[str, Any]:
        """Run only function evolution (no paper generation) for quick testing"""

        try:
            print(
                f"INFO: Running quick evolution for job {job_id}, type: {evolution_type}"
            )

            await self.db_manager.update_job_status(
                job_id=job_id, status="evolution_running", progress=20
            )

            result = await self.funsearch.evolve_visual_cue_function(
                job_id=job_id,
                vr_performance_data=vr_performance_data,
                evolution_type=evolution_type,
            )

            await self.db_manager.update_job_status(
                job_id=job_id, status="evolution_completed", progress=100
            )

            return result

        except Exception as e:
            await self.db_manager.update_job_status(
                job_id=job_id, status="evolution_failed", error=str(e)
            )
            raise

    async def run_paper_generation_only(
        self,
        job_id: str,
        vr_performance_data: Dict[str, Any],
        paper_type: str = "vr_affordance_discovery",
        custom_focus: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run only paper generation (using existing evolved functions)"""

        try:
            print(
                f"INFO: Running paper generation for job {job_id}, type: {paper_type}"
            )

            # Get existing evolved functions
            evolved_functions = await self.funsearch.get_evolved_functions_for_job(
                job_id
            )

            if not evolved_functions:
                raise ValueError(
                    f"No evolved functions found for job {job_id}. Run evolution first."
                )

            await self.db_manager.update_job_status(
                job_id=job_id, status="paper_generation_running", progress=20
            )

            result = await self.ai_scientist.generate_research_paper(
                job_id=job_id,
                vr_performance_data=vr_performance_data,
                evolved_functions=evolved_functions,
                paper_type=paper_type,
                custom_focus=custom_focus,
            )

            await self.db_manager.update_job_status(
                job_id=job_id, status="paper_generation_completed", progress=100
            )

            return result

        except Exception as e:
            await self.db_manager.update_job_status(
                job_id=job_id, status="paper_generation_failed", error=str(e)
            )
            raise
