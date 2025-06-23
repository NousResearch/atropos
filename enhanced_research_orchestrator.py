"""
Enhanced Research Orchestrator for AMIEN
Integrates AI Scientist and FunSearch for autonomous discovery
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_scientist_manager import AIScientistManager
from bigquery_manager import ResearchDataManager
from enhanced_padres_perplexity import EnhancedPadresPerplexity
from funsearch_manager import FunSearchManager


class EnhancedResearchOrchestrator:
    """
    Orchestrates autonomous research using AI Scientist, FunSearch, and Padres API
    """

    def __init__(self, project_id: str, config: Dict[str, Any]):
        self.project_id = project_id
        self.config = config

        # Initialize managers
        self.ai_scientist = AIScientistManager(
            project_id, use_v2=config.get("use_ai_scientist_v2", False)
        )
        self.funsearch = FunSearchManager(project_id)
        self.padres = EnhancedPadresPerplexity()
        self.data_manager = ResearchDataManager(project_id)

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Discovery pipeline configuration
        self.discovery_config = {
            "daily_experiments": config.get("daily_experiments", 50),
            "papers_per_week": config.get("papers_per_week", 25),
            "functions_per_week": config.get("functions_per_week", 25),
            "max_parallel_discoveries": config.get("max_parallel_discoveries", 5),
            "discovery_cycle_hours": config.get("discovery_cycle_hours", 6),
        }

        # Cross-domain inspiration sources
        self.inspiration_domains = {
            "fireflies": {
                "keywords": ["bioluminescence", "swarm behavior", "light patterns"],
                "applications": ["visual_cues", "navigation", "attention_capture"],
            },
            "casino_psychology": {
                "keywords": ["attention capture", "reward systems", "engagement"],
                "applications": ["user_retention", "motivation", "feedback_systems"],
            },
            "nature_patterns": {
                "keywords": ["fractals", "organic movement", "natural navigation"],
                "applications": [
                    "ui_design",
                    "movement_patterns",
                    "spatial_organization",
                ],
            },
            "cognitive_science": {
                "keywords": ["spatial cognition", "memory", "perception"],
                "applications": [
                    "learning_systems",
                    "memory_aids",
                    "perception_enhancement",
                ],
            },
        }

        # Active discovery tracking
        self.active_discoveries = {
            "ai_scientist_experiments": [],
            "funsearch_experiments": [],
            "cross_domain_explorations": [],
        }

    async def setup_discovery_pipeline(self):
        """Set up the complete discovery pipeline"""
        try:
            self.logger.info("Setting up enhanced discovery pipeline...")

            # Setup AI Scientist
            self.ai_scientist.setup_repositories()

            # Setup FunSearch
            self.funsearch.setup_repository()

            # Initialize data storage
            await self.data_manager.initialize_storage()

            self.logger.info("Discovery pipeline setup complete")

        except Exception as e:
            self.logger.error(f"Failed to setup discovery pipeline: {e}")
            raise

    async def run_discovery_cycle(self):
        """Run a complete discovery cycle"""
        try:
            cycle_start = datetime.now()
            self.logger.info(f"Starting discovery cycle at {cycle_start}")

            # Phase 1: Collect recent experiment data
            recent_data = await self._collect_recent_experiment_data()

            # Phase 2: Run parallel discoveries
            discovery_tasks = []

            # AI Scientist paper generation
            if (
                len(self.active_discoveries["ai_scientist_experiments"])
                < self.discovery_config["max_parallel_discoveries"]
            ):
                discovery_tasks.append(self._run_ai_scientist_discovery(recent_data))

            # FunSearch function discovery
            if (
                len(self.active_discoveries["funsearch_experiments"])
                < self.discovery_config["max_parallel_discoveries"]
            ):
                discovery_tasks.append(self._run_funsearch_discovery(recent_data))

            # Cross-domain inspiration
            discovery_tasks.append(self._run_cross_domain_exploration(recent_data))

            # Execute discoveries in parallel
            if discovery_tasks:
                results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
                await self._process_discovery_results(results)

            # Phase 3: Generate new experiments based on discoveries
            await self._generate_follow_up_experiments(recent_data)

            cycle_duration = datetime.now() - cycle_start
            self.logger.info(f"Discovery cycle completed in {cycle_duration}")

        except Exception as e:
            self.logger.error(f"Discovery cycle failed: {e}")
            raise

    async def _collect_recent_experiment_data(self) -> Dict[str, Any]:
        """Collect recent experiment data from all sources"""
        try:
            # Get recent Padres experiments
            end_date = datetime.now()
            start_date = end_date - timedelta(
                hours=self.discovery_config["discovery_cycle_hours"]
            )

            padres_data = await self.data_manager.get_experiments_by_date_range(
                start_date.isoformat(), end_date.isoformat()
            )

            # Get recent AI Scientist papers
            recent_papers = self.ai_scientist.list_generated_papers(limit=20)

            # Get recent FunSearch functions
            recent_functions = self.funsearch.list_discovered_functions(limit=20)

            # Aggregate statistics
            stats = {
                "padres_experiments": len(padres_data),
                "ai_scientist_papers": len(recent_papers),
                "funsearch_functions": len(recent_functions),
                "collection_timestamp": datetime.now().isoformat(),
            }

            return {
                "padres_data": padres_data,
                "recent_papers": recent_papers,
                "recent_functions": recent_functions,
                "stats": stats,
            }

        except Exception as e:
            self.logger.error(f"Failed to collect recent data: {e}")
            return {}

    async def _run_ai_scientist_discovery(
        self, recent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run AI Scientist discovery process"""
        try:
            self.logger.info("Starting AI Scientist discovery...")

            # Determine research focus based on recent data
            research_focus = self._determine_research_focus(recent_data)

            # Generate research paper
            paper_result = self.ai_scientist.generate_research_paper(
                experiment_results=recent_data["padres_data"],
                template_name=research_focus["template"],
                num_ideas=research_focus["num_ideas"],
            )

            # Track active experiment
            experiment_id = f"ai_scientist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.active_discoveries["ai_scientist_experiments"].append(
                {
                    "id": experiment_id,
                    "start_time": datetime.now().isoformat(),
                    "focus": research_focus,
                    "status": "completed",
                    "result": paper_result,
                }
            )

            return {
                "type": "ai_scientist",
                "experiment_id": experiment_id,
                "result": paper_result,
                "success": True,
            }

        except Exception as e:
            self.logger.error(f"AI Scientist discovery failed: {e}")
            return {"type": "ai_scientist", "success": False, "error": str(e)}

    async def _run_funsearch_discovery(
        self, recent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run FunSearch discovery process"""
        try:
            self.logger.info("Starting FunSearch discovery...")

            # Determine function type to discover
            function_focus = self._determine_function_focus(recent_data)

            # Discover spatial function
            function_result = self.funsearch.discover_spatial_function(
                function_type=function_focus["type"],
                padres_data=recent_data["padres_data"],
                max_iterations=function_focus["max_iterations"],
            )

            # Track active experiment
            experiment_id = f"funsearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.active_discoveries["funsearch_experiments"].append(
                {
                    "id": experiment_id,
                    "start_time": datetime.now().isoformat(),
                    "focus": function_focus,
                    "status": "completed",
                    "result": function_result,
                }
            )

            return {
                "type": "funsearch",
                "experiment_id": experiment_id,
                "result": function_result,
                "success": True,
            }

        except Exception as e:
            self.logger.error(f"FunSearch discovery failed: {e}")
            return {"type": "funsearch", "success": False, "error": str(e)}

    async def _run_cross_domain_exploration(
        self, recent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run cross-domain inspiration exploration"""
        try:
            self.logger.info("Starting cross-domain exploration...")

            # Select inspiration domain
            domain = self._select_inspiration_domain(recent_data)

            # Generate cross-domain research questions
            research_questions = await self._generate_cross_domain_questions(
                domain, recent_data
            )

            # Run inspired experiments
            inspired_experiments = []
            for question in research_questions[:3]:  # Limit to 3 questions per cycle
                experiment = await self._run_inspired_experiment(
                    question, domain, recent_data
                )
                inspired_experiments.append(experiment)

            # Track exploration
            exploration_id = f"cross_domain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.active_discoveries["cross_domain_explorations"].append(
                {
                    "id": exploration_id,
                    "start_time": datetime.now().isoformat(),
                    "domain": domain,
                    "questions": research_questions,
                    "experiments": inspired_experiments,
                    "status": "completed",
                }
            )

            return {
                "type": "cross_domain",
                "exploration_id": exploration_id,
                "domain": domain,
                "experiments": inspired_experiments,
                "success": True,
            }

        except Exception as e:
            self.logger.error(f"Cross-domain exploration failed: {e}")
            return {"type": "cross_domain", "success": False, "error": str(e)}

    def _determine_research_focus(self, recent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine research focus for AI Scientist based on recent data"""

        # Analyze recent experiment patterns
        padres_data = recent_data.get("padres_data", [])

        if not padres_data:
            return {
                "template": "vr_affordances",
                "num_ideas": 3,
                "reasoning": "Default focus - no recent data",
            }

        # Simple heuristic: focus on area with most activity
        experiment_types = {}
        for exp in padres_data:
            exp_type = exp.get("experiment_type", "unknown")
            experiment_types[exp_type] = experiment_types.get(exp_type, 0) + 1

        if experiment_types:
            most_active = max(experiment_types, key=experiment_types.get)

            # Map experiment types to templates
            template_mapping = {
                "spatial_reasoning": "spatial_reasoning",
                "visual_cues": "visual_cues",
                "affordances": "vr_affordances",
            }

            template = template_mapping.get(most_active, "vr_affordances")
        else:
            template = "vr_affordances"

        return {
            "template": template,
            "num_ideas": 3,
            "reasoning": f"Focus on {template} based on recent activity",
        }

    def _determine_function_focus(self, recent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine function discovery focus for FunSearch"""

        # Cycle through different function types
        function_types = [
            "spatial_priority",
            "affordance_ranking",
            "visual_cue_priority",
            "navigation_heuristic",
        ]

        # Simple round-robin selection
        current_hour = datetime.now().hour
        selected_type = function_types[current_hour % len(function_types)]

        return {
            "type": selected_type,
            "max_iterations": 500,
            "reasoning": f"Round-robin selection: {selected_type}",
        }

    def _select_inspiration_domain(self, recent_data: Dict[str, Any]) -> str:
        """Select inspiration domain for cross-domain exploration"""

        # Cycle through inspiration domains
        domains = list(self.inspiration_domains.keys())
        current_day = datetime.now().day
        selected_domain = domains[current_day % len(domains)]

        return selected_domain

    async def _generate_cross_domain_questions(
        self, domain: str, recent_data: Dict[str, Any]
    ) -> List[str]:
        """Generate research questions inspired by cross-domain insights"""

        domain_info = self.inspiration_domains[domain]

        # Generate questions using Perplexity AI
        prompt = """
        Based on insights from {domain} (keywords: {', '.join(domain_info['keywords'])}),
        generate 5 research questions for VR spatial reasoning and affordance discovery.

        Recent experiment data shows:
        - {len(recent_data.get('padres_data', []))} spatial reasoning experiments
        - Focus areas: {', '.join(domain_info['applications'])}

        Generate questions that bridge {domain} concepts with VR spatial reasoning.
        """

        try:
            response = await self.padres.query_perplexity(prompt)

            # Parse questions from response
            questions = []
            lines = response.split("\n")
            for line in lines:
                if line.strip() and (
                    "?" in line or "How" in line or "What" in line or "Why" in line
                ):
                    questions.append(line.strip())

            return questions[:5]  # Return up to 5 questions

        except Exception as e:
            self.logger.error(f"Failed to generate cross-domain questions: {e}")
            return [
                f"How can {domain} principles improve VR spatial reasoning?",
                f"What {domain} patterns can enhance VR affordance discovery?",
                f"How do {domain} insights inform VR navigation algorithms?",
            ]

    async def _run_inspired_experiment(
        self, question: str, domain: str, recent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run an experiment inspired by cross-domain insights"""

        try:
            # Generate experiment design
            experiment_design = await self._design_inspired_experiment(question, domain)

            # Run Padres experiment
            padres_result = await self.padres.run_spatial_reasoning_experiment(
                experiment_config=experiment_design["config"],
                num_trials=experiment_design.get("num_trials", 10),
            )

            return {
                "question": question,
                "domain": domain,
                "design": experiment_design,
                "result": padres_result,
                "success": True,
            }

        except Exception as e:
            self.logger.error(f"Inspired experiment failed: {e}")
            return {
                "question": question,
                "domain": domain,
                "success": False,
                "error": str(e),
            }

    async def _design_inspired_experiment(
        self, question: str, domain: str
    ) -> Dict[str, Any]:
        """Design an experiment based on cross-domain inspiration"""

        domain_info = self.inspiration_domains[domain]

        # Generate experiment design using Perplexity AI
        prompt = """
        Design a VR spatial reasoning experiment to answer: "{question}"

        Inspiration domain: {domain}
        Domain keywords: {', '.join(domain_info['keywords'])}
        Applications: {', '.join(domain_info['applications'])}

        Provide:
        1. Experiment configuration (JSON format)
        2. Number of trials
        3. Expected outcomes
        4. Success metrics
        """

        try:
            response = await self.padres.query_perplexity(prompt)

            # Parse experiment design from response
            # This is a simplified version - would need more sophisticated parsing
            design = {
                "config": {
                    "experiment_type": "cross_domain_inspired",
                    "domain": domain,
                    "question": question,
                    "environment": "vr_lab",
                    "task": "spatial_reasoning",
                },
                "num_trials": 15,
                "expected_outcomes": f"Insights from {domain} applied to VR",
                "success_metrics": ["accuracy", "completion_time", "user_satisfaction"],
            }

            return design

        except Exception as e:
            self.logger.error(f"Failed to design inspired experiment: {e}")
            return {
                "config": {
                    "experiment_type": "cross_domain_inspired",
                    "domain": domain,
                    "question": question,
                },
                "num_trials": 10,
            }

    async def _process_discovery_results(self, results: List[Any]):
        """Process results from parallel discoveries"""

        successful_discoveries = []
        failed_discoveries = []

        for result in results:
            if isinstance(result, Exception):
                failed_discoveries.append(str(result))
            elif result.get("success", False):
                successful_discoveries.append(result)
            else:
                failed_discoveries.append(result.get("error", "Unknown error"))

        # Log results
        self.logger.info(
            f"Discovery results: {len(successful_discoveries)} successful, {len(failed_discoveries)} failed"
        )

        # Store results
        for discovery in successful_discoveries:
            await self._store_discovery_result(discovery)

        # Generate summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "successful_discoveries": len(successful_discoveries),
            "failed_discoveries": len(failed_discoveries),
            "discoveries": successful_discoveries,
        }

        await self.data_manager.store_discovery_summary(summary)

    async def _store_discovery_result(self, discovery: Dict[str, Any]):
        """Store individual discovery result"""

        try:
            discovery_type = discovery["type"]

            if discovery_type == "ai_scientist":
                await self.data_manager.store_ai_scientist_result(discovery)
            elif discovery_type == "funsearch":
                await self.data_manager.store_funsearch_result(discovery)
            elif discovery_type == "cross_domain":
                await self.data_manager.store_cross_domain_result(discovery)

            self.logger.info(f"Stored {discovery_type} discovery result")

        except Exception as e:
            self.logger.error(f"Failed to store discovery result: {e}")

    async def _generate_follow_up_experiments(self, recent_data: Dict[str, Any]):
        """Generate follow-up experiments based on discoveries"""

        try:
            # Analyze recent discoveries for patterns
            discovery_patterns = await self._analyze_discovery_patterns()

            # Generate new experiment hypotheses
            new_hypotheses = await self._generate_new_hypotheses(
                discovery_patterns, recent_data
            )

            # Queue new experiments
            for hypothesis in new_hypotheses:
                await self._queue_follow_up_experiment(hypothesis)

            self.logger.info(f"Generated {len(new_hypotheses)} follow-up experiments")

        except Exception as e:
            self.logger.error(f"Failed to generate follow-up experiments: {e}")

    async def _analyze_discovery_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in recent discoveries"""

        # This would implement sophisticated pattern analysis
        # For now, return a simple summary

        return {
            "ai_scientist_trends": ["spatial_reasoning", "visual_cues"],
            "funsearch_trends": ["priority_functions", "ranking_algorithms"],
            "cross_domain_trends": ["firefly_patterns", "casino_psychology"],
            "emerging_themes": [
                "attention_capture",
                "user_adaptation",
                "context_awareness",
            ],
        }

    async def _generate_new_hypotheses(
        self, patterns: Dict[str, Any], recent_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate new hypotheses based on discovery patterns"""

        # Generate hypotheses using Perplexity AI
        prompt = """
        Based on recent discovery patterns:
        - AI Scientist trends: {', '.join(patterns['ai_scientist_trends'])}
        - FunSearch trends: {', '.join(patterns['funsearch_trends'])}
        - Cross-domain trends: {', '.join(patterns['cross_domain_trends'])}
        - Emerging themes: {', '.join(patterns['emerging_themes'])}

        Generate 3 new research hypotheses for VR spatial reasoning experiments.
        Each hypothesis should be testable and build on the discovered patterns.
        """

        try:
            response = await self.padres.query_perplexity(prompt)

            # Parse hypotheses from response
            hypotheses = []
            lines = response.split("\n")
            for line in lines:
                if line.strip() and ("hypothesis" in line.lower() or "H" in line[:3]):
                    hypotheses.append(
                        {
                            "hypothesis": line.strip(),
                            "priority": "high",
                            "experiment_type": "follow_up",
                        }
                    )

            return hypotheses[:3]  # Return up to 3 hypotheses

        except Exception as e:
            self.logger.error(f"Failed to generate new hypotheses: {e}")
            return []

    async def _queue_follow_up_experiment(self, hypothesis: Dict[str, Any]):
        """Queue a follow-up experiment"""

        try:
            experiment_config = {
                "hypothesis": hypothesis["hypothesis"],
                "priority": hypothesis["priority"],
                "experiment_type": hypothesis["experiment_type"],
                "scheduled_time": (datetime.now() + timedelta(hours=1)).isoformat(),
                "status": "queued",
            }

            await self.data_manager.queue_experiment(experiment_config)
            self.logger.info(
                f"Queued follow-up experiment: {hypothesis['hypothesis'][:50]}..."
            )

        except Exception as e:
            self.logger.error(f"Failed to queue follow-up experiment: {e}")

    async def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive discovery statistics"""

        try:
            # Get statistics from all managers
            ai_scientist_stats = self.ai_scientist.get_paper_statistics()
            funsearch_stats = self.funsearch.get_function_statistics()

            # Get active discovery counts
            active_stats = {
                "active_ai_scientist": len(
                    self.active_discoveries["ai_scientist_experiments"]
                ),
                "active_funsearch": len(
                    self.active_discoveries["funsearch_experiments"]
                ),
                "active_cross_domain": len(
                    self.active_discoveries["cross_domain_explorations"]
                ),
            }

            # Calculate discovery rates
            discovery_rates = await self._calculate_discovery_rates()

            return {
                "ai_scientist": ai_scientist_stats,
                "funsearch": funsearch_stats,
                "active_discoveries": active_stats,
                "discovery_rates": discovery_rates,
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get discovery statistics: {e}")
            return {}

    async def _calculate_discovery_rates(self) -> Dict[str, float]:
        """Calculate discovery rates over time"""

        # This would implement rate calculation based on historical data
        # For now, return placeholder values

        return {
            "papers_per_day": 3.5,
            "functions_per_day": 3.2,
            "cross_domain_insights_per_day": 2.8,
            "total_discoveries_per_day": 9.5,
        }
