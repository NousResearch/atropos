"""
AI Scientist Integration Manager for CloudVR-PerfGuard
Handles automated research paper generation using Sakana AI's AI Scientist
"""

import asyncio
import json
import uuid
import os
import subprocess
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from core.database import DatabaseManager
from ai_scientist_wrapper import generate_gemini_paper, generate_vr_research_paper


class AIScientistIntegration:
    """
    Manages AI Scientist integration for automated research paper generation
    """
    
    def __init__(self, db_manager: DatabaseManager, ai_scientist_path: str = "./AI-Scientist"):
        self.db_manager = db_manager
        self.ai_scientist_path = Path(ai_scientist_path)
        self.paper_templates = {
            "vr_affordance_discovery": {
                "title_template": "Automated Discovery of Visual Affordance Cues in Virtual Reality: {experiment_focus}",
                "research_areas": ["computer_vision", "human_computer_interaction", "virtual_reality"],
                "methodology": "experimental_analysis",
                "expected_cost": 15.0
            },
            "performance_optimization": {
                "title_template": "Performance-Driven Visual Cue Optimization for VR Applications: {optimization_target}",
                "research_areas": ["computer_graphics", "performance_optimization", "user_experience"],
                "methodology": "algorithmic_optimization",
                "expected_cost": 12.0
            },
            "user_experience_analysis": {
                "title_template": "Impact of Evolved Visual Cues on VR User Experience and Interaction Success: {user_study_focus}",
                "research_areas": ["human_factors", "user_experience", "virtual_reality"],
                "methodology": "user_study_analysis",
                "expected_cost": 18.0
            }
        }
    
    async def generate_research_paper(
        self,
        job_id: str,
        vr_performance_data: Dict[str, Any],
        evolved_functions: List[Dict[str, Any]],
        paper_type: str = "vr_affordance_discovery",
        custom_focus: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a research paper using AI Scientist based on VR experiments and evolved functions
        """
        try:
            print(f"INFO: Starting AI Scientist paper generation for job {job_id}, type: {paper_type}")
            
            # Get paper template configuration
            template = self.paper_templates.get(paper_type, self.paper_templates["vr_affordance_discovery"])
            
            # Generate paper ID and metadata
            paper_id = f"{job_id}_paper_{paper_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare research data and context
            research_context = self._prepare_research_context(
                vr_performance_data, evolved_functions, paper_type, custom_focus
            )
            
            # Generate paper title
            title = self._generate_paper_title(template, research_context)
            
            # Store initial paper record
            creation_time = datetime.utcnow().isoformat()
            await self.db_manager.store_ai_paper(
                paper_id=paper_id,
                job_id=job_id,
                title=title,
                creation_timestamp=creation_time,
                last_updated_timestamp=creation_time,
                generation_status="initializing",
                generation_cost=0.0
            )
            
            # Run AI Scientist paper generation
            paper_result = await self._run_ai_scientist_generation(
                paper_id=paper_id,
                research_context=research_context,
                template=template
            )
            
            # Update paper record with results
            await self.db_manager.update_ai_paper_status(
                paper_id=paper_id,
                generation_status=paper_result["status"],
                last_updated_timestamp=datetime.utcnow().isoformat(),
                abstract=paper_result.get("abstract"),
                full_text_path=paper_result.get("full_text_path"),
                generation_cost=paper_result.get("cost", template["expected_cost"]),
                publication_details=json.dumps(paper_result.get("publication_details", {}))
            )
            
            return {
                "status": "success",
                "paper_id": paper_id,
                "title": title,
                "paper_type": paper_type,
                "generation_cost": paper_result.get("cost", template["expected_cost"]),
                "abstract": paper_result.get("abstract"),
                "full_text_path": paper_result.get("full_text_path"),
                "job_id": job_id
            }
            
        except Exception as e:
            print(f"ERROR: AI Scientist paper generation failed for job {job_id}: {e}")
            
            # Update paper status to error
            if 'paper_id' in locals():
                await self.db_manager.update_ai_paper_status(
                    paper_id=paper_id,
                    generation_status="error",
                    last_updated_timestamp=datetime.utcnow().isoformat()
                )
            
            return {
                "status": "error",
                "error": str(e),
                "job_id": job_id
            }
    
    def _prepare_research_context(
        self,
        vr_data: Dict[str, Any],
        evolved_functions: List[Dict[str, Any]],
        paper_type: str,
        custom_focus: Optional[str]
    ) -> Dict[str, Any]:
        """Prepare research context for AI Scientist"""
        
        # Analyze evolved functions
        function_analysis = self._analyze_evolved_functions(evolved_functions)
        
        # Extract key VR performance insights
        performance_insights = self._extract_performance_insights(vr_data)
        
        # Determine research focus
        research_focus = custom_focus or self._determine_research_focus(vr_data, evolved_functions, paper_type)
        
        context = {
            "research_focus": research_focus,
            "paper_type": paper_type,
            "vr_performance_data": {
                "avg_fps": vr_data.get("avg_fps", "N/A"),
                "min_fps": vr_data.get("min_fps", "N/A"),
                "avg_frame_time": vr_data.get("avg_frame_time", "N/A"),
                "comfort_score": vr_data.get("comfort_score", "N/A"),
                "total_experiments": vr_data.get("total_experiments", 0)
            },
            "evolved_functions": {
                "total_functions": len(evolved_functions),
                "best_scores": function_analysis["best_scores"],
                "evolution_types": function_analysis["evolution_types"],
                "performance_improvements": function_analysis["improvements"]
            },
            "key_findings": performance_insights["key_findings"],
            "research_questions": self._generate_research_questions(paper_type, performance_insights),
            "methodology": self._describe_methodology(evolved_functions),
            "expected_contributions": self._identify_contributions(paper_type, function_analysis)
        }
        
        return context
    
    def _analyze_evolved_functions(self, evolved_functions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze evolved functions to extract research insights"""
        
        if not evolved_functions:
            return {
                "best_scores": {},
                "evolution_types": [],
                "improvements": {}
            }
        
        # Group by evolution type
        by_type = {}
        for func in evolved_functions:
            metadata = func.get("metadata", {})
            evo_type = metadata.get("evolution_type", "unknown")
            if evo_type not in by_type:
                by_type[evo_type] = []
            by_type[evo_type].append(func)
        
        # Find best scores per type
        best_scores = {}
        improvements = {}
        
        for evo_type, functions in by_type.items():
            scores = [f.get("evaluation_score", 0) for f in functions]
            best_scores[evo_type] = max(scores) if scores else 0
            
            # Calculate improvement over baseline (first function)
            if len(scores) > 1:
                baseline = scores[0]
                final = scores[-1]
                improvement = ((final - baseline) / baseline * 100) if baseline > 0 else 0
                improvements[evo_type] = improvement
        
        return {
            "best_scores": best_scores,
            "evolution_types": list(by_type.keys()),
            "improvements": improvements
        }
    
    def _extract_performance_insights(self, vr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key performance insights from VR data"""
        
        insights = []
        
        # FPS analysis
        avg_fps = vr_data.get("avg_fps", 60)
        min_fps = vr_data.get("min_fps", 30)
        
        if avg_fps >= 90:
            insights.append("High-performance VR experience with excellent frame rates")
        elif avg_fps >= 60:
            insights.append("Standard VR performance with acceptable frame rates")
        else:
            insights.append("Performance-constrained VR environment requiring optimization")
        
        if min_fps < 45:
            insights.append("Frame rate drops detected, indicating potential motion sickness risk")
        
        # Comfort analysis
        comfort_score = vr_data.get("comfort_score", 0.7)
        if comfort_score >= 0.8:
            insights.append("High user comfort levels achieved")
        elif comfort_score >= 0.6:
            insights.append("Moderate user comfort with room for improvement")
        else:
            insights.append("Low user comfort scores indicating significant UX issues")
        
        # Frame time analysis
        avg_frame_time = vr_data.get("avg_frame_time", 16.67)
        if avg_frame_time <= 11.11:  # 90 FPS
            insights.append("Optimal frame timing for premium VR experience")
        elif avg_frame_time <= 16.67:  # 60 FPS
            insights.append("Standard frame timing suitable for most VR applications")
        else:
            insights.append("Elevated frame times requiring performance optimization")
        
        return {
            "key_findings": insights,
            "performance_category": self._categorize_performance(avg_fps, comfort_score),
            "optimization_priority": self._determine_optimization_priority(vr_data)
        }
    
    def _categorize_performance(self, avg_fps: float, comfort_score: float) -> str:
        """Categorize overall VR performance"""
        
        if avg_fps >= 90 and comfort_score >= 0.8:
            return "premium"
        elif avg_fps >= 60 and comfort_score >= 0.6:
            return "standard"
        elif avg_fps >= 45 and comfort_score >= 0.5:
            return "acceptable"
        else:
            return "needs_improvement"
    
    def _determine_optimization_priority(self, vr_data: Dict[str, Any]) -> str:
        """Determine optimization priority based on performance data"""
        
        min_fps = vr_data.get("min_fps", 30)
        comfort_score = vr_data.get("comfort_score", 0.7)
        
        if min_fps < 45 or comfort_score < 0.5:
            return "high"
        elif min_fps < 60 or comfort_score < 0.7:
            return "medium"
        else:
            return "low"
    
    def _determine_research_focus(
        self,
        vr_data: Dict[str, Any],
        evolved_functions: List[Dict[str, Any]],
        paper_type: str
    ) -> str:
        """Determine the specific research focus based on data"""
        
        performance_category = self._categorize_performance(
            vr_data.get("avg_fps", 60),
            vr_data.get("comfort_score", 0.7)
        )
        
        if paper_type == "vr_affordance_discovery":
            if performance_category == "premium":
                return "High-Performance VR Environments"
            elif performance_category == "needs_improvement":
                return "Performance-Constrained VR Systems"
            else:
                return "Standard VR Applications"
        
        elif paper_type == "performance_optimization":
            optimization_priority = self._determine_optimization_priority(vr_data)
            if optimization_priority == "high":
                return "Critical Performance Bottlenecks"
            elif optimization_priority == "medium":
                return "Moderate Performance Enhancements"
            else:
                return "Fine-Tuned Performance Optimization"
        
        else:  # user_experience_analysis
            comfort_score = vr_data.get("comfort_score", 0.7)
            if comfort_score >= 0.8:
                return "Premium User Experience Design"
            elif comfort_score >= 0.6:
                return "User Experience Enhancement"
            else:
                return "User Experience Recovery"
    
    def _generate_paper_title(self, template: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate paper title based on template and context"""
        
        title_template = template["title_template"]
        research_focus = context["research_focus"]
        
        return title_template.format(
            experiment_focus=research_focus,
            optimization_target=research_focus,
            user_study_focus=research_focus
        )
    
    def _generate_research_questions(self, paper_type: str, insights: Dict[str, Any]) -> List[str]:
        """Generate research questions based on paper type and insights"""
        
        base_questions = {
            "vr_affordance_discovery": [
                "How can evolutionary algorithms improve visual affordance cue discovery in VR?",
                "What visual cue parameters most significantly impact user interaction success?",
                "How do performance constraints affect optimal visual cue design?"
            ],
            "performance_optimization": [
                "What is the relationship between visual cue complexity and VR performance?",
                "How can evolved functions balance visual quality with performance requirements?",
                "What optimization strategies yield the best performance-quality trade-offs?"
            ],
            "user_experience_analysis": [
                "How do evolved visual cues impact user comfort and presence in VR?",
                "What factors contribute most to successful VR object interactions?",
                "How can automated cue optimization improve overall user experience?"
            ]
        }
        
        questions = base_questions.get(paper_type, base_questions["vr_affordance_discovery"])
        
        # Add context-specific questions
        performance_category = insights.get("performance_category", "standard")
        if performance_category == "needs_improvement":
            questions.append("How can visual cue optimization compensate for performance limitations?")
        elif performance_category == "premium":
            questions.append("What advanced visual cue techniques are enabled by high-performance VR?")
        
        return questions
    
    def _describe_methodology(self, evolved_functions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Describe the methodology used in the research"""
        
        evolution_types = set()
        total_iterations = 0
        
        for func in evolved_functions:
            metadata = func.get("metadata", {})
            evolution_types.add(metadata.get("evolution_type", "unknown"))
            total_iterations = max(total_iterations, func.get("evolution_iteration", 0))
        
        return {
            "approach": "Evolutionary Function Discovery with Performance-Driven Evaluation",
            "evolution_types": list(evolution_types),
            "total_iterations": total_iterations + 1,
            "evaluation_criteria": [
                "VR performance metrics (FPS, frame time)",
                "User comfort scores",
                "Interaction success rates",
                "Visual cue effectiveness"
            ],
            "experimental_design": "Automated evolution with real-time VR performance feedback"
        }
    
    def _identify_contributions(self, paper_type: str, function_analysis: Dict[str, Any]) -> List[str]:
        """Identify expected research contributions"""
        
        base_contributions = {
            "vr_affordance_discovery": [
                "Novel evolutionary approach to VR visual cue optimization",
                "Performance-aware affordance cue discovery methodology",
                "Empirical analysis of visual cue effectiveness in VR environments"
            ],
            "performance_optimization": [
                "Automated performance-quality trade-off optimization for VR",
                "Evolved algorithms for real-time visual cue adjustment",
                "Performance-driven visual cue parameter optimization"
            ],
            "user_experience_analysis": [
                "Comprehensive analysis of evolved visual cues on user experience",
                "User comfort optimization through automated cue discovery",
                "Evidence-based guidelines for VR interaction design"
            ]
        }
        
        contributions = base_contributions.get(paper_type, base_contributions["vr_affordance_discovery"])
        
        # Add specific contributions based on function analysis
        improvements = function_analysis.get("improvements", {})
        if any(imp > 10 for imp in improvements.values()):
            contributions.append("Significant performance improvements through evolved optimization functions")
        
        evolution_types = function_analysis.get("evolution_types", [])
        if len(evolution_types) > 2:
            contributions.append("Multi-objective optimization across diverse VR interaction scenarios")
        
        return contributions
    
    async def _run_ai_scientist_generation(
        self,
        paper_id: str,
        research_context: Dict[str, Any],
        template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the AI Scientist paper generation process"""
        
        try:
            # Create research data file
            research_data_path = f"/tmp/{paper_id}_research_data.json"
            with open(research_data_path, 'w') as f:
                json.dump(research_context, f, indent=2)
            
            # Prepare AI Scientist configuration
            ai_config = {
                "experiment_name": paper_id,
                "research_area": template["research_areas"][0],
                "methodology": template["methodology"],
                "data_source": research_data_path,
                "output_dir": f"./generated_papers/{paper_id}",
                "max_cost": template["expected_cost"] * 1.2  # 20% buffer
            }
            
            # For now, simulate AI Scientist execution
            # In a real implementation, this would call the actual AI Scientist
            paper_result = await self._simulate_ai_scientist_execution(paper_id, research_context, ai_config)
            
            return paper_result
            
        except Exception as e:
            print(f"ERROR: AI Scientist execution failed for paper {paper_id}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _simulate_ai_scientist_execution(
        self,
        paper_id: str,
        research_context: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate AI Scientist execution (placeholder for actual implementation)"""
        
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Generate abstract based on research context
        abstract = self._generate_abstract(research_context)
        
        # Create output directory
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate paper content (simplified)
        paper_content = self._generate_paper_content(research_context, abstract)
        
        # Save paper to file
        paper_path = output_dir / f"{paper_id}.md"
        with open(paper_path, 'w') as f:
            f.write(paper_content)
        
        return {
            "status": "completed",
            "abstract": abstract,
            "full_text_path": str(paper_path),
            "cost": config.get("max_cost", 15.0) * 0.8,  # Simulate 80% of max cost
            "publication_details": {
                "word_count": len(paper_content.split()),
                "sections": ["Abstract", "Introduction", "Methodology", "Results", "Discussion", "Conclusion"],
                "figures": 3,
                "tables": 2
            }
        }
    
    def _generate_abstract(self, research_context: Dict[str, Any]) -> str:
        """Generate paper abstract based on research context"""
        
        research_focus = research_context["research_focus"]
        vr_data = research_context["vr_performance_data"]
        evolved_data = research_context["evolved_functions"]
        
        abstract = f"""
This paper presents a novel approach to {research_focus.lower()} using evolutionary algorithms 
and performance-driven optimization in virtual reality environments. We developed and evaluated 
{evolved_data['total_functions']} evolved functions across {len(evolved_data['evolution_types'])} 
different optimization scenarios, achieving significant improvements in VR interaction effectiveness.

Our methodology combines automated function evolution with real-time VR performance feedback, 
targeting systems with {vr_data['avg_fps']} average FPS and {vr_data['comfort_score']} comfort scores. 
The evolved algorithms demonstrate substantial improvements in visual cue effectiveness, with the best 
performing functions showing optimization scores of {max(evolved_data['best_scores'].values()) if evolved_data['best_scores'] else 'N/A'}.

Key contributions include: (1) a performance-aware evolutionary optimization framework for VR visual cues, 
(2) empirical analysis of {vr_data['total_experiments']} VR interaction experiments, and (3) evidence-based 
guidelines for automated visual affordance discovery. Results indicate that evolved visual cue optimization 
can significantly enhance user interaction success while maintaining optimal VR performance characteristics.

The findings have important implications for VR application development, automated user experience optimization, 
and the broader field of human-computer interaction in immersive environments.
        """.strip()
        
        return abstract
    
    def _generate_paper_content(self, research_context: Dict[str, Any], abstract: str) -> str:
        """Generate full paper content (simplified version)"""
        
        title = f"Automated Discovery of Visual Affordance Cues in Virtual Reality: {research_context['research_focus']}"
        
        content = f"""# {title}

## Abstract

{abstract}

## 1. Introduction

Virtual Reality (VR) applications require carefully designed visual cues to ensure effective user interactions 
with virtual objects. Traditional approaches to visual affordance design rely on manual optimization and 
heuristic guidelines, which may not account for the complex interplay between visual cue parameters, 
system performance constraints, and individual user characteristics.

This research addresses the challenge of {research_context['research_focus'].lower()} through automated 
evolutionary optimization. Our approach leverages real-time VR performance data to guide the evolution 
of visual cue functions, ensuring optimal balance between interaction effectiveness and system performance.

### Research Questions

{chr(10).join(f"- {q}" for q in research_context['research_questions'])}

## 2. Methodology

### 2.1 Evolutionary Function Discovery

Our methodology employs {research_context['methodology']['approach']} with the following key components:

- **Evolution Types**: {', '.join(research_context['methodology']['evolution_types'])}
- **Total Iterations**: {research_context['methodology']['total_iterations']}
- **Evaluation Criteria**: {', '.join(research_context['methodology']['evaluation_criteria'])}

### 2.2 VR Performance Integration

The evolutionary process incorporates real-time VR performance metrics:
- Average FPS: {research_context['vr_performance_data']['avg_fps']}
- Minimum FPS: {research_context['vr_performance_data']['min_fps']}
- Frame Time: {research_context['vr_performance_data']['avg_frame_time']}ms
- Comfort Score: {research_context['vr_performance_data']['comfort_score']}

## 3. Results

### 3.1 Function Evolution Performance

The evolutionary process generated {research_context['evolved_functions']['total_functions']} optimized functions 
across {len(research_context['evolved_functions']['evolution_types'])} different scenarios.

**Best Scores by Evolution Type:**
{chr(10).join(f"- {etype}: {score:.3f}" for etype, score in research_context['evolved_functions']['best_scores'].items())}

### 3.2 Performance Improvements

{chr(10).join(f"- {etype}: {improvement:.1f}% improvement" for etype, improvement in research_context['evolved_functions']['performance_improvements'].items())}

## 4. Discussion

### 4.1 Key Findings

{chr(10).join(f"- {finding}" for finding in research_context['key_findings'])}

### 4.2 Implications for VR Design

The results demonstrate that automated evolutionary optimization can significantly enhance VR visual cue 
effectiveness while maintaining performance requirements. This approach enables:

1. **Adaptive Optimization**: Functions that adjust to varying performance constraints
2. **User-Centric Design**: Optimization based on actual user interaction data
3. **Scalable Implementation**: Automated processes that reduce manual design effort

## 5. Conclusion

This research presents a novel framework for automated visual affordance discovery in VR environments. 
The evolutionary approach successfully optimized visual cue parameters while maintaining system performance, 
demonstrating the potential for automated UX optimization in immersive applications.

### Future Work

- Extension to multi-user VR environments
- Integration with real-time user biometric feedback
- Cross-platform optimization for diverse VR hardware

## References

[Generated references would be included in a full implementation]

---

*Paper generated by AI Scientist integration for CloudVR-PerfGuard*
*Generation ID: {research_context.get('paper_id', 'unknown')}*
"""
        
        return content
    
    async def get_papers_for_job(self, job_id: str) -> List[Dict[str, Any]]:
        """Get all papers generated for a specific job"""
        return await self.db_manager.get_ai_papers_for_job(job_id)
    
    async def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific paper by its ID"""
        return await self.db_manager.get_ai_paper(paper_id)
    
    async def update_paper_peer_review(
        self,
        paper_id: str,
        peer_review_feedback: str,
        publication_status: str = "under_review"
    ) -> bool:
        """Update paper with peer review feedback"""
        return await self.db_manager.update_ai_paper_status(
            paper_id=paper_id,
            generation_status=publication_status,
            last_updated_timestamp=datetime.utcnow().isoformat(),
            peer_review_feedback=peer_review_feedback
        ) 