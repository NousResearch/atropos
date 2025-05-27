"""
Research Paper Generator for CloudVR-PerfGuard
Interfaces with AI Scientist for automated research paper generation
"""

import os
import json
import subprocess
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class ResearchPaperGenerator:
    """
    Generates research papers from VR performance data using AI Scientist
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.ai_scientist_path = self.config.get("ai_scientist_path", "../AI-Scientist")
        self.output_dir = self.config.get("output_dir", "generated_papers")
        self.max_cost_per_paper = self.config.get("max_cost_per_paper", 20.0)
        
        # Paper templates for different research types
        self.paper_templates = {
            "performance_analysis": {
                "title": "Performance Analysis of VR Applications: A Systematic Study",
                "focus": "VR performance metrics and optimization opportunities",
                "methodology": "Automated performance testing with statistical analysis"
            },
            "regression_study": {
                "title": "Performance Regression Detection in VR Applications",
                "focus": "Automated detection and analysis of performance regressions",
                "methodology": "Statistical regression analysis with effect size calculations"
            },
            "comparative_study": {
                "title": "Comparative Performance Analysis Across VR Hardware Configurations",
                "focus": "Performance differences across GPU types and configurations",
                "methodology": "Controlled comparative testing with statistical validation"
            }
        }
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def check_ai_scientist_availability(self) -> bool:
        """Check if AI Scientist is available and properly configured"""
        
        # Try multiple possible paths
        possible_paths = [
            self.ai_scientist_path,
            "AI-Scientist",
            "../AI-Scientist",
            "../../AI-Scientist"
        ]
        
        for path in possible_paths:
            ai_scientist_dir = Path(path)
            if ai_scientist_dir.exists():
                self.ai_scientist_path = str(ai_scientist_dir)
                break
        else:
            print(f"AI Scientist not found in any of: {possible_paths}")
            print("Please clone: git clone https://github.com/SakanaAI/AI-Scientist.git")
            return False
        
        # Check for required files
        required_files = ["launch_scientist.py", "requirements.txt"]
        for file in required_files:
            if not (ai_scientist_dir / file).exists():
                print(f"Required file {file} not found in AI Scientist directory")
                return False
        
        return True
    
    def generate_paper(
        self, 
        research_data: Dict[str, Any], 
        paper_type: str = "performance_analysis",
        custom_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a research paper from VR performance data
        
        Args:
            research_data: Formatted research data from PerformanceDataAdapter
            paper_type: Type of paper to generate
            custom_title: Custom title for the paper
            
        Returns:
            Paper generation results including metadata and content
        """
        
        if paper_type not in self.paper_templates:
            raise ValueError(f"Unknown paper type: {paper_type}")
        
        template = self.paper_templates[paper_type]
        
        # Prepare paper specification
        paper_spec = self._create_paper_specification(research_data, template, custom_title)
        
        # Check if AI Scientist is available
        if self.check_ai_scientist_availability():
            # Use actual AI Scientist
            result = self._generate_with_ai_scientist(paper_spec, research_data)
        else:
            # Use fallback paper generation
            result = self._generate_fallback_paper(paper_spec, research_data)
        
        # Store the generated paper
        paper_id = self._store_paper(result)
        result["paper_id"] = paper_id
        
        return result
    
    def _create_paper_specification(
        self, 
        research_data: Dict[str, Any], 
        template: Dict[str, Any],
        custom_title: Optional[str]
    ) -> Dict[str, Any]:
        """Create a specification for paper generation"""
        
        experiment_metadata = research_data.get("experiment_metadata", {})
        performance_metrics = research_data.get("performance_metrics", {})
        
        return {
            "title": custom_title or template["title"],
            "abstract_focus": template["focus"],
            "methodology": template["methodology"],
            "research_questions": [
                f"What are the performance characteristics of {experiment_metadata.get('app_name', 'the VR application')}?",
                "How does performance vary across different hardware configurations?",
                "What optimization opportunities can be identified from the data?",
                "How reliable are the performance measurements?"
            ],
            "key_findings": self._extract_key_findings(research_data),
            "statistical_summary": self._create_statistical_summary(performance_metrics),
            "data_overview": {
                "test_count": experiment_metadata.get("test_count", 0),
                "success_rate": experiment_metadata.get("success_rate", 0),
                "test_duration": experiment_metadata.get("test_duration", 0)
            }
        }
    
    def _extract_key_findings(self, research_data: Dict[str, Any]) -> List[str]:
        """Extract key findings from the research data"""
        
        findings = []
        performance_metrics = research_data.get("performance_metrics", {})
        
        # FPS analysis
        fps_stats = performance_metrics.get("fps_statistics", {})
        if fps_stats.get("mean", 0) > 0:
            findings.append(f"Average frame rate: {fps_stats['mean']:.1f} FPS (std: {fps_stats.get('std', 0):.1f})")
        
        # Frame time analysis
        frame_time_stats = performance_metrics.get("frame_time_statistics", {})
        if frame_time_stats.get("p95", 0) > 0:
            findings.append(f"95th percentile frame time: {frame_time_stats['p95']:.1f}ms")
        
        # GPU performance breakdown
        gpu_breakdown = performance_metrics.get("gpu_breakdown", {})
        if gpu_breakdown:
            best_gpu = max(gpu_breakdown.items(), key=lambda x: x[1].get("avg_fps", 0))
            findings.append(f"Best performing GPU: {best_gpu[0]} ({best_gpu[1].get('avg_fps', 0):.1f} FPS)")
        
        # Comfort score analysis
        comfort_stats = performance_metrics.get("comfort_score_statistics", {})
        if comfort_stats.get("mean", 0) > 0:
            findings.append(f"Average VR comfort score: {comfort_stats['mean']:.1f}/100")
        
        return findings
    
    def _create_statistical_summary(self, performance_metrics: Dict[str, Any]) -> str:
        """Create a statistical summary for the paper"""
        
        summary_parts = []
        
        # FPS summary
        fps_stats = performance_metrics.get("fps_statistics", {})
        if fps_stats.get("count", 0) > 0:
            summary_parts.append(
                f"Frame rate analysis (n={fps_stats['count']}): "
                f"M={fps_stats.get('mean', 0):.1f}, SD={fps_stats.get('std', 0):.1f}, "
                f"Range=[{fps_stats.get('min', 0):.1f}, {fps_stats.get('max', 0):.1f}]"
            )
        
        # Frame time summary
        frame_time_stats = performance_metrics.get("frame_time_statistics", {})
        if frame_time_stats.get("count", 0) > 0:
            summary_parts.append(
                f"Frame time analysis: "
                f"M={frame_time_stats.get('mean', 0):.1f}ms, "
                f"P95={frame_time_stats.get('p95', 0):.1f}ms"
            )
        
        return "; ".join(summary_parts)
    
    def _generate_with_ai_scientist(
        self, 
        paper_spec: Dict[str, Any], 
        research_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate paper using actual AI Scientist"""
        
        try:
            # Create temporary directory for AI Scientist work
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prepare input files for AI Scientist
                data_file = os.path.join(temp_dir, "research_data.json")
                spec_file = os.path.join(temp_dir, "paper_spec.json")
                
                with open(data_file, 'w') as f:
                    json.dump(research_data, f, indent=2)
                
                with open(spec_file, 'w') as f:
                    json.dump(paper_spec, f, indent=2)
                
                # Run AI Scientist via wrapper
                wrapper_path = os.path.join(os.path.dirname(__file__), "..", "..", "ai_scientist_wrapper.py")
                cmd = [
                    "python", 
                    wrapper_path,
                    "--data-file", data_file,
                    "--spec-file", spec_file,
                    "--output-dir", temp_dir,
                    "--max-cost", str(self.max_cost_per_paper)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    # Parse AI Scientist output
                    output_file = os.path.join(temp_dir, "generated_paper.json")
                    if os.path.exists(output_file):
                        with open(output_file, 'r') as f:
                            ai_result = json.load(f)
                        
                        return {
                            "title": ai_result.get("title", paper_spec["title"]),
                            "abstract": ai_result.get("abstract", ""),
                            "content": ai_result.get("content", ""),
                            "generation_method": "ai_scientist",
                            "generation_cost": ai_result.get("cost", 0),
                            "quality_score": ai_result.get("quality_score", 0),
                            "generation_time": datetime.utcnow().isoformat()
                        }
                
                # If AI Scientist failed, fall back to template generation
                print(f"AI Scientist failed: {result.stderr}")
                return self._generate_fallback_paper(paper_spec, research_data)
        
        except Exception as e:
            print(f"Error running AI Scientist: {e}")
            return self._generate_fallback_paper(paper_spec, research_data)
    
    def _generate_fallback_paper(
        self, 
        paper_spec: Dict[str, Any], 
        research_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate paper using template-based approach when AI Scientist is unavailable"""
        
        # Generate paper content using templates
        title = paper_spec["title"]
        abstract = self._generate_abstract(paper_spec, research_data)
        content = self._generate_content(paper_spec, research_data)
        
        return {
            "title": title,
            "abstract": abstract,
            "content": content,
            "generation_method": "template_based",
            "generation_cost": 0,
            "quality_score": 75.0,  # Estimated quality for template-based generation
            "generation_time": datetime.utcnow().isoformat()
        }
    
    def _generate_abstract(self, paper_spec: Dict[str, Any], research_data: Dict[str, Any]) -> str:
        """Generate paper abstract"""
        
        experiment_metadata = research_data.get("experiment_metadata", {})
        key_findings = paper_spec.get("key_findings", [])
        
        abstract = f"""
        This study presents a comprehensive performance analysis of {experiment_metadata.get('app_name', 'VR applications')} 
        using automated testing methodologies. We conducted {experiment_metadata.get('test_count', 0)} performance tests 
        with a {experiment_metadata.get('success_rate', 0)*100:.1f}% success rate. 
        
        Key findings include: {'; '.join(key_findings[:3])}. 
        
        The results provide insights into VR performance characteristics and identify optimization opportunities 
        for improved user experience. Our methodology demonstrates the effectiveness of automated performance 
        testing for VR application analysis.
        """
        
        return abstract.strip()
    
    def _generate_content(self, paper_spec: Dict[str, Any], research_data: Dict[str, Any]) -> str:
        """Generate full paper content"""
        
        sections = [
            "# Introduction",
            "Virtual Reality applications require consistent high performance to maintain user comfort and prevent motion sickness.",
            "This study employs automated testing to analyze VR performance characteristics.",
            "",
            "# Methodology",
            f"We used {paper_spec['methodology']} to collect performance data.",
            f"Data collection focused on {paper_spec['abstract_focus']}.",
            "",
            "# Results",
            "## Performance Metrics",
            paper_spec.get("statistical_summary", "Statistical analysis of performance data."),
            "",
            "## Key Findings",
            "\n".join(f"- {finding}" for finding in paper_spec.get("key_findings", [])),
            "",
            "# Discussion",
            "The results indicate several opportunities for performance optimization.",
            "Frame rate consistency appears to be a critical factor for VR comfort.",
            "",
            "# Conclusion",
            "This study demonstrates the value of automated VR performance testing.",
            "The findings provide actionable insights for VR application optimization."
        ]
        
        return "\n".join(sections)
    
    def _store_paper(self, paper_result: Dict[str, Any]) -> str:
        """Store generated paper to filesystem"""
        
        paper_id = f"vr_paper_{int(datetime.utcnow().timestamp())}"
        paper_dir = Path(self.output_dir) / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        
        # Save paper content
        with open(paper_dir / "paper.md", "w") as f:
            f.write(f"# {paper_result['title']}\n\n")
            f.write(f"## Abstract\n{paper_result['abstract']}\n\n")
            f.write(paper_result["content"])
        
        # Save metadata
        with open(paper_dir / "metadata.json", "w") as f:
            json.dump(paper_result, f, indent=2)
        
        return paper_id
    
    def list_generated_papers(self) -> List[Dict[str, Any]]:
        """List all generated papers"""
        
        papers = []
        output_path = Path(self.output_dir)
        
        if not output_path.exists():
            return papers
        
        for paper_dir in output_path.iterdir():
            if paper_dir.is_dir() and paper_dir.name.startswith("vr_paper_"):
                metadata_file = paper_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        papers.append({
                            "paper_id": paper_dir.name,
                            "title": metadata.get("title", "Unknown"),
                            "generation_method": metadata.get("generation_method", "unknown"),
                            "generation_cost": metadata.get("generation_cost", 0),
                            "quality_score": metadata.get("quality_score", 0),
                            "generation_time": metadata.get("generation_time", "unknown")
                        })
                    except Exception as e:
                        print(f"Error reading metadata for {paper_dir.name}: {e}")
        
        return sorted(papers, key=lambda x: x["generation_time"], reverse=True)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about paper generation"""
        
        papers = self.list_generated_papers()
        
        if not papers:
            return {
                "total_papers": 0,
                "total_cost": 0,
                "average_quality": 0,
                "generation_methods": {}
            }
        
        total_cost = sum(p["generation_cost"] for p in papers)
        avg_quality = sum(p["quality_score"] for p in papers) / len(papers)
        
        methods = {}
        for paper in papers:
            method = paper["generation_method"]
            methods[method] = methods.get(method, 0) + 1
        
        return {
            "total_papers": len(papers),
            "total_cost": total_cost,
            "average_cost": total_cost / len(papers),
            "average_quality": avg_quality,
            "generation_methods": methods,
            "latest_paper": papers[0] if papers else None
        } 