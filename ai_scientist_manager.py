"""
AI Scientist Integration Manager for AMIEN
Handles both AI Scientist v1 (template-based) and v2 (agentic tree search)
"""

import os
import json
import subprocess
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from google.cloud import storage
from google.cloud import secretmanager

class AIScientistManager:
    """
    Manages AI Scientist v1 and v2 integration with AMIEN pipeline
    """
    
    def __init__(self, project_id: str, use_v2: bool = False):
        self.project_id = project_id
        self.use_v2 = use_v2
        self.storage_client = storage.Client()
        self.secret_client = secretmanager.SecretManagerServiceClient()
        
        # Set up paths
        self.base_path = Path("/app/ai_scientist")
        self.v1_path = self.base_path / "AI-Scientist"
        self.v2_path = self.base_path / "AI-Scientist-v2"
        
        # GCS buckets
        self.papers_bucket = f"{project_id}-research-papers"
        self.data_bucket = f"{project_id}-research-data"
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load API keys
        self._load_api_keys()
        
        # VR/Spatial reasoning templates for v1
        self.spatial_templates = {
            "vr_affordances": {
                "description": "VR affordance discovery and optimization",
                "skeleton": self._create_vr_affordance_template(),
                "evaluation_metrics": ["accuracy", "user_satisfaction", "discovery_rate"]
            },
            "spatial_reasoning": {
                "description": "Spatial reasoning algorithm development",
                "skeleton": self._create_spatial_reasoning_template(),
                "evaluation_metrics": ["spatial_accuracy", "processing_speed", "generalization"]
            },
            "visual_cues": {
                "description": "Visual cue evolution for VR environments",
                "skeleton": self._create_visual_cue_template(),
                "evaluation_metrics": ["attention_capture", "navigation_improvement", "user_preference"]
            }
        }
    
    def _load_api_keys(self):
        """Load API keys from Google Secret Manager"""
        try:
            # OpenAI API Key
            openai_secret = f"projects/{self.project_id}/secrets/openai-api-key/versions/latest"
            openai_response = self.secret_client.access_secret_version(request={"name": openai_secret})
            os.environ["OPENAI_API_KEY"] = openai_response.payload.data.decode("UTF-8")
            
            # Anthropic API Key (for Claude)
            anthropic_secret = f"projects/{self.project_id}/secrets/anthropic-api-key/versions/latest"
            anthropic_response = self.secret_client.access_secret_version(request={"name": anthropic_secret})
            os.environ["ANTHROPIC_API_KEY"] = anthropic_response.payload.data.decode("UTF-8")
            
            # Semantic Scholar API Key
            s2_secret = f"projects/{self.project_id}/secrets/s2-api-key/versions/latest"
            s2_response = self.secret_client.access_secret_version(request={"name": s2_secret})
            os.environ["S2_API_KEY"] = s2_response.payload.data.decode("UTF-8")
            
            self.logger.info("Successfully loaded API keys from Secret Manager")
            
        except Exception as e:
            self.logger.warning(f"Could not load some API keys: {e}")
    
    def setup_repositories(self):
        """Clone and set up AI Scientist repositories"""
        try:
            # Create base directory
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Clone AI Scientist v1
            if not self.v1_path.exists():
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/SakanaAI/AI-Scientist.git",
                    str(self.v1_path)
                ], check=True)
                self.logger.info("Cloned AI Scientist v1")
            
            # Clone AI Scientist v2
            if not self.v2_path.exists():
                subprocess.run([
                    "git", "clone",
                    "https://github.com/SakanaAI/AI-Scientist-v2.git", 
                    str(self.v2_path)
                ], check=True)
                self.logger.info("Cloned AI Scientist v2")
            
            # Install dependencies
            self._install_dependencies()
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to set up repositories: {e}")
            raise
    
    def _install_dependencies(self):
        """Install required dependencies for AI Scientist"""
        try:
            # Install v1 dependencies
            v1_requirements = self.v1_path / "requirements.txt"
            if v1_requirements.exists():
                subprocess.run([
                    "pip", "install", "-r", str(v1_requirements)
                ], check=True)
            
            # Install v2 dependencies  
            v2_requirements = self.v2_path / "requirements.txt"
            if v2_requirements.exists():
                subprocess.run([
                    "pip", "install", "-r", str(v2_requirements)
                ], check=True)
                
            self.logger.info("Installed AI Scientist dependencies")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            raise
    
    def generate_research_paper(self, 
                              experiment_results: Dict[str, Any],
                              template_name: str = "vr_affordances",
                              num_ideas: int = 3) -> Dict[str, Any]:
        """
        Generate a research paper using AI Scientist
        
        Args:
            experiment_results: Results from AMIEN experiments
            template_name: Template to use for v1 (ignored for v2)
            num_ideas: Number of ideas to explore
            
        Returns:
            Dictionary containing paper generation results
        """
        try:
            if self.use_v2:
                return self._generate_paper_v2(experiment_results, num_ideas)
            else:
                return self._generate_paper_v1(experiment_results, template_name, num_ideas)
                
        except Exception as e:
            self.logger.error(f"Failed to generate research paper: {e}")
            raise
    
    def _generate_paper_v1(self, 
                          experiment_results: Dict[str, Any],
                          template_name: str,
                          num_ideas: int) -> Dict[str, Any]:
        """Generate paper using AI Scientist v1 (template-based)"""
        
        # Prepare template
        template = self.spatial_templates.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = self.v1_path / "experiments" / f"{template_name}_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Write experiment data
        experiment_file = experiment_dir / "experiment_data.json"
        with open(experiment_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        # Write template files
        self._write_template_files(experiment_dir, template)
        
        # Run AI Scientist v1
        cmd = [
            "python", str(self.v1_path / "launch_scientist.py"),
            "--model", "claude-3-5-sonnet-20241022",
            "--experiment", template_name,
            "--num-ideas", str(num_ideas),
            "--parallel"
        ]
        
        result = subprocess.run(cmd, 
                              cwd=str(self.v1_path),
                              capture_output=True, 
                              text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"AI Scientist v1 failed: {result.stderr}")
        
        # Process results
        return self._process_v1_results(experiment_dir)
    
    def _generate_paper_v2(self,
                          experiment_results: Dict[str, Any], 
                          num_ideas: int) -> Dict[str, Any]:
        """Generate paper using AI Scientist v2 (agentic tree search)"""
        
        # Create research topic file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_file = self.v2_path / "ai_scientist" / "ideas" / f"vr_spatial_research_{timestamp}.md"
        
        # Generate topic description from experiment results
        topic_content = self._create_topic_description(experiment_results)
        
        with open(topic_file, 'w') as f:
            f.write(topic_content)
        
        # Run ideation step
        ideation_cmd = [
            "python", str(self.v2_path / "ai_scientist" / "perform_ideation_temp_free.py"),
            "--workshop-file", str(topic_file),
            "--model", "gpt-4o-2024-05-13",
            "--max-num-generations", str(num_ideas * 5),
            "--num-reflections", "5"
        ]
        
        result = subprocess.run(ideation_cmd,
                              cwd=str(self.v2_path),
                              capture_output=True,
                              text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"AI Scientist v2 ideation failed: {result.stderr}")
        
        # Run main experiment pipeline
        ideas_file = topic_file.with_suffix('.json')
        
        main_cmd = [
            "python", str(self.v2_path / "launch_scientist_bfts.py"),
            "--load_ideas", str(ideas_file),
            "--load_code",
            "--add_dataset_ref",
            "--model_writeup", "o1-preview-2024-09-12",
            "--model_citation", "gpt-4o-2024-11-20", 
            "--model_review", "gpt-4o-2024-11-20",
            "--model_agg_plots", "o3-mini-2025-01-31",
            "--num_cite_rounds", "20"
        ]
        
        result = subprocess.run(main_cmd,
                              cwd=str(self.v2_path),
                              capture_output=True,
                              text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"AI Scientist v2 main pipeline failed: {result.stderr}")
        
        # Process results
        return self._process_v2_results(ideas_file)
    
    def _create_vr_affordance_template(self) -> str:
        """Create VR affordance discovery template"""
        return """
# VR Affordance Discovery Template

## Problem Description
Discover and optimize visual affordances in VR environments for enhanced spatial reasoning.

## Evaluation Function
def evaluate_affordance(affordance_config, user_data, environment_data):
    # Evaluate affordance effectiveness
    # Returns score based on user performance metrics
    pass

## Solve Function  
def solve_affordance_discovery(environment_params):
    # Main algorithm for discovering affordances
    # Uses priority function to rank potential affordances
    pass

## Priority Function (to be evolved)
def priority(vr_object, user_context, environment):
    # This function will be evolved by AI Scientist
    return 1.0  # placeholder
"""
    
    def _create_spatial_reasoning_template(self) -> str:
        """Create spatial reasoning algorithm template"""
        return """
# Spatial Reasoning Algorithm Template

## Problem Description
Develop algorithms for enhanced spatial reasoning in VR environments.

## Evaluation Function
def evaluate_spatial_algorithm(algorithm, test_cases):
    # Evaluate spatial reasoning performance
    # Returns accuracy and efficiency metrics
    pass

## Solve Function
def solve_spatial_reasoning(spatial_data):
    # Main spatial reasoning algorithm
    # Uses priority function for spatial decisions
    pass

## Priority Function (to be evolved)
def spatial_priority(spatial_element, context, constraints):
    # This function will be evolved by AI Scientist
    return 1.0  # placeholder
"""
    
    def _create_visual_cue_template(self) -> str:
        """Create visual cue evolution template"""
        return """
# Visual Cue Evolution Template

## Problem Description
Evolve effective visual cues for VR navigation and interaction.

## Evaluation Function
def evaluate_visual_cue(cue_config, user_responses):
    # Evaluate visual cue effectiveness
    # Returns attention capture and usability metrics
    pass

## Solve Function
def solve_visual_cue_evolution(cue_space):
    # Main visual cue evolution algorithm
    # Uses priority function to select optimal cues
    pass

## Priority Function (to be evolved)
def cue_priority(visual_cue, user_profile, context):
    # This function will be evolved by AI Scientist
    return 1.0  # placeholder
"""
    
    def _create_topic_description(self, experiment_results: Dict[str, Any]) -> str:
        """Create topic description for AI Scientist v2"""
        return f"""# VR Spatial Reasoning Research

## Title
Autonomous Discovery of Visual Affordances in Virtual Reality Environments

## Keywords
Virtual Reality, Spatial Reasoning, Visual Affordances, Human-Computer Interaction, Cognitive Science

## TL;DR
This research explores the autonomous discovery and optimization of visual affordances in VR environments to enhance spatial reasoning capabilities across diverse user populations.

## Abstract
Virtual Reality environments present unique opportunities for spatial reasoning enhancement through carefully designed visual affordances. This research investigates the autonomous discovery of optimal visual cue configurations that improve spatial navigation, object interaction, and environmental understanding in VR settings.

Current experiment data shows:
- Total experiments conducted: {experiment_results.get('total_experiments', 'N/A')}
- Average spatial accuracy: {experiment_results.get('avg_spatial_accuracy', 'N/A')}
- User satisfaction scores: {experiment_results.get('user_satisfaction', 'N/A')}

The goal is to develop novel algorithms that can automatically discover and optimize visual affordances for enhanced VR spatial reasoning across diverse user populations and environmental contexts.

## Research Questions
1. How can visual affordances be automatically discovered and optimized for VR spatial reasoning?
2. What patterns emerge across different user populations and environmental contexts?
3. How do cross-domain inspirations (e.g., firefly bioluminescence, casino psychology) inform VR affordance design?

## Methodology
- Autonomous experimentation using Padres API for spatial reasoning evaluation
- Multi-environment testing across diverse VR contexts
- Synthetic user population generation for comprehensive evaluation
- Cross-domain inspiration integration for novel affordance discovery
"""
    
    def _write_template_files(self, experiment_dir: Path, template: Dict[str, Any]):
        """Write template files for AI Scientist v1"""
        
        # Write main template file
        template_file = experiment_dir / "template.py"
        with open(template_file, 'w') as f:
            f.write(template["skeleton"])
        
        # Write evaluation configuration
        eval_config = {
            "metrics": template["evaluation_metrics"],
            "description": template["description"]
        }
        
        eval_file = experiment_dir / "evaluation_config.json"
        with open(eval_file, 'w') as f:
            json.dump(eval_config, f, indent=2)
    
    def _process_v1_results(self, experiment_dir: Path) -> Dict[str, Any]:
        """Process AI Scientist v1 results"""
        results = {
            "version": "v1",
            "experiment_dir": str(experiment_dir),
            "papers": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Look for generated papers
        papers_dir = experiment_dir / "papers"
        if papers_dir.exists():
            for paper_file in papers_dir.glob("*.pdf"):
                results["papers"].append({
                    "file": str(paper_file),
                    "title": paper_file.stem,
                    "size_bytes": paper_file.stat().st_size
                })
        
        # Upload to GCS
        self._upload_results_to_gcs(results)
        
        return results
    
    def _process_v2_results(self, ideas_file: Path) -> Dict[str, Any]:
        """Process AI Scientist v2 results"""
        results = {
            "version": "v2", 
            "ideas_file": str(ideas_file),
            "papers": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Look for experiment results
        experiments_dir = self.v2_path / "experiments"
        if experiments_dir.exists():
            # Find most recent experiment
            experiment_dirs = sorted(experiments_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
            
            if experiment_dirs:
                latest_dir = experiment_dirs[0]
                results["experiment_dir"] = str(latest_dir)
                
                # Look for generated papers
                for paper_file in latest_dir.rglob("*.pdf"):
                    results["papers"].append({
                        "file": str(paper_file),
                        "title": paper_file.stem,
                        "size_bytes": paper_file.stat().st_size
                    })
        
        # Upload to GCS
        self._upload_results_to_gcs(results)
        
        return results
    
    def _upload_results_to_gcs(self, results: Dict[str, Any]):
        """Upload results to Google Cloud Storage"""
        try:
            bucket = self.storage_client.bucket(self.papers_bucket)
            
            # Upload results metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_blob = bucket.blob(f"results/{timestamp}_metadata.json")
            metadata_blob.upload_from_string(json.dumps(results, indent=2))
            
            # Upload paper files
            for paper in results.get("papers", []):
                paper_path = Path(paper["file"])
                if paper_path.exists():
                    paper_blob = bucket.blob(f"papers/{timestamp}_{paper_path.name}")
                    paper_blob.upload_from_filename(str(paper_path))
            
            self.logger.info(f"Uploaded results to GCS bucket: {self.papers_bucket}")
            
        except Exception as e:
            self.logger.error(f"Failed to upload results to GCS: {e}")
    
    def list_generated_papers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recently generated papers"""
        try:
            bucket = self.storage_client.bucket(self.papers_bucket)
            blobs = bucket.list_blobs(prefix="papers/", max_results=limit)
            
            papers = []
            for blob in blobs:
                if blob.name.endswith('.pdf'):
                    papers.append({
                        "name": blob.name,
                        "size": blob.size,
                        "created": blob.time_created.isoformat(),
                        "download_url": f"gs://{self.papers_bucket}/{blob.name}"
                    })
            
            return sorted(papers, key=lambda x: x["created"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to list papers: {e}")
            return []
    
    def get_paper_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated papers"""
        try:
            bucket = self.storage_client.bucket(self.papers_bucket)
            
            # Count papers
            paper_blobs = list(bucket.list_blobs(prefix="papers/"))
            total_papers = len([b for b in paper_blobs if b.name.endswith('.pdf')])
            
            # Count results
            result_blobs = list(bucket.list_blobs(prefix="results/"))
            total_experiments = len([b for b in result_blobs if b.name.endswith('.json')])
            
            # Calculate total size
            total_size = sum(blob.size for blob in paper_blobs if blob.size)
            
            return {
                "total_papers": total_papers,
                "total_experiments": total_experiments,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {} 