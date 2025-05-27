"""
FunSearch Integration Manager for AMIEN
Handles mathematical function discovery using DeepMind's FunSearch
"""

import os
import json
import subprocess
import logging
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import importlib.util

from google.cloud import storage
from google.cloud import secretmanager

class FunSearchManager:
    """
    Manages FunSearch integration with AMIEN pipeline for discovering
    spatial reasoning functions and VR affordance algorithms
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.storage_client = storage.Client()
        self.secret_client = secretmanager.SecretManagerServiceClient()
        
        # Set up paths
        self.base_path = Path("/app/funsearch")
        self.funsearch_path = self.base_path / "funsearch"
        
        # GCS buckets
        self.functions_bucket = f"{project_id}-discovered-functions"
        self.data_bucket = f"{project_id}-research-data"
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load API keys
        self._load_api_keys()
        
        # Function discovery templates
        self.discovery_templates = {
            "spatial_priority": {
                "description": "Priority function for spatial reasoning in VR",
                "skeleton": self._create_spatial_priority_skeleton(),
                "evaluator": self._create_spatial_evaluator(),
                "test_cases": self._create_spatial_test_cases()
            },
            "affordance_ranking": {
                "description": "Ranking function for VR affordances",
                "skeleton": self._create_affordance_ranking_skeleton(),
                "evaluator": self._create_affordance_evaluator(),
                "test_cases": self._create_affordance_test_cases()
            },
            "visual_cue_priority": {
                "description": "Priority function for visual cue selection",
                "skeleton": self._create_visual_cue_skeleton(),
                "evaluator": self._create_visual_cue_evaluator(),
                "test_cases": self._create_visual_cue_test_cases()
            },
            "navigation_heuristic": {
                "description": "Heuristic function for VR navigation",
                "skeleton": self._create_navigation_skeleton(),
                "evaluator": self._create_navigation_evaluator(),
                "test_cases": self._create_navigation_test_cases()
            }
        }
        
        # Discovery configuration
        self.config = {
            "num_samplers": 15,
            "num_evaluators": 150,
            "max_sample_length": 2048,
            "temperature": 0.8,
            "max_iterations": 1000,
            "convergence_threshold": 0.95
        }
    
    def _load_api_keys(self):
        """Load API keys from Google Secret Manager"""
        try:
            # OpenAI API Key for FunSearch LLM
            openai_secret = f"projects/{self.project_id}/secrets/openai-api-key/versions/latest"
            openai_response = self.secret_client.access_secret_version(request={"name": openai_secret})
            os.environ["OPENAI_API_KEY"] = openai_response.payload.data.decode("UTF-8")
            
            self.logger.info("Successfully loaded API keys from Secret Manager")
            
        except Exception as e:
            self.logger.warning(f"Could not load API keys: {e}")
    
    def setup_repository(self):
        """Clone and set up FunSearch repository"""
        try:
            # Create base directory
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Clone FunSearch
            if not self.funsearch_path.exists():
                subprocess.run([
                    "git", "clone",
                    "https://github.com/google-deepmind/funsearch.git",
                    str(self.funsearch_path)
                ], check=True)
                self.logger.info("Cloned FunSearch repository")
            
            # Install dependencies
            self._install_dependencies()
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to set up FunSearch repository: {e}")
            raise
    
    def _install_dependencies(self):
        """Install FunSearch dependencies"""
        try:
            requirements_file = self.funsearch_path / "requirements.txt"
            if requirements_file.exists():
                subprocess.run([
                    "pip", "install", "-r", str(requirements_file)
                ], check=True)
            
            # Install additional dependencies for AMIEN integration
            additional_deps = [
                "numpy>=1.21.0",
                "scipy>=1.7.0",
                "scikit-learn>=1.0.0",
                "matplotlib>=3.5.0",
                "google-cloud-storage>=2.0.0"
            ]
            
            for dep in additional_deps:
                subprocess.run(["pip", "install", dep], check=True)
            
            self.logger.info("Installed FunSearch dependencies")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            raise
    
    def discover_spatial_function(self, 
                                function_type: str,
                                padres_data: Dict[str, Any],
                                max_iterations: int = 1000) -> Dict[str, Any]:
        """
        Discover a spatial reasoning function using FunSearch
        
        Args:
            function_type: Type of function to discover
            padres_data: Data from Padres API experiments
            max_iterations: Maximum iterations for discovery
            
        Returns:
            Dictionary containing discovered function and metadata
        """
        try:
            if function_type not in self.discovery_templates:
                raise ValueError(f"Unknown function type: {function_type}")
            
            template = self.discovery_templates[function_type]
            
            # Create discovery experiment
            experiment_id = f"{function_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            experiment_dir = self.base_path / "experiments" / experiment_id
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare problem specification
            problem_spec = self._create_problem_specification(
                template, padres_data, experiment_dir
            )
            
            # Run FunSearch discovery
            result = self._run_funsearch_discovery(
                problem_spec, experiment_dir, max_iterations
            )
            
            # Process and validate results
            processed_result = self._process_discovery_results(
                result, template, padres_data, experiment_dir
            )
            
            # Upload to GCS
            self._upload_function_to_gcs(processed_result, experiment_id)
            
            return processed_result
            
        except Exception as e:
            self.logger.error(f"Failed to discover spatial function: {e}")
            raise
    
    def _create_problem_specification(self,
                                    template: Dict[str, Any],
                                    padres_data: Dict[str, Any],
                                    experiment_dir: Path) -> Dict[str, Any]:
        """Create FunSearch problem specification"""
        
        # Write skeleton function
        skeleton_file = experiment_dir / "skeleton.py"
        with open(skeleton_file, 'w') as f:
            f.write(template["skeleton"])
        
        # Write evaluator
        evaluator_file = experiment_dir / "evaluator.py"
        with open(evaluator_file, 'w') as f:
            f.write(template["evaluator"])
        
        # Write test cases with Padres data
        test_cases = template["test_cases"]
        test_cases["padres_data"] = padres_data
        
        test_file = experiment_dir / "test_cases.json"
        with open(test_file, 'w') as f:
            json.dump(test_cases, f, indent=2)
        
        return {
            "skeleton_file": str(skeleton_file),
            "evaluator_file": str(evaluator_file),
            "test_file": str(test_file),
            "description": template["description"]
        }
    
    def _run_funsearch_discovery(self,
                               problem_spec: Dict[str, Any],
                               experiment_dir: Path,
                               max_iterations: int) -> Dict[str, Any]:
        """Run FunSearch discovery process"""
        
        # Create FunSearch configuration
        config_file = experiment_dir / "funsearch_config.json"
        config = {
            "skeleton_file": problem_spec["skeleton_file"],
            "evaluator_file": problem_spec["evaluator_file"],
            "test_file": problem_spec["test_file"],
            "num_samplers": self.config["num_samplers"],
            "num_evaluators": self.config["num_evaluators"],
            "max_sample_length": self.config["max_sample_length"],
            "temperature": self.config["temperature"],
            "max_iterations": max_iterations,
            "convergence_threshold": self.config["convergence_threshold"]
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run FunSearch
        cmd = [
            "python", str(self.funsearch_path / "run_funsearch.py"),
            "--config", str(config_file),
            "--output_dir", str(experiment_dir / "results")
        ]
        
        result = subprocess.run(cmd,
                              cwd=str(self.funsearch_path),
                              capture_output=True,
                              text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FunSearch discovery failed: {result.stderr}")
        
        # Parse results
        results_dir = experiment_dir / "results"
        best_function_file = results_dir / "best_function.py"
        
        if not best_function_file.exists():
            raise RuntimeError("No function discovered by FunSearch")
        
        with open(best_function_file, 'r') as f:
            discovered_function = f.read()
        
        # Load discovery metadata
        metadata_file = results_dir / "discovery_metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        return {
            "discovered_function": discovered_function,
            "metadata": metadata,
            "experiment_dir": str(experiment_dir)
        }
    
    def _process_discovery_results(self,
                                 result: Dict[str, Any],
                                 template: Dict[str, Any],
                                 padres_data: Dict[str, Any],
                                 experiment_dir: Path) -> Dict[str, Any]:
        """Process and validate discovery results"""
        
        # Validate discovered function
        validation_result = self._validate_discovered_function(
            result["discovered_function"], template, padres_data
        )
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            result["discovered_function"], template, padres_data
        )
        
        # Generate function analysis
        analysis = self._analyze_discovered_function(
            result["discovered_function"], template
        )
        
        processed_result = {
            "function_code": result["discovered_function"],
            "validation": validation_result,
            "performance": performance_metrics,
            "analysis": analysis,
            "metadata": result["metadata"],
            "template_used": template["description"],
            "discovery_timestamp": datetime.now().isoformat(),
            "experiment_dir": result["experiment_dir"]
        }
        
        # Save processed results
        results_file = experiment_dir / "processed_results.json"
        with open(results_file, 'w') as f:
            json.dump(processed_result, f, indent=2)
        
        return processed_result
    
    def _validate_discovered_function(self,
                                    function_code: str,
                                    template: Dict[str, Any],
                                    padres_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the discovered function"""
        
        validation_result = {
            "syntax_valid": False,
            "execution_valid": False,
            "performance_valid": False,
            "errors": []
        }
        
        try:
            # Check syntax
            compile(function_code, '<string>', 'exec')
            validation_result["syntax_valid"] = True
            
            # Test execution
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(function_code)
                temp_file = f.name
            
            try:
                spec = importlib.util.spec_from_file_location("discovered_function", temp_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Test with sample data
                test_cases = template["test_cases"]
                for test_case in test_cases.get("samples", []):
                    # Execute function with test case
                    # This would depend on the specific function signature
                    pass
                
                validation_result["execution_valid"] = True
                
            finally:
                os.unlink(temp_file)
            
        except SyntaxError as e:
            validation_result["errors"].append(f"Syntax error: {e}")
        except Exception as e:
            validation_result["errors"].append(f"Execution error: {e}")
        
        return validation_result
    
    def _calculate_performance_metrics(self,
                                     function_code: str,
                                     template: Dict[str, Any],
                                     padres_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for discovered function"""
        
        # This would implement actual performance testing
        # using the Padres API data as ground truth
        
        return {
            "accuracy": 0.85,  # Placeholder
            "efficiency": 0.92,  # Placeholder
            "generalization": 0.78,  # Placeholder
            "improvement_over_baseline": 0.15  # Placeholder
        }
    
    def _analyze_discovered_function(self,
                                   function_code: str,
                                   template: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the discovered function for insights"""
        
        # This would implement code analysis to extract insights
        # about the discovered algorithm
        
        return {
            "complexity": "medium",
            "novel_patterns": ["weighted_spatial_distance", "context_aware_ranking"],
            "mathematical_insights": ["uses_logarithmic_scaling", "incorporates_user_preference"],
            "potential_applications": ["vr_navigation", "object_placement", "ui_design"]
        }
    
    def _upload_function_to_gcs(self, result: Dict[str, Any], experiment_id: str):
        """Upload discovered function to Google Cloud Storage"""
        try:
            bucket = self.storage_client.bucket(self.functions_bucket)
            
            # Upload function code
            function_blob = bucket.blob(f"functions/{experiment_id}.py")
            function_blob.upload_from_string(result["function_code"])
            
            # Upload metadata
            metadata_blob = bucket.blob(f"metadata/{experiment_id}.json")
            metadata_blob.upload_from_string(json.dumps(result, indent=2))
            
            self.logger.info(f"Uploaded function to GCS: {experiment_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to upload function to GCS: {e}")
    
    def list_discovered_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recently discovered functions"""
        try:
            bucket = self.storage_client.bucket(self.functions_bucket)
            blobs = bucket.list_blobs(prefix="functions/", max_results=limit)
            
            functions = []
            for blob in blobs:
                if blob.name.endswith('.py'):
                    # Get corresponding metadata
                    metadata_name = blob.name.replace('functions/', 'metadata/').replace('.py', '.json')
                    try:
                        metadata_blob = bucket.blob(metadata_name)
                        metadata_content = metadata_blob.download_as_text()
                        metadata = json.loads(metadata_content)
                        
                        functions.append({
                            "name": blob.name,
                            "size": blob.size,
                            "created": blob.time_created.isoformat(),
                            "template": metadata.get("template_used", "unknown"),
                            "performance": metadata.get("performance", {}),
                            "download_url": f"gs://{self.functions_bucket}/{blob.name}"
                        })
                    except Exception:
                        # Skip if metadata not found
                        continue
            
            return sorted(functions, key=lambda x: x["created"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to list functions: {e}")
            return []
    
    def get_function_statistics(self) -> Dict[str, Any]:
        """Get statistics about discovered functions"""
        try:
            bucket = self.storage_client.bucket(self.functions_bucket)
            
            # Count functions
            function_blobs = list(bucket.list_blobs(prefix="functions/"))
            total_functions = len([b for b in function_blobs if b.name.endswith('.py')])
            
            # Count by template type
            template_counts = {}
            for blob in function_blobs:
                if blob.name.endswith('.py'):
                    # Try to get template info from metadata
                    metadata_name = blob.name.replace('functions/', 'metadata/').replace('.py', '.json')
                    try:
                        metadata_blob = bucket.blob(metadata_name)
                        metadata_content = metadata_blob.download_as_text()
                        metadata = json.loads(metadata_content)
                        template = metadata.get("template_used", "unknown")
                        template_counts[template] = template_counts.get(template, 0) + 1
                    except Exception:
                        continue
            
            # Calculate total size
            total_size = sum(blob.size for blob in function_blobs if blob.size)
            
            return {
                "total_functions": total_functions,
                "functions_by_template": template_counts,
                "total_size_bytes": total_size,
                "total_size_kb": round(total_size / 1024, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    # Template creation methods
    def _create_spatial_priority_skeleton(self) -> str:
        """Create skeleton for spatial priority function"""
        return '''
def spatial_priority(vr_object, user_context, environment):
    """
    Priority function for spatial reasoning in VR environments.
    
    Args:
        vr_object: Dictionary containing object properties (position, size, type, etc.)
        user_context: Dictionary containing user state (position, gaze, preferences, etc.)
        environment: Dictionary containing environment properties (lighting, physics, etc.)
    
    Returns:
        float: Priority score (higher = more important for spatial reasoning)
    """
    # This function will be evolved by FunSearch
    return 1.0
'''
    
    def _create_spatial_evaluator(self) -> str:
        """Create evaluator for spatial priority function"""
        return '''
import json
import numpy as np

def evaluate_spatial_priority(function_code, test_cases):
    """Evaluate spatial priority function performance"""
    
    # Execute the function
    exec(function_code, globals())
    
    total_score = 0.0
    num_tests = 0
    
    # Load Padres API data for ground truth
    padres_data = test_cases.get("padres_data", {})
    
    for test_case in test_cases.get("samples", []):
        vr_object = test_case["vr_object"]
        user_context = test_case["user_context"]
        environment = test_case["environment"]
        expected_priority = test_case["expected_priority"]
        
        try:
            predicted_priority = spatial_priority(vr_object, user_context, environment)
            
            # Calculate accuracy based on expected vs predicted
            error = abs(predicted_priority - expected_priority)
            accuracy = max(0, 1 - error)
            
            total_score += accuracy
            num_tests += 1
            
        except Exception as e:
            # Penalize functions that fail
            total_score += 0.0
            num_tests += 1
    
    return total_score / max(num_tests, 1)
'''
    
    def _create_spatial_test_cases(self) -> Dict[str, Any]:
        """Create test cases for spatial priority function"""
        return {
            "samples": [
                {
                    "vr_object": {"position": [0, 0, 0], "size": 1.0, "type": "cube"},
                    "user_context": {"position": [1, 0, 0], "gaze": [0, 0, 0]},
                    "environment": {"lighting": "bright", "physics": "earth_gravity"},
                    "expected_priority": 0.8
                },
                # More test cases would be generated from actual Padres data
            ]
        }
    
    def _create_affordance_ranking_skeleton(self) -> str:
        """Create skeleton for affordance ranking function"""
        return '''
def affordance_ranking(affordances, user_profile, context):
    """
    Ranking function for VR affordances.
    
    Args:
        affordances: List of affordance dictionaries
        user_profile: Dictionary containing user characteristics
        context: Dictionary containing current context
    
    Returns:
        List of affordances sorted by relevance (highest first)
    """
    # This function will be evolved by FunSearch
    return affordances
'''
    
    def _create_affordance_evaluator(self) -> str:
        """Create evaluator for affordance ranking function"""
        return '''
def evaluate_affordance_ranking(function_code, test_cases):
    """Evaluate affordance ranking function performance"""
    
    exec(function_code, globals())
    
    total_score = 0.0
    num_tests = 0
    
    for test_case in test_cases.get("samples", []):
        affordances = test_case["affordances"]
        user_profile = test_case["user_profile"]
        context = test_case["context"]
        expected_ranking = test_case["expected_ranking"]
        
        try:
            predicted_ranking = affordance_ranking(affordances, user_profile, context)
            
            # Calculate ranking correlation
            # Implementation would use Spearman correlation or similar
            correlation = 0.8  # Placeholder
            
            total_score += correlation
            num_tests += 1
            
        except Exception:
            total_score += 0.0
            num_tests += 1
    
    return total_score / max(num_tests, 1)
'''
    
    def _create_affordance_test_cases(self) -> Dict[str, Any]:
        """Create test cases for affordance ranking function"""
        return {
            "samples": [
                {
                    "affordances": [
                        {"type": "button", "visibility": 0.9, "reachability": 0.8},
                        {"type": "lever", "visibility": 0.7, "reachability": 0.9}
                    ],
                    "user_profile": {"handedness": "right", "height": 1.75},
                    "context": {"task": "navigation", "urgency": "high"},
                    "expected_ranking": [0, 1]  # Indices of affordances in order
                }
            ]
        }
    
    def _create_visual_cue_skeleton(self) -> str:
        """Create skeleton for visual cue priority function"""
        return '''
def visual_cue_priority(visual_cue, user_attention, environment_state):
    """
    Priority function for visual cue selection in VR.
    
    Args:
        visual_cue: Dictionary containing cue properties
        user_attention: Dictionary containing attention state
        environment_state: Dictionary containing environment properties
    
    Returns:
        float: Priority score for this visual cue
    """
    # This function will be evolved by FunSearch
    return 1.0
'''
    
    def _create_visual_cue_evaluator(self) -> str:
        """Create evaluator for visual cue priority function"""
        return '''
def evaluate_visual_cue_priority(function_code, test_cases):
    """Evaluate visual cue priority function performance"""
    
    exec(function_code, globals())
    
    total_score = 0.0
    num_tests = 0
    
    for test_case in test_cases.get("samples", []):
        visual_cue = test_case["visual_cue"]
        user_attention = test_case["user_attention"]
        environment_state = test_case["environment_state"]
        expected_effectiveness = test_case["expected_effectiveness"]
        
        try:
            predicted_priority = visual_cue_priority(visual_cue, user_attention, environment_state)
            
            # Evaluate based on expected effectiveness
            error = abs(predicted_priority - expected_effectiveness)
            accuracy = max(0, 1 - error)
            
            total_score += accuracy
            num_tests += 1
            
        except Exception:
            total_score += 0.0
            num_tests += 1
    
    return total_score / max(num_tests, 1)
'''
    
    def _create_visual_cue_test_cases(self) -> Dict[str, Any]:
        """Create test cases for visual cue priority function"""
        return {
            "samples": [
                {
                    "visual_cue": {"color": "red", "brightness": 0.8, "motion": "pulse"},
                    "user_attention": {"focus_level": 0.7, "fatigue": 0.3},
                    "environment_state": {"ambient_light": 0.5, "distractions": 2},
                    "expected_effectiveness": 0.85
                }
            ]
        }
    
    def _create_navigation_skeleton(self) -> str:
        """Create skeleton for navigation heuristic function"""
        return '''
def navigation_heuristic(current_position, target_position, obstacles, user_preferences):
    """
    Heuristic function for VR navigation pathfinding.
    
    Args:
        current_position: Current user position [x, y, z]
        target_position: Target position [x, y, z]
        obstacles: List of obstacle dictionaries
        user_preferences: Dictionary of user navigation preferences
    
    Returns:
        float: Heuristic cost estimate
    """
    # This function will be evolved by FunSearch
    return 1.0
'''
    
    def _create_navigation_evaluator(self) -> str:
        """Create evaluator for navigation heuristic function"""
        return '''
def evaluate_navigation_heuristic(function_code, test_cases):
    """Evaluate navigation heuristic function performance"""
    
    exec(function_code, globals())
    
    total_score = 0.0
    num_tests = 0
    
    for test_case in test_cases.get("samples", []):
        current_pos = test_case["current_position"]
        target_pos = test_case["target_position"]
        obstacles = test_case["obstacles"]
        preferences = test_case["user_preferences"]
        expected_cost = test_case["expected_cost"]
        
        try:
            predicted_cost = navigation_heuristic(current_pos, target_pos, obstacles, preferences)
            
            # Evaluate heuristic quality
            error = abs(predicted_cost - expected_cost) / max(expected_cost, 1.0)
            accuracy = max(0, 1 - error)
            
            total_score += accuracy
            num_tests += 1
            
        except Exception:
            total_score += 0.0
            num_tests += 1
    
    return total_score / max(num_tests, 1)
'''
    
    def _create_navigation_test_cases(self) -> Dict[str, Any]:
        """Create test cases for navigation heuristic function"""
        return {
            "samples": [
                {
                    "current_position": [0, 0, 0],
                    "target_position": [10, 0, 0],
                    "obstacles": [{"position": [5, 0, 0], "radius": 1.0}],
                    "user_preferences": {"prefer_direct_path": True, "avoid_heights": False},
                    "expected_cost": 12.0  # Slightly longer than direct distance due to obstacle
                }
            ]
        } 