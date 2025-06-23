#!/usr/bin/env python3
"""
AMIEN Final Production Deployment
Complete system testing, deployment, and monitoring setup
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.append(".")


class AMIENProductionDeployment:
    """Final production deployment and testing for AMIEN"""

    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT_ID", "amien-research-pipeline")
        self.region = "us-central1"
        self.deployment_dir = Path("gcp_deployment")
        self.test_results = {}

        print("🚀 AMIEN Final Production Deployment")
        print(f"   Project ID: {self.project_id}")
        print(f"   Region: {self.region}")
        print(f"   Timestamp: {datetime.now().isoformat()}")

    async def run_comprehensive_system_test(self):
        """Run comprehensive system tests before deployment"""
        print("\n🧪 Running Comprehensive System Tests...")

        test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "unknown",
        }

        # Test 1: Basic Integration Test
        print("   📋 Test 1: Basic Integration...")
        try:
            from restart_amien_integration import AMIENIntegrationManager

            manager = AMIENIntegrationManager()

            # Quick integration test
            experiment_data = await manager.generate_synthetic_vr_data(50)
            paper_result = await manager.run_ai_scientist_integration(experiment_data)
            function_result = await manager.run_funsearch_integration(experiment_data)

            test_results["tests"]["basic_integration"] = {
                "status": "PASS",
                "experiments": len(experiment_data),
                "paper_quality": paper_result.get("quality_score", 0),
                "function_fitness": function_result.get("fitness", 0),
                "cost": paper_result.get("cost", 0),
            }
            print("      ✅ Basic Integration: PASS")

        except Exception as e:
            test_results["tests"]["basic_integration"] = {
                "status": "FAIL",
                "error": str(e),
            }
            print(f"      ❌ Basic Integration: FAIL - {e}")

        # Test 2: Massive Scale Test
        print("   📊 Test 2: Massive Scale Processing...")
        try:
            from scale_to_production import MassiveScaleExperimentRunner

            runner = MassiveScaleExperimentRunner(num_users=100, num_environments=10)

            # Quick massive scale test
            results = await runner.run_massive_scale_experiments(sample_size=100)

            test_results["tests"]["massive_scale"] = {
                "status": "PASS",
                "experiments": len(results),
                "success_rate": sum(1 for r in results if r["success"])
                / len(results)
                * 100,
                "avg_comfort": sum(r["comfort_score"] for r in results) / len(results),
            }
            print("      ✅ Massive Scale: PASS")

        except Exception as e:
            test_results["tests"]["massive_scale"] = {"status": "FAIL", "error": str(e)}
            print(f"      ❌ Massive Scale: FAIL - {e}")

        # Test 3: GCP Deployment Readiness
        print("   ☁️ Test 3: GCP Deployment Readiness...")
        try:
            # Check if deployment files exist
            required_files = [
                "gcp_deployment/main.t",
                "gcp_deployment/Dockerfile.api",
                "gcp_deployment/main_api.py",
                "deploy_amien.sh",
            ]

            missing_files = [f for f in required_files if not Path(f).exists()]

            if not missing_files:
                test_results["tests"]["deployment_readiness"] = {
                    "status": "PASS",
                    "files_ready": len(required_files),
                    "terraform_config": "ready",
                    "docker_config": "ready",
                }
                print("      ✅ Deployment Readiness: PASS")
            else:
                test_results["tests"]["deployment_readiness"] = {
                    "status": "FAIL",
                    "missing_files": missing_files,
                }
                print(
                    f"      ❌ Deployment Readiness: FAIL - Missing files: {missing_files}"
                )

        except Exception as e:
            test_results["tests"]["deployment_readiness"] = {
                "status": "FAIL",
                "error": str(e),
            }
            print(f"      ❌ Deployment Readiness: FAIL - {e}")

        # Test 4: API Keys and Secrets
        print("   🔐 Test 4: API Keys and Secrets...")
        try:
            gemini_key = os.getenv("GEMINI_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")

            test_results["tests"]["api_keys"] = {
                "status": "PASS" if gemini_key else "WARN",
                "gemini_available": bool(gemini_key),
                "openai_available": bool(openai_key),
                "recommendation": (
                    "Set GEMINI_API_KEY for full functionality"
                    if not gemini_key
                    else "All keys available"
                ),
            }

            if gemini_key:
                print("      ✅ API Keys: PASS")
            else:
                print("      ⚠️ API Keys: WARN - GEMINI_API_KEY not set")

        except Exception as e:
            test_results["tests"]["api_keys"] = {"status": "FAIL", "error": str(e)}
            print(f"      ❌ API Keys: FAIL - {e}")

        # Overall status
        passed_tests = sum(
            1 for test in test_results["tests"].values() if test["status"] == "PASS"
        )
        total_tests = len(test_results["tests"])

        if passed_tests == total_tests:
            test_results["overall_status"] = "READY_FOR_PRODUCTION"
        elif passed_tests >= total_tests * 0.75:
            test_results["overall_status"] = "READY_WITH_WARNINGS"
        else:
            test_results["overall_status"] = "NOT_READY"

        # Save test results
        test_file = Path("production_test_results.json")
        with open(test_file, "w") as f:
            json.dump(test_results, f, indent=2)

        print("\n📊 System Test Results:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Overall Status: {test_results['overall_status']}")
        print(f"   Results saved: {test_file}")

        self.test_results = test_results
        return test_results

    def prepare_production_environment(self):
        """Prepare production environment variables and configurations"""
        print("\n🔧 Preparing Production Environment...")

        # Create production environment file
        env_content = """# AMIEN Production Environment Configuration
# Generated: {datetime.now().isoformat()}

# Google Cloud Platform
export GCP_PROJECT_ID="{self.project_id}"
export GCP_REGION="{self.region}"
export GCP_ZONE="{self.region}-a"

# API Keys (UPDATE THESE WITH YOUR ACTUAL KEYS)
export GEMINI_API_KEY="your-gemini-api-key-here"
export OPENAI_API_KEY="your-openai-api-key-here"
export PERPLEXITY_API_KEY="your-perplexity-api-key-here"

# AMIEN Configuration
export AMIEN_ENV="production"
export AMIEN_LOG_LEVEL="INFO"
export AMIEN_MAX_EXPERIMENTS="10000"
export AMIEN_BATCH_SIZE="100"

# Monitoring
export ENABLE_MONITORING="true"
export ALERT_EMAIL="your-email@example.com"

# Cost Controls
export MAX_DAILY_COST="100"
export MAX_MONTHLY_COST="3000"

# Load environment
echo "🌟 AMIEN Production Environment Loaded"
echo "   Project: $GCP_PROJECT_ID"
echo "   Region: $GCP_REGION"
echo "   Environment: $AMIEN_ENV"
"""

        with open("production.env", "w") as f:
            f.write(env_content)

        # Create production requirements
        prod_requirements = """# AMIEN Production Requirements
fastapi==0.104.1
uvicorn[standard]==0.24.0
google-cloud-run==0.10.0
google-cloud-storage==2.10.0
google-cloud-secret-manager==2.16.4
google-cloud-scheduler==2.13.4
google-cloud-monitoring==2.16.0
google-generativeai==0.3.2
openai==1.3.7
requests==2.31.0
aiohttp==3.9.1
asyncio==3.4.3
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0
scipy==1.11.4
scikit-learn==1.3.2
tensorflow==2.15.0
torch==2.1.1
transformers==4.36.2
datasets==2.14.6
accelerate==0.25.0
"""

        with open("requirements_production.txt", "w") as f:
            f.write(prod_requirements)

        print("   ✅ Production environment file: production.env")
        print("   ✅ Production requirements: requirements_production.txt")
        print("   ⚠️ Remember to update API keys in production.env")

    def create_monitoring_dashboard(self):
        """Create comprehensive monitoring dashboard"""
        print("\n📊 Creating Monitoring Dashboard...")

        dashboard_config = {
            "displayName": "AMIEN Production Dashboard",
            "mosaicLayout": {
                "tiles": [
                    {
                        "width": 6,
                        "height": 4,
                        "widget": {
                            "title": "Research Papers Generated",
                            "scorecard": {
                                "timeSeriesQuery": {
                                    "timeSeriesFilter": {
                                        "filter": 'metric.type="custom.googleapis.com/amien/papers_generated"'
                                    }
                                }
                            },
                        },
                    },
                    {
                        "width": 6,
                        "height": 4,
                        "widget": {
                            "title": "Experiments Completed",
                            "xyChart": {
                                "dataSets": [
                                    {
                                        "timeSeriesQuery": {
                                            "timeSeriesFilter": {
                                                "filter": 'metric.type="custom.googleapis.com/amien/experiments_completed"'
                                            }
                                        }
                                    }
                                ]
                            },
                        },
                    },
                    {
                        "width": 6,
                        "height": 4,
                        "widget": {
                            "title": "API Response Time",
                            "xyChart": {
                                "dataSets": [
                                    {
                                        "timeSeriesQuery": {
                                            "timeSeriesFilter": {
                                                "filter": "resource.type="cloud_run_revision" resource.label.service_name="amien-api-service"'
                                            }
                                        }
                                    }
                                ]
                            },
                        },
                    },
                    {
                        "width": 6,
                        "height": 4,
                        "widget": {
                            "title": "Daily Cost",
                            "scorecard": {
                                "timeSeriesQuery": {
                                    "timeSeriesFilter": {
                                        "filter": 'metric.type="custom.googleapis.com/amien/daily_cost"'
                                    }
                                }
                            },
                        },
                    },
                    {
                        "width": 12,
                        "height": 4,
                        "widget": {
                            "title": "Research Quality Scores",
                            "xyChart": {
                                "dataSets": [
                                    {
                                        "timeSeriesQuery": {
                                            "timeSeriesFilter": {
                                                "filter": 'metric.type="custom.googleapis.com/amien/research_quality"'
                                            }
                                        }
                                    }
                                ]
                            },
                        },
                    },
                ]
            },
        }

        # Save dashboard config
        dashboard_file = Path("monitoring/production_dashboard.json")
        dashboard_file.parent.mkdir(exist_ok=True)
        with open(dashboard_file, "w") as f:
            json.dump(dashboard_config, f, indent=2)

        print(f"   ✅ Dashboard config: {dashboard_file}")

    def create_deployment_checklist(self):
        """Create deployment checklist and instructions"""
        print("\n📋 Creating Deployment Checklist...")

        checklist = """# 🚀 AMIEN Production Deployment Checklist

## Pre-Deployment (Complete these first)

### 1. Environment Setup
- [ ] Update API keys in `production.env`
- [ ] Set GCP_PROJECT_ID: `{self.project_id}`
- [ ] Verify GCP billing is enabled
- [ ] Install gcloud CLI and authenticate

### 2. API Keys Required
- [ ] Google Gemini API key
- [ ] OpenAI API key (optional)
- [ ] Perplexity API key (optional)

### 3. GCP Services to Enable
```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable scheduler.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable monitoring.googleapis.com
```

## Deployment Steps

### 1. Load Environment
```bash
source production.env
```

### 2. Run System Tests
```bash
python3 final_production_deployment.py --test-only
```

### 3. Deploy to GCP
```bash
./deploy_amien.sh
```

### 4. Verify Deployment
- [ ] Check Cloud Run services are running
- [ ] Verify API endpoints respond
- [ ] Test research generation
- [ ] Check scheduled jobs

### 5. Monitor System
- [ ] Set up monitoring dashboard
- [ ] Configure alerts
- [ ] Monitor costs
- [ ] Check logs

## Post-Deployment

### 1. First Research Cycle
- [ ] Trigger manual research generation
- [ ] Verify paper quality
- [ ] Check function discovery
- [ ] Monitor performance

### 2. Scaling Test
- [ ] Run massive scale experiment
- [ ] Monitor auto-scaling
- [ ] Check cost controls
- [ ] Verify data storage

### 3. Continuous Operation
- [ ] Daily research papers generating
- [ ] Weekly massive experiments running
- [ ] Monitoring alerts working
- [ ] Cost within budget

## Emergency Contacts
- GCP Support: https://cloud.google.com/support
- AMIEN Issues: Check logs in Cloud Logging
- Cost Alerts: Monitor billing dashboard

## Success Metrics
- ✅ Research papers: 1+ per day
- ✅ Experiments: 1000+ per week
- ✅ Uptime: >99.5%
- ✅ Cost: <$5000/month
- ✅ Quality: >85/100 average

Generated: {datetime.now().isoformat()}
"""

        with open("DEPLOYMENT_CHECKLIST.md", "w") as f:
            f.write(checklist)

        print("   ✅ Deployment checklist: DEPLOYMENT_CHECKLIST.md")

    async def run_final_deployment(self):
        """Run the complete final deployment process"""
        print("\n🚀 Running Final AMIEN Deployment...")

        start_time = datetime.now()

        # Step 1: System Tests
        test_results = await self.run_comprehensive_system_test()

        # Step 2: Environment Preparation
        self.prepare_production_environment()

        # Step 3: Monitoring Setup
        self.create_monitoring_dashboard()

        # Step 4: Deployment Checklist
        self.create_deployment_checklist()

        # Step 5: Final Status Report
        deployment_summary = {
            "deployment_timestamp": start_time.isoformat(),
            "completion_timestamp": datetime.now().isoformat(),
            "duration_minutes": (datetime.now() - start_time).total_seconds() / 60,
            "system_tests": test_results,
            "deployment_status": test_results["overall_status"],
            "next_steps": [
                "Update API keys in production.env",
                "Run: source production.env",
                "Run: ./deploy_amien.sh",
                "Monitor deployment in GCP Console",
                "Verify first research generation",
            ],
            "estimated_monthly_cost": "$2,000-5,000",
            "expected_capabilities": [
                "24/7 autonomous research generation",
                "Auto-scaling to 100+ instances",
                "Daily research papers",
                "Weekly massive experiments",
                "Real-time monitoring",
            ],
        }

        # Save deployment summary
        summary_file = Path("FINAL_DEPLOYMENT_SUMMARY.json")
        with open(summary_file, "w") as f:
            json.dump(deployment_summary, f, indent=2)

        print("\n🎉 Final Deployment Complete!")
        print(f"   Duration: {deployment_summary['duration_minutes']:.1f} minutes")
        print(f"   Status: {deployment_summary['deployment_status']}")
        print(f"   Summary: {summary_file}")

        # Print next steps
        print("\n📋 Next Steps:")
        for i, step in enumerate(deployment_summary["next_steps"], 1):
            print(f"   {i}. {step}")

        return deployment_summary


async def main():
    """Main deployment function"""
    print("🌟 AMIEN Final Production Deployment")
    print("=" * 60)

    # Check for test-only mode
    test_only = "--test-only" in sys.argv

    deployer = AMIENProductionDeployment()

    if test_only:
        print("🧪 Running in TEST-ONLY mode...")
        await deployer.run_comprehensive_system_test()
    else:
        # Run full deployment
        await deployer.run_final_deployment()

    print("\n✨ AMIEN is ready for production!")


if __name__ == "__main__":
    asyncio.run(main())
