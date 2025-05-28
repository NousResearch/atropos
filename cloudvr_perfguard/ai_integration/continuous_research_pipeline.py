"""
Continuous Research Pipeline for CloudVR-PerfGuard
Production-ready automated research generation with scaling and monitoring
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.database import DatabaseManager
from ai_integration.real_data_integration import RealDataResearchPipeline


@dataclass
class ResearchSchedule:
    """Research generation schedule configuration"""
    daily_research: bool = True
    weekly_comprehensive: bool = True
    monthly_deep_analysis: bool = True
    real_time_triggers: bool = True
    
    # Timing configuration
    daily_hour: int = 2  # 2 AM
    weekly_day: int = 0  # Monday
    monthly_day: int = 1  # 1st of month
    
    # Quality thresholds
    min_data_points: int = 10
    max_cost_per_day: float = 50.0
    min_quality_score: float = 75.0


@dataclass
class ResearchMetrics:
    """Research pipeline metrics"""
    papers_generated: int = 0
    functions_discovered: int = 0
    total_cost: float = 0.0
    avg_quality: float = 0.0
    apps_analyzed: int = 0
    data_points_processed: int = 0
    uptime_hours: float = 0.0
    errors_count: int = 0


class ContinuousResearchPipeline:
    """
    Production-ready continuous research pipeline
    Automatically generates AI research from CloudVR-PerfGuard data
    """
    
    def __init__(self, config_path: str = "research_config.json"):
        self.config_path = config_path
        self.schedule = ResearchSchedule()
        self.metrics = ResearchMetrics()
        self.pipeline = RealDataResearchPipeline()
        
        # Research state
        self.is_running = False
        self.last_daily_run = None
        self.last_weekly_run = None
        self.last_monthly_run = None
        
        # Research storage
        self.research_output_dir = Path("research_outputs")
        self.research_output_dir.mkdir(exist_ok=True)
        
        # Logging setup
        self.setup_logging()
        
        # Load configuration
        self.load_config()
    
    def setup_logging(self):
        """Setup logging for the research pipeline"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('research_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ContinuousResearch')
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update schedule from config
                for key, value in config_data.get("schedule", {}).items():
                    if hasattr(self.schedule, key):
                        setattr(self.schedule, key, value)
                
                self.logger.info(f"Configuration loaded from {self.config_path}")
            else:
                self.save_config()
                self.logger.info(f"Default configuration created at {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_data = {
                "schedule": {
                    "daily_research": self.schedule.daily_research,
                    "weekly_comprehensive": self.schedule.weekly_comprehensive,
                    "monthly_deep_analysis": self.schedule.monthly_deep_analysis,
                    "real_time_triggers": self.schedule.real_time_triggers,
                    "daily_hour": self.schedule.daily_hour,
                    "weekly_day": self.schedule.weekly_day,
                    "monthly_day": self.schedule.monthly_day,
                    "min_data_points": self.schedule.min_data_points,
                    "max_cost_per_day": self.schedule.max_cost_per_day,
                    "min_quality_score": self.schedule.min_quality_score
                }
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    async def initialize(self):
        """Initialize the continuous research pipeline"""
        await self.pipeline.initialize()
        self.logger.info("🚀 Continuous Research Pipeline initialized")
    
    async def should_run_daily_research(self) -> bool:
        """Check if daily research should run"""
        if not self.schedule.daily_research:
            return False
        
        now = datetime.now()
        
        # Check if it's the right hour
        if now.hour != self.schedule.daily_hour:
            return False
        
        # Check if we already ran today
        if self.last_daily_run:
            if self.last_daily_run.date() == now.date():
                return False
        
        return True
    
    async def should_run_weekly_research(self) -> bool:
        """Check if weekly comprehensive research should run"""
        if not self.schedule.weekly_comprehensive:
            return False
        
        now = datetime.now()
        
        # Check if it's the right day of week
        if now.weekday() != self.schedule.weekly_day:
            return False
        
        # Check if we already ran this week
        if self.last_weekly_run:
            week_start = now - timedelta(days=now.weekday())
            if self.last_weekly_run >= week_start:
                return False
        
        return True
    
    async def should_run_monthly_research(self) -> bool:
        """Check if monthly deep analysis should run"""
        if not self.schedule.monthly_deep_analysis:
            return False
        
        now = datetime.now()
        
        # Check if it's the right day of month
        if now.day != self.schedule.monthly_day:
            return False
        
        # Check if we already ran this month
        if self.last_monthly_run:
            if (self.last_monthly_run.year == now.year and 
                self.last_monthly_run.month == now.month):
                return False
        
        return True
    
    async def run_daily_research(self) -> Dict[str, Any]:
        """Run daily automated research"""
        self.logger.info("🔄 Starting daily research generation...")
        
        try:
            # Get apps with recent data
            db_manager = DatabaseManager()
            await db_manager.initialize()
            
            recent_jobs = await db_manager.get_recent_jobs(limit=100)
            apps_with_data = set(job["app_name"] for job in recent_jobs 
                               if job["status"] == "completed")
            
            await db_manager.close()
            
            daily_results = {
                "type": "daily_research",
                "timestamp": datetime.utcnow().isoformat(),
                "apps_analyzed": list(apps_with_data),
                "research_items": [],
                "total_cost": 0.0,
                "success": True
            }
            
            # Generate research for each app with sufficient data
            for app_name in apps_with_data:
                if daily_results["total_cost"] >= self.schedule.max_cost_per_day:
                    self.logger.warning(f"Daily cost limit reached: ${daily_results['total_cost']:.2f}")
                    break
                
                app_research = await self.pipeline.generate_research_from_real_data(
                    app_name=app_name,
                    research_type="performance_analysis"
                )
                
                if app_research["success"]:
                    # Save research output
                    output_file = self.research_output_dir / f"daily_{app_name}_{datetime.now().strftime('%Y%m%d')}.json"
                    with open(output_file, 'w') as f:
                        json.dump(app_research, f, indent=2)
                    
                    daily_results["research_items"].append(app_research)
                    daily_results["total_cost"] += app_research.get("total_cost", 0)
                    
                    # Update metrics
                    self.metrics.papers_generated += len(app_research.get("papers", []))
                    self.metrics.functions_discovered += len(app_research.get("functions", []))
                    self.metrics.total_cost += app_research.get("total_cost", 0)
                    self.metrics.data_points_processed += app_research.get("data_count", 0)
            
            self.last_daily_run = datetime.now()
            
            self.logger.info(f"✅ Daily research completed: {len(daily_results['research_items'])} items, ${daily_results['total_cost']:.2f}")
            return daily_results
            
        except Exception as e:
            self.logger.error(f"❌ Daily research failed: {e}")
            self.metrics.errors_count += 1
            return {"success": False, "error": str(e)}
    
    async def run_weekly_research(self) -> Dict[str, Any]:
        """Run weekly comprehensive research"""
        self.logger.info("🔄 Starting weekly comprehensive research...")
        
        try:
            # Generate comprehensive cross-app analysis
            weekly_research = await self.pipeline.generate_research_from_real_data(
                app_name=None,  # All apps
                research_type="comprehensive"
            )
            
            if weekly_research["success"]:
                # Save weekly research
                output_file = self.research_output_dir / f"weekly_comprehensive_{datetime.now().strftime('%Y%m%d')}.json"
                with open(output_file, 'w') as f:
                    json.dump(weekly_research, f, indent=2)
                
                # Update metrics
                self.metrics.papers_generated += len(weekly_research.get("papers", []))
                self.metrics.functions_discovered += len(weekly_research.get("functions", []))
                self.metrics.total_cost += weekly_research.get("total_cost", 0)
                
                self.last_weekly_run = datetime.now()
                
                self.logger.info(f"✅ Weekly research completed: {len(weekly_research.get('papers', []))} papers, {len(weekly_research.get('functions', []))} functions")
                return weekly_research
            else:
                self.logger.error(f"❌ Weekly research failed: {weekly_research.get('error', 'Unknown')}")
                return weekly_research
                
        except Exception as e:
            self.logger.error(f"❌ Weekly research failed: {e}")
            self.metrics.errors_count += 1
            return {"success": False, "error": str(e)}
    
    async def run_monthly_research(self) -> Dict[str, Any]:
        """Run monthly deep analysis"""
        self.logger.info("🔄 Starting monthly deep analysis...")
        
        try:
            # Get extended historical data
            monthly_data = await self.pipeline.get_real_performance_data(
                limit=500,  # More data for deep analysis
                days_back=60  # Look back 2 months
            )
            
            if len(monthly_data) < self.schedule.min_data_points:
                return {
                    "success": False,
                    "error": f"Insufficient data for monthly analysis: {len(monthly_data)} points"
                }
            
            # Generate deep analysis research
            monthly_research = await self.pipeline.generate_research_from_real_data(
                app_name=None,
                research_type="comprehensive"
            )
            
            if monthly_research["success"]:
                # Save monthly research
                output_file = self.research_output_dir / f"monthly_deep_analysis_{datetime.now().strftime('%Y%m')}.json"
                with open(output_file, 'w') as f:
                    json.dump(monthly_research, f, indent=2)
                
                # Generate monthly metrics report
                metrics_report = self.generate_monthly_metrics_report()
                metrics_file = self.research_output_dir / f"monthly_metrics_{datetime.now().strftime('%Y%m')}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics_report, f, indent=2)
                
                self.last_monthly_run = datetime.now()
                
                self.logger.info(f"✅ Monthly deep analysis completed")
                return monthly_research
            else:
                self.logger.error(f"❌ Monthly research failed: {monthly_research.get('error', 'Unknown')}")
                return monthly_research
                
        except Exception as e:
            self.logger.error(f"❌ Monthly research failed: {e}")
            self.metrics.errors_count += 1
            return {"success": False, "error": str(e)}
    
    def generate_monthly_metrics_report(self) -> Dict[str, Any]:
        """Generate monthly metrics report"""
        return {
            "report_date": datetime.utcnow().isoformat(),
            "metrics": {
                "papers_generated": self.metrics.papers_generated,
                "functions_discovered": self.metrics.functions_discovered,
                "total_cost": self.metrics.total_cost,
                "avg_quality": self.metrics.avg_quality,
                "apps_analyzed": self.metrics.apps_analyzed,
                "data_points_processed": self.metrics.data_points_processed,
                "uptime_hours": self.metrics.uptime_hours,
                "errors_count": self.metrics.errors_count
            },
            "efficiency": {
                "cost_per_paper": self.metrics.total_cost / max(1, self.metrics.papers_generated),
                "cost_per_function": self.metrics.total_cost / max(1, self.metrics.functions_discovered),
                "papers_per_day": self.metrics.papers_generated / max(1, self.metrics.uptime_hours / 24),
                "error_rate": self.metrics.errors_count / max(1, self.metrics.papers_generated + self.metrics.functions_discovered)
            }
        }
    
    async def run_continuous_loop(self):
        """Main continuous research loop"""
        self.is_running = True
        start_time = datetime.now()
        
        self.logger.info("🚀 Starting continuous research pipeline...")
        
        while self.is_running:
            try:
                loop_start = datetime.now()
                
                # Check and run scheduled research
                if await self.should_run_daily_research():
                    await self.run_daily_research()
                
                if await self.should_run_weekly_research():
                    await self.run_weekly_research()
                
                if await self.should_run_monthly_research():
                    await self.run_monthly_research()
                
                # Update uptime metrics
                self.metrics.uptime_hours = (datetime.now() - start_time).total_seconds() / 3600
                
                # Sleep for 1 hour before next check
                await asyncio.sleep(3600)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Continuous research pipeline stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in continuous loop: {e}")
                self.metrics.errors_count += 1
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
        
        self.is_running = False
        self.logger.info("🏁 Continuous research pipeline stopped")
    
    async def stop(self):
        """Stop the continuous research pipeline"""
        self.is_running = False
        await self.pipeline.close()
        self.logger.info("✅ Continuous research pipeline stopped gracefully")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "is_running": self.is_running,
            "metrics": {
                "papers_generated": self.metrics.papers_generated,
                "functions_discovered": self.metrics.functions_discovered,
                "total_cost": self.metrics.total_cost,
                "uptime_hours": self.metrics.uptime_hours,
                "errors_count": self.metrics.errors_count
            },
            "last_runs": {
                "daily": self.last_daily_run.isoformat() if self.last_daily_run else None,
                "weekly": self.last_weekly_run.isoformat() if self.last_weekly_run else None,
                "monthly": self.last_monthly_run.isoformat() if self.last_monthly_run else None
            },
            "schedule": {
                "daily_research": self.schedule.daily_research,
                "weekly_comprehensive": self.schedule.weekly_comprehensive,
                "monthly_deep_analysis": self.schedule.monthly_deep_analysis
            }
        }


# Convenience functions for production deployment
async def start_continuous_research(config_path: str = "research_config.json"):
    """Start the continuous research pipeline"""
    pipeline = ContinuousResearchPipeline(config_path)
    await pipeline.initialize()
    
    try:
        await pipeline.run_continuous_loop()
    finally:
        await pipeline.stop()


async def run_manual_research(research_type: str = "comprehensive") -> Dict[str, Any]:
    """Run manual research generation"""
    pipeline = ContinuousResearchPipeline()
    await pipeline.initialize()
    
    try:
        result = await pipeline.pipeline.generate_research_from_real_data(
            research_type=research_type
        )
        return result
    finally:
        await pipeline.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CloudVR-PerfGuard Continuous Research Pipeline")
    parser.add_argument("--mode", choices=["continuous", "daily", "weekly", "monthly", "status"], 
                       default="continuous", help="Pipeline mode")
    parser.add_argument("--config", default="research_config.json", help="Configuration file path")
    
    args = parser.parse_args()
    
    async def main():
        pipeline = ContinuousResearchPipeline(args.config)
        await pipeline.initialize()
        
        try:
            if args.mode == "continuous":
                await pipeline.run_continuous_loop()
            elif args.mode == "daily":
                result = await pipeline.run_daily_research()
                print(json.dumps(result, indent=2))
            elif args.mode == "weekly":
                result = await pipeline.run_weekly_research()
                print(json.dumps(result, indent=2))
            elif args.mode == "monthly":
                result = await pipeline.run_monthly_research()
                print(json.dumps(result, indent=2))
            elif args.mode == "status":
                status = pipeline.get_status()
                print(json.dumps(status, indent=2))
        finally:
            await pipeline.stop()
    
    asyncio.run(main()) 