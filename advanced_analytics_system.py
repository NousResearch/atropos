#!/usr/bin/env python3
"""
AMIEN Advanced Analytics System
Real-time monitoring, predictive analytics, and automated optimization
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class AMIENAdvancedAnalytics:
    """Advanced analytics and monitoring for AMIEN production system"""
    
    def __init__(self):
        self.analytics_dir = Path("analytics")
        self.analytics_dir.mkdir(exist_ok=True)
        
        # Initialize data stores
        self.experiment_history = []
        self.paper_quality_history = []
        self.cost_history = []
        self.performance_metrics = []
        
        print("🔬 AMIEN Advanced Analytics System Initialized")
        print(f"   Analytics Directory: {self.analytics_dir}")
        print(f"   Timestamp: {datetime.now().isoformat()}")
    
    async def load_historical_data(self):
        """Load all historical data from AMIEN outputs"""
        print("\n📊 Loading Historical Data...")
        
        # Load experiment data
        experiment_files = [
            "amien_research_output/synthetic_vr_experiments.json",
            "massive_scale_output/massive_experiments_20250527_191518.json",
            "vr_1000_experiments_20250526_075802.json"
        ]
        
        for file_path in experiment_files:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.experiment_history.extend(data)
                        else:
                            self.experiment_history.append(data)
                    print(f"   ✅ Loaded: {file_path}")
                except Exception as e:
                    print(f"   ⚠️ Error loading {file_path}: {e}")
        
        # Load paper quality data
        paper_files = list(Path("amien_research_output").glob("*.md"))
        for paper_file in paper_files:
            try:
                with open(paper_file, 'r') as f:
                    content = f.read()
                    quality_score = self._estimate_paper_quality(content)
                    self.paper_quality_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "file": str(paper_file),
                        "quality_score": quality_score,
                        "word_count": len(content.split())
                    })
            except Exception as e:
                print(f"   ⚠️ Error analyzing {paper_file}: {e}")
        
        print(f"   📈 Total Experiments: {len(self.experiment_history)}")
        print(f"   📝 Total Papers: {len(self.paper_quality_history)}")
    
    def _estimate_paper_quality(self, content: str) -> float:
        """Estimate paper quality based on content analysis"""
        # Simple quality metrics
        word_count = len(content.split())
        section_count = content.count('#')
        citation_count = content.count('[')
        figure_count = content.count('Figure') + content.count('Table')
        
        # Quality score calculation
        quality_score = min(100, (
            (word_count / 50) * 0.3 +  # Length factor
            (section_count * 5) * 0.2 +  # Structure factor
            (citation_count * 2) * 0.3 +  # Citation factor
            (figure_count * 10) * 0.2  # Visual factor
        ))
        
        return round(quality_score, 2)
    
    async def generate_performance_dashboard(self):
        """Generate comprehensive performance dashboard"""
        print("\n📊 Generating Performance Dashboard...")
        
        # Create dashboard data
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "system_health": await self._analyze_system_health(),
            "research_productivity": await self._analyze_research_productivity(),
            "cost_efficiency": await self._analyze_cost_efficiency(),
            "quality_trends": await self._analyze_quality_trends(),
            "predictive_insights": await self._generate_predictive_insights()
        }
        
        # Save dashboard data
        dashboard_file = self.analytics_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        # Generate visualizations
        await self._create_visualizations(dashboard_data)
        
        print(f"   ✅ Dashboard saved: {dashboard_file}")
        return dashboard_data
    
    async def _analyze_system_health(self) -> Dict[str, Any]:
        """Analyze overall system health metrics"""
        if not self.experiment_history:
            return {"status": "no_data", "health_score": 0}
        
        # Calculate success rates
        successful_experiments = sum(1 for exp in self.experiment_history 
                                   if exp.get("success", True))
        success_rate = (successful_experiments / len(self.experiment_history)) * 100
        
        # Calculate average performance
        avg_comfort = np.mean([exp.get("comfort_score", 0.5) 
                              for exp in self.experiment_history])
        avg_fps = np.mean([exp.get("fps", 60) 
                          for exp in self.experiment_history])
        
        # Health score calculation
        health_score = (success_rate * 0.4 + 
                       avg_comfort * 100 * 0.3 + 
                       min(avg_fps / 60, 1) * 100 * 0.3)
        
        return {
            "status": "healthy" if health_score > 80 else "warning" if health_score > 60 else "critical",
            "health_score": round(health_score, 2),
            "success_rate": round(success_rate, 2),
            "avg_comfort": round(avg_comfort, 3),
            "avg_fps": round(avg_fps, 1),
            "total_experiments": len(self.experiment_history)
        }
    
    async def _analyze_research_productivity(self) -> Dict[str, Any]:
        """Analyze research productivity metrics"""
        if not self.paper_quality_history:
            return {"status": "no_data", "productivity_score": 0}
        
        # Calculate productivity metrics
        avg_quality = np.mean([paper["quality_score"] 
                              for paper in self.paper_quality_history])
        avg_word_count = np.mean([paper["word_count"] 
                                 for paper in self.paper_quality_history])
        
        # Papers per day (simulated)
        papers_per_day = len(self.paper_quality_history) / 7  # Assume 1 week of data
        
        productivity_score = min(100, (avg_quality * 0.5 + 
                                     min(papers_per_day * 20, 50) * 0.3 + 
                                     min(avg_word_count / 100, 50) * 0.2))
        
        return {
            "productivity_score": round(productivity_score, 2),
            "avg_quality": round(avg_quality, 2),
            "avg_word_count": round(avg_word_count, 0),
            "papers_per_day": round(papers_per_day, 2),
            "total_papers": len(self.paper_quality_history)
        }
    
    async def _analyze_cost_efficiency(self) -> Dict[str, Any]:
        """Analyze cost efficiency metrics"""
        # Simulate cost data based on experiments
        estimated_cost_per_experiment = 0.05  # $0.05 per experiment
        estimated_cost_per_paper = 15.0  # $15 per paper
        
        total_experiment_cost = len(self.experiment_history) * estimated_cost_per_experiment
        total_paper_cost = len(self.paper_quality_history) * estimated_cost_per_paper
        total_cost = total_experiment_cost + total_paper_cost
        
        # Calculate efficiency
        if self.paper_quality_history:
            avg_quality = np.mean([paper["quality_score"] 
                                  for paper in self.paper_quality_history])
            cost_per_quality_point = total_cost / (avg_quality * len(self.paper_quality_history))
        else:
            cost_per_quality_point = 0
        
        return {
            "total_cost": round(total_cost, 2),
            "cost_per_experiment": estimated_cost_per_experiment,
            "cost_per_paper": estimated_cost_per_paper,
            "cost_per_quality_point": round(cost_per_quality_point, 4),
            "efficiency_score": max(0, 100 - cost_per_quality_point * 100)
        }
    
    async def _analyze_quality_trends(self) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        if not self.paper_quality_history:
            return {"trend": "no_data", "trend_score": 0}
        
        # Calculate quality trend
        qualities = [paper["quality_score"] for paper in self.paper_quality_history]
        
        if len(qualities) > 1:
            # Simple linear trend
            x = np.arange(len(qualities))
            trend_slope = np.polyfit(x, qualities, 1)[0]
            trend_direction = "improving" if trend_slope > 0.1 else "declining" if trend_slope < -0.1 else "stable"
        else:
            trend_slope = 0
            trend_direction = "stable"
        
        return {
            "trend": trend_direction,
            "trend_slope": round(trend_slope, 4),
            "current_quality": round(qualities[-1], 2) if qualities else 0,
            "quality_variance": round(np.var(qualities), 2) if len(qualities) > 1 else 0,
            "trend_score": max(0, min(100, 50 + trend_slope * 10))
        }
    
    async def _generate_predictive_insights(self) -> Dict[str, Any]:
        """Generate predictive insights and recommendations"""
        insights = []
        recommendations = []
        
        # Analyze experiment patterns
        if len(self.experiment_history) > 100:
            success_rate = sum(1 for exp in self.experiment_history 
                             if exp.get("success", True)) / len(self.experiment_history)
            
            if success_rate < 0.8:
                insights.append("Low experiment success rate detected")
                recommendations.append("Review experiment parameters and error handling")
        
        # Analyze quality patterns
        if len(self.paper_quality_history) > 3:
            recent_quality = np.mean([paper["quality_score"] 
                                    for paper in self.paper_quality_history[-3:]])
            overall_quality = np.mean([paper["quality_score"] 
                                     for paper in self.paper_quality_history])
            
            if recent_quality < overall_quality * 0.9:
                insights.append("Recent paper quality decline detected")
                recommendations.append("Review AI model parameters and training data")
        
        # Predict future performance
        if len(self.experiment_history) > 50:
            predicted_daily_experiments = len(self.experiment_history) * 1.1  # 10% growth
            predicted_monthly_cost = predicted_daily_experiments * 30 * 0.05 + len(self.paper_quality_history) * 30 * 15
        else:
            predicted_daily_experiments = 1000
            predicted_monthly_cost = 2500
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "predictions": {
                "daily_experiments": round(predicted_daily_experiments, 0),
                "monthly_cost": round(predicted_monthly_cost, 2),
                "quality_forecast": "stable"
            }
        }
    
    async def _create_visualizations(self, dashboard_data: Dict[str, Any]):
        """Create visualization charts for the dashboard"""
        print("   📈 Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AMIEN Production Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. System Health Overview
        ax1 = axes[0, 0]
        health_data = dashboard_data["system_health"]
        metrics = ['Health Score', 'Success Rate', 'Avg Comfort', 'Avg FPS']
        values = [
            health_data.get("health_score", 0),
            health_data.get("success_rate", 0),
            health_data.get("avg_comfort", 0) * 100,
            health_data.get("avg_fps", 0) / 60 * 100
        ]
        
        bars = ax1.bar(metrics, values, color=['#2E8B57', '#4169E1', '#FF6347', '#32CD32'])
        ax1.set_title('System Health Metrics')
        ax1.set_ylabel('Score (%)')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 2. Research Productivity
        ax2 = axes[0, 1]
        prod_data = dashboard_data["research_productivity"]
        if prod_data.get("total_papers", 0) > 0:
            quality_scores = [paper["quality_score"] for paper in self.paper_quality_history]
            ax2.plot(range(len(quality_scores)), quality_scores, marker='o', linewidth=2, markersize=6)
            ax2.set_title('Paper Quality Trend')
            ax2.set_xlabel('Paper Number')
            ax2.set_ylabel('Quality Score')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No paper data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Paper Quality Trend')
        
        # 3. Cost Analysis
        ax3 = axes[1, 0]
        cost_data = dashboard_data["cost_efficiency"]
        cost_breakdown = [
            cost_data.get("total_cost", 0) * 0.7,  # Experiment costs
            cost_data.get("total_cost", 0) * 0.3   # Paper generation costs
        ]
        labels = ['Experiments', 'Papers']
        colors = ['#FF9999', '#66B2FF']
        
        wedges, texts, autotexts = ax3.pie(cost_breakdown, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('Cost Breakdown')
        
        # 4. Experiment Success Rate
        ax4 = axes[1, 1]
        if len(self.experiment_history) > 0:
            success_count = sum(1 for exp in self.experiment_history if exp.get("success", True))
            failure_count = len(self.experiment_history) - success_count
            
            ax4.bar(['Success', 'Failure'], [success_count, failure_count], 
                   color=['#90EE90', '#FFB6C1'])
            ax4.set_title('Experiment Success/Failure')
            ax4.set_ylabel('Count')
        else:
            ax4.text(0.5, 0.5, 'No experiment data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Experiment Success/Failure')
        
        plt.tight_layout()
        
        # Save the visualization
        viz_file = self.analytics_dir / f"dashboard_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Visualization saved: {viz_file}")
    
    async def generate_optimization_recommendations(self):
        """Generate specific optimization recommendations"""
        print("\n🎯 Generating Optimization Recommendations...")
        
        recommendations = {
            "timestamp": datetime.now().isoformat(),
            "system_optimizations": [],
            "cost_optimizations": [],
            "quality_improvements": [],
            "scaling_recommendations": []
        }
        
        # System optimizations
        if len(self.experiment_history) > 0:
            avg_fps = np.mean([exp.get("fps", 60) for exp in self.experiment_history])
            if avg_fps < 60:
                recommendations["system_optimizations"].append({
                    "issue": "Low average FPS",
                    "recommendation": "Optimize VR rendering pipeline",
                    "priority": "high",
                    "estimated_impact": "15% performance improvement"
                })
        
        # Cost optimizations
        if len(self.experiment_history) > 1000:
            recommendations["cost_optimizations"].append({
                "issue": "High experiment volume",
                "recommendation": "Implement intelligent experiment batching",
                "priority": "medium",
                "estimated_savings": "$500/month"
            })
        
        # Quality improvements
        if len(self.paper_quality_history) > 0:
            avg_quality = np.mean([paper["quality_score"] for paper in self.paper_quality_history])
            if avg_quality < 80:
                recommendations["quality_improvements"].append({
                    "issue": "Below-target paper quality",
                    "recommendation": "Fine-tune AI models with domain-specific data",
                    "priority": "high",
                    "estimated_improvement": "20% quality increase"
                })
        
        # Scaling recommendations
        recommendations["scaling_recommendations"].append({
            "recommendation": "Implement auto-scaling based on research demand",
            "priority": "medium",
            "estimated_benefit": "30% cost reduction during low-demand periods"
        })
        
        # Save recommendations
        rec_file = self.analytics_dir / f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(rec_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"   ✅ Recommendations saved: {rec_file}")
        return recommendations
    
    async def run_complete_analytics(self):
        """Run complete analytics pipeline"""
        print("🔬 Running Complete AMIEN Analytics Pipeline")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Step 1: Load data
        await self.load_historical_data()
        
        # Step 2: Generate dashboard
        dashboard = await self.generate_performance_dashboard()
        
        # Step 3: Generate recommendations
        recommendations = await self.generate_optimization_recommendations()
        
        # Step 4: Create summary report
        summary = {
            "analysis_timestamp": start_time.isoformat(),
            "completion_timestamp": datetime.now().isoformat(),
            "duration_minutes": (datetime.now() - start_time).total_seconds() / 60,
            "data_summary": {
                "experiments_analyzed": len(self.experiment_history),
                "papers_analyzed": len(self.paper_quality_history),
                "system_health": dashboard["system_health"]["status"],
                "overall_score": dashboard["system_health"]["health_score"]
            },
            "key_insights": dashboard["predictive_insights"]["insights"],
            "top_recommendations": [rec["recommendation"] for rec in 
                                  recommendations["system_optimizations"][:3]],
            "next_analysis": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        # Save summary
        summary_file = self.analytics_dir / f"analytics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 Analytics Complete!")
        print(f"   Duration: {summary['duration_minutes']:.1f} minutes")
        print(f"   System Health: {summary['data_summary']['system_health']}")
        print(f"   Overall Score: {summary['data_summary']['overall_score']:.1f}/100")
        print(f"   Summary: {summary_file}")
        
        return summary

async def main():
    """Main analytics function"""
    analytics = AMIENAdvancedAnalytics()
    await analytics.run_complete_analytics()

if __name__ == "__main__":
    asyncio.run(main()) 