"""
Test script for the 3D visualization dashboard
"""

from advanced_visualization import SocietyVisualizationDashboard
import pandas as pd
import sqlite3

def test_dashboard():
    """Test the dashboard components"""
    print("🧪 Testing Society Visualization Dashboard...")
    
    # Create dashboard instance
    dashboard = SocietyVisualizationDashboard(max_points=100)
    
    # Test data loading
    print("📊 Testing data loading...")
    df = dashboard.load_simulation_data()
    print(f"✅ Loaded {len(df)} data points")
    
    if len(df) > 0:
        print(f"📈 Data range - Steps: {df['step'].min()}-{df['step'].max()}")
        print(f"👥 Agents: {df['agent_id'].nunique()}")
        
        # Test visualization creation
        print("🎨 Testing 3D scatter plot...")
        fig_3d = dashboard.create_3d_scatter(df)
        print(f"✅ 3D plot created with {len(fig_3d.data)} traces")
        
        print("🔥 Testing density heatmap...")
        fig_heat = dashboard.create_agent_heatmap(df)
        print(f"✅ Heatmap created with {len(fig_heat.data)} traces")
        
        print("📊 Testing metrics timeline...")
        fig_timeline = dashboard.create_metrics_timeline(df)
        print(f"✅ Timeline created with {len(fig_timeline.data)} traces")
        
        # Show sample statistics
        latest_step = df['step'].max()
        latest_data = df[df['step'] == latest_step]
        
        print(f"\n📊 Latest Statistics (Step {latest_step}):")
        print(f"   👥 Agents: {len(latest_data)}")
        print(f"   ⚡ Avg Energy: {latest_data['energy'].mean():.2f}")
        print(f"   😊 Avg Happiness: {latest_data['happiness'].mean():.2f}")
        print(f"   💰 Avg Wealth: {latest_data['wealth'].mean():.2f}")
        print(f"   🤝 Total Interactions: {latest_data['interaction_count'].sum()}")
        
    else:
        print("⚠️ No simulation data found. Run a simulation first!")
    
    print("\n🎯 Dashboard test completed!")
    print("🌐 To launch full dashboard run: python advanced_visualization.py")
    print("📱 Then open: http://localhost:8050")

if __name__ == "__main__":
    test_dashboard()