"""
Test individual dashboard components to identify issues
"""

from working_dashboard import WorkingSocietyDashboard
import traceback

def test_components():
    print("🧪 Testing Dashboard Components...")
    
    try:
        # Test dashboard creation
        print("1. Creating dashboard instance...")
        dashboard = WorkingSocietyDashboard()
        print("✅ Dashboard instance created")
        
        # Test data loading
        print("2. Testing data loading...")
        df, categories = dashboard.load_metrics_data()
        print(f"✅ Loaded {len(df)} rows")
        print(f"✅ Categories: {list(categories.keys())}")
        
        if not df.empty:
            print(f"✅ Columns: {list(df.columns)}")
            print(f"✅ Steps range: {df['step'].min()} - {df['step'].max()}")
            
            # Test visualization creation
            print("3. Testing chart creation...")
            
            try:
                fig1 = dashboard.create_performance_chart(df)
                print("✅ Performance chart created")
            except Exception as e:
                print(f"❌ Performance chart error: {e}")
                
            try:
                fig2 = dashboard.create_3d_visualization(df)
                print("✅ 3D visualization created")
            except Exception as e:
                print(f"❌ 3D visualization error: {e}")
                
            try:
                cards = dashboard.create_summary_cards(df)
                print("✅ Summary cards created")
            except Exception as e:
                print(f"❌ Summary cards error: {e}")
        else:
            print("⚠️ No data to test visualizations")
            
        print("4. Testing app layout...")
        layout = dashboard.app.layout
        print("✅ App layout created")
        
        print("\n🎯 Component test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_components()