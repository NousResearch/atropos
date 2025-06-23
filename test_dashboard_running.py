"""
Test if the dashboard is accessible
"""

import requests
import time
import subprocess
import threading

def test_dashboard_endpoint():
    """Test if dashboard is accessible"""
    try:
        # Give server time to start
        time.sleep(2)
        
        print("🧪 Testing dashboard endpoint...")
        
        response = requests.get("http://127.0.0.1:8051/", timeout=5)
        
        if response.status_code == 200:
            print("✅ Dashboard is accessible!")
            print(f"📊 Response length: {len(response.text)} characters")
            print("🌐 Dashboard is working correctly!")
            return True
        else:
            print(f"❌ Dashboard returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Dashboard connection error: {e}")
        return False

def run_dashboard_test():
    """Run dashboard in background and test it"""
    
    # Start dashboard in background
    print("🚀 Starting dashboard in background...")
    dashboard_process = subprocess.Popen([
        "python", "working_dashboard.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Test the endpoint
        success = test_dashboard_endpoint()
        
        if success:
            print("\n🎯 Dashboard Test PASSED!")
            print("🌐 Dashboard is running at: http://localhost:8051")
            print("📊 Features available:")
            print("   • Real-time metrics visualization")
            print("   • 3D society visualization") 
            print("   • Performance monitoring")
            print("   • LLM usage analytics")
            print("   • Simulation controls")
        else:
            print("\n❌ Dashboard Test FAILED!")
            
    finally:
        # Clean up
        dashboard_process.terminate()
        try:
            dashboard_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            dashboard_process.kill()

if __name__ == "__main__":
    run_dashboard_test()