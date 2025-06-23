"""
Dashboard Demo Script
Shows how to use the society visualization dashboard
"""

import subprocess
import sys
import time
import webbrowser


def run_demo():
    """Run dashboard demo"""

    print("🎬 Society Simulation Dashboard Demo")
    print("=" * 50)

    print("\n1. 📊 Generating fresh simulation data...")
    # Run a quick simulation to ensure we have data
    subprocess.run(
        [
            "python",
            "run_simulation.py",
            "--agents",
            "200",
            "--steps",
            "15",
            "--optimized",
            "--workers",
            "4",
            "--quiet",
        ]
    )

    print("2. 🚀 Starting dashboard server...")

    # Start dashboard
    dashboard_process = subprocess.Popen(
        ["python", "working_dashboard.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Give server time to start
        time.sleep(3)

        print("3. 🌐 Opening dashboard in browser...")

        # Try to open browser
        try:
            webbrowser.open("http://localhost:8051")
            print("✅ Dashboard opened in browser!")
        except Exception:
            print("⚠️ Could not auto-open browser")

        print("\n📊 Dashboard Features Available:")
        print("   🔢 Summary Cards - Key metrics at a glance")
        print("   📈 Performance Chart - SPS, runtime, efficiency")
        print("   👥 Agent Analytics - Count, energy, happiness")
        print("   🧠 LLM Metrics - Cache efficiency, request rates")
        print("   🌍 3D Visualization - Society spatial representation")
        print("   🔄 Real-time Updates - Auto-refresh every 10s")
        print("   🎮 Simulation Controls - Run new experiments")

        print("\n🌐 Dashboard URL: http://localhost:8051")
        print("\n⚡ Dashboard Controls:")
        print("   • 🔄 Refresh - Update data manually")
        print("   • 🚀 Run Simulation - Generate new data")
        print("   • Auto-refresh enabled every 10 seconds")

        print("\n📊 Current Metrics Available:")
        print("   • Steps per Second (SPS)")
        print("   • Agent Count & Behavior")
        print("   • Average Energy & Happiness")
        print("   • LLM Cache Hit Rate")
        print("   • Total LLM Requests")
        print("   • Runtime Performance")

        input("\n🎯 Press Enter to stop the demo...")

    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    finally:
        print("\n🧹 Cleaning up...")
        dashboard_process.terminate()
        try:
            dashboard_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            dashboard_process.kill()
        print("✅ Demo completed!")


if __name__ == "__main__":
    run_demo()
