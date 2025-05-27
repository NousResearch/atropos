#!/usr/bin/env python3
"""
CloudVR-PerfGuard - Main Entry Point
Automated Performance Regression Detection for VR Applications

Usage:
    python main.py                    # Start API server
    python main.py --dev              # Start in development mode
    python main.py --test             # Run basic functionality test
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="CloudVR-PerfGuard: VR Performance Regression Testing")
    parser.add_argument("--dev", action="store_true", help="Run in development mode")
    parser.add_argument("--test", action="store_true", help="Run basic functionality test")
    parser.add_argument("--port", type=int, default=8000, help="Port to run API server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind API server to")
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(run_tests())
    else:
        asyncio.run(start_api_server(args.host, args.port, args.dev))

async def start_api_server(host: str, port: int, dev_mode: bool = False):
    """Start the FastAPI server"""
    
    print("🚀 Starting CloudVR-PerfGuard API Server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Development mode: {dev_mode}")
    
    try:
        import uvicorn
        from api.main import app
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            reload=dev_mode,
            log_level="info" if not dev_mode else "debug"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        print("   Make sure you've installed the requirements: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

async def run_tests():
    """Run basic functionality tests"""
    
    print("🧪 Running CloudVR-PerfGuard functionality tests...")
    
    try:
        # Test 1: Import all modules
        print("  ✓ Testing module imports...")
        from core.performance_tester import VRPerformanceTester
        from core.regression_detector import RegressionDetector
        from core.database import DatabaseManager
        from core.gpu_monitor import GPUMonitor
        from core.container_manager import ContainerManager
        print("    ✅ All core modules imported successfully")
        
        # Test 2: Database initialization
        print("  ✓ Testing database initialization...")
        db_manager = DatabaseManager(db_path="test_cloudvr_perfguard.db")
        await db_manager.initialize()
        await db_manager.close()
        
        # Cleanup test database
        if os.path.exists("test_cloudvr_perfguard.db"):
            os.remove("test_cloudvr_perfguard.db")
        
        print("    ✅ Database initialization successful")
        
        # Test 3: GPU monitor initialization
        print("  ✓ Testing GPU monitor...")
        gpu_monitor = GPUMonitor()
        await gpu_monitor.initialize()
        await gpu_monitor.cleanup()
        print("    ✅ GPU monitor initialization successful")
        
        # Test 4: Performance tester initialization
        print("  ✓ Testing performance tester...")
        # Note: This will fail without Docker, but we can test the import
        try:
            performance_tester = VRPerformanceTester()
            print("    ✅ Performance tester created successfully")
        except Exception as e:
            print(f"    ⚠️ Performance tester initialization failed (expected without Docker): {e}")
        
        print("\n🎉 All basic functionality tests passed!")
        print("\n📋 Next steps:")
        print("   1. Install Docker and nvidia-docker for full functionality")
        print("   2. Start the API server: python main.py")
        print("   3. Visit http://localhost:8000/docs for API documentation")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 