"""
CloudVR-PerfGuard: Automated Performance Regression Detection for VR Applications
Adapted from AMIEN infrastructure for VR performance testing
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import uuid
import asyncio
import json
from datetime import datetime
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.performance_tester import VRPerformanceTester
from core.regression_detector import RegressionDetector
from core.database import DatabaseManager

app = FastAPI(
    title="CloudVR-PerfGuard API",
    description="Automated Performance Regression Detection for VR Applications",
    version="1.0.0"
)

# Global instances
performance_tester: Optional[VRPerformanceTester] = None
regression_detector: Optional[RegressionDetector] = None
db_manager: Optional[DatabaseManager] = None

# --- Pydantic Models ---

class BuildSubmissionRequest(BaseModel):
    app_name: str = Field(..., example="MyVRGame")
    build_version: str = Field(..., example="v1.2.3")
    platform: str = Field(..., example="windows", description="windows, linux, android")
    submission_type: str = Field(..., example="baseline", description="baseline or regression_test")
    baseline_version: Optional[str] = Field(None, example="v1.2.2", description="Required for regression_test")

class PerformanceTestConfig(BaseModel):
    gpu_types: List[str] = Field(default=["T4"], example=["T4", "L4"])
    test_duration_seconds: int = Field(default=60, example=60)
    test_scenes: List[str] = Field(default=["main_menu"], example=["main_menu", "gameplay_scene"])

class RegressionTestResult(BaseModel):
    job_id: str
    app_name: str
    build_version: str
    baseline_version: str
    status: str  # "queued", "running", "completed", "failed"
    regressions_detected: Optional[List[Dict[str, Any]]] = None
    performance_data: Optional[Dict[str, Any]] = None
    created_at: str

# --- Startup/Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    global performance_tester, regression_detector, db_manager
    
    try:
        # Initialize core components
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        performance_tester = VRPerformanceTester()
        await performance_tester.initialize()
        
        regression_detector = RegressionDetector(db_manager)
        
        print("INFO: CloudVR-PerfGuard API initialized successfully")
        
    except Exception as e:
        print(f"FATAL: Failed to initialize CloudVR-PerfGuard: {e}")
        import traceback
        traceback.print_exc()

@app.on_event("shutdown")
async def shutdown_event():
    if performance_tester:
        await performance_tester.cleanup()
    if db_manager:
        await db_manager.close()
    print("INFO: CloudVR-PerfGuard API shutdown complete")

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {
        "message": "CloudVR-PerfGuard API is operational!",
        "version": "1.0.0",
        "description": "Automated Performance Regression Detection for VR Applications"
    }

@app.get("/status")
async def get_status():
    """Get API and service status"""
    status = {
        "api_status": "OPERATIONAL",
        "services": {
            "performance_tester": "initialized" if performance_tester else "not_initialized",
            "regression_detector": "initialized" if regression_detector else "not_initialized",
            "database": "connected" if db_manager else "not_connected"
        },
        "supported_platforms": ["windows", "linux", "android"],
        "supported_gpu_types": ["T4", "L4", "A100"],
        "timestamp": datetime.utcnow().isoformat()
    }
    return status

@app.post("/submit_build", response_model=Dict[str, Any])
async def submit_build_for_testing(
    file: UploadFile = File(...),
    app_name: str = Form(...),
    build_version: str = Form(...),
    platform: str = Form(...),
    submission_type: str = Form(...),
    baseline_version: Optional[str] = Form(None),
    test_config: Optional[str] = Form(None)  # JSON string
):
    """Submit a VR application build for performance testing"""
    
    if not performance_tester or not db_manager:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    # Validate submission type
    if submission_type not in ["baseline", "regression_test"]:
        raise HTTPException(status_code=400, detail="submission_type must be 'baseline' or 'regression_test'")
    
    if submission_type == "regression_test" and not baseline_version:
        raise HTTPException(status_code=400, detail="baseline_version required for regression_test")
    
    # Parse test configuration
    config = PerformanceTestConfig()
    if test_config:
        try:
            config_dict = json.loads(test_config)
            config = PerformanceTestConfig(**config_dict)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid test_config JSON: {e}")
    
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Store build file (implement file storage logic)
        build_path = await store_build_file(file, app_name, build_version, job_id)
        
        # Create database record
        await db_manager.create_test_job(
            job_id=job_id,
            app_name=app_name,
            build_version=build_version,
            platform=platform,
            submission_type=submission_type,
            baseline_version=baseline_version,
            build_path=build_path,
            test_config=config.dict()
        )
        
        # Queue performance test (implement async task queue)
        await queue_performance_test(job_id, build_path, config)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"Build submitted for {submission_type} testing",
            "app_name": app_name,
            "build_version": build_version,
            "estimated_completion": "5-10 minutes"
        }
        
    except Exception as e:
        print(f"ERROR in submit_build: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit build: {str(e)}")

@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a performance test job"""
    
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        job_data = await db_manager.get_test_job(job_id)
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "job_id": job_id,
            "status": job_data["status"],
            "app_name": job_data["app_name"],
            "build_version": job_data["build_version"],
            "submission_type": job_data["submission_type"],
            "created_at": job_data["created_at"],
            "progress": job_data.get("progress", 0),
            "estimated_completion": job_data.get("estimated_completion")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in get_job_status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.get("/regression_report/{job_id}")
async def get_regression_report(job_id: str):
    """Get detailed regression test report"""
    
    if not db_manager or not regression_detector:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    try:
        job_data = await db_manager.get_test_job(job_id)
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_data["status"] != "completed":
            return {
                "job_id": job_id,
                "status": job_data["status"],
                "message": "Report not ready yet"
            }
        
        # Get regression analysis
        report = await regression_detector.generate_report(job_id)
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in get_regression_report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@app.get("/regression_report/{job_id}/html", response_class=HTMLResponse)
async def get_regression_report_html(job_id: str):
    """Get regression report as HTML"""
    
    try:
        report_data = await get_regression_report(job_id)
        
        # Generate HTML report (implement HTML template)
        html_content = generate_html_report(report_data)
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate HTML report: {str(e)}")

@app.get("/apps/{app_name}/baselines")
async def get_app_baselines(app_name: str):
    """Get available baseline versions for an app"""
    
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        baselines = await db_manager.get_app_baselines(app_name)
        return {
            "app_name": app_name,
            "baselines": baselines
        }
        
    except Exception as e:
        print(f"ERROR in get_app_baselines: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get baselines: {str(e)}")

# --- Helper Functions ---

async def store_build_file(file: UploadFile, app_name: str, build_version: str, job_id: str) -> str:
    """Store uploaded build file and return path"""
    # TODO: Implement GCS storage
    # For now, store locally
    storage_dir = f"/tmp/cloudvr_builds/{app_name}/{build_version}"
    os.makedirs(storage_dir, exist_ok=True)
    
    file_path = f"{storage_dir}/{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return file_path

async def queue_performance_test(job_id: str, build_path: str, config: Dict[str, Any]):
    """Queue performance test for execution"""
    # TODO: Implement proper async task queue (Celery, Cloud Tasks, etc.)
    # For now, start background task
    asyncio.create_task(run_performance_test_background(job_id, build_path, config))

async def run_performance_test_background(job_id: str, build_path: str, config: Dict[str, Any]):
    """Run performance test in background"""
    try:
        # Update job status
        await db_manager.update_job_status(job_id, "running")
        
        # Run performance test
        results = await performance_tester.run_test(build_path, config)
        
        # Store results
        await db_manager.store_performance_results(job_id, results)
        
        # If this is a regression test, run regression analysis
        job_data = await db_manager.get_test_job(job_id)
        if job_data["submission_type"] == "regression_test":
            await regression_detector.analyze_regression(job_id)
        
        # Update job status
        await db_manager.update_job_status(job_id, "completed")
        
    except Exception as e:
        print(f"ERROR in background performance test {job_id}: {e}")
        await db_manager.update_job_status(job_id, "failed", error=str(e))

def generate_html_report(report_data: Dict[str, Any]) -> str:
    """Generate HTML report from report data"""
    # TODO: Implement proper HTML template
    return f"""
    <html>
    <head><title>CloudVR-PerfGuard Report</title></head>
    <body>
        <h1>Performance Regression Report</h1>
        <pre>{json.dumps(report_data, indent=2)}</pre>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 