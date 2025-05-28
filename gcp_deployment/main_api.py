#!/usr/bin/env python3
"""
AMIEN Main API Service
FastAPI service for AMIEN research pipeline
"""

import asyncio
import os
import json
import traceback
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage, secretmanager
import uvicorn

app = FastAPI(title="AMIEN Research API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Cloud clients
storage_client = storage.Client()
secret_client = secretmanager.SecretManagerServiceClient()

@app.get("/")
async def root():
    return {"message": "AMIEN AI Research Pipeline API", "status": "operational"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/research/generate")
async def generate_research(background_tasks: BackgroundTasks):
    """Generate AI research papers and functions"""
    background_tasks.add_task(run_research_generation)
    return {"message": "Research generation started", "status": "processing"}

@app.post("/experiments/massive")
async def run_massive_experiments(background_tasks: BackgroundTasks, sample_size: int = 1000):
    """Run massive scale VR experiments"""
    background_tasks.add_task(run_massive_scale, sample_size)
    return {"message": f"Massive scale experiments started with {sample_size} samples", "status": "processing"}

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "api_status": "healthy",
        "services": {
            "ai_scientist": "available",
            "funsearch": "available", 
            "gemini": "available" if os.getenv("GEMINI_API_KEY") else "unavailable",
            "cloud_storage": "available",
            "secret_manager": "available"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/research/list")
async def list_research():
    """List generated research artifacts"""
    try:
        bucket = storage_client.bucket("amien-research-artifacts")
        blobs = bucket.list_blobs(prefix="papers/")
        papers = [{"name": blob.name, "created": blob.time_created.isoformat(), "size": blob.size} for blob in blobs]
        return {"papers": papers, "count": len(papers)}
    except Exception as e:
        return {"papers": [], "count": 0, "error": str(e)}

async def run_research_generation():
    """Background task for real research generation"""
    try:
        print("🔬 Starting AMIEN research generation...")
        
        # Import research modules
        import sys
        sys.path.append('/app')
        
        # Generate research using evolution system
        from evolution_research_generator import generate_vr_research
        
        # Run evolution-based research
        research_results = await generate_vr_research()
        
        # Store results in Cloud Storage
        await store_research_artifacts(research_results)
        
        print("✅ Research generation completed successfully")
        
    except Exception as e:
        print(f"❌ Research generation failed: {str(e)}")
        print(traceback.format_exc())

async def run_massive_scale(sample_size: int):
    """Background task for massive scale VR experiments"""
    try:
        print(f"🧪 Starting massive scale experiments with {sample_size} samples...")
        
        # Import experiment modules
        import sys
        sys.path.append('/app')
        
        from massive_scale_runner import run_vr_experiments
        
        # Run large-scale experiments
        experiment_results = await run_vr_experiments(sample_size)
        
        # Store results in Cloud Storage
        await store_experiment_results(experiment_results, sample_size)
        
        print("✅ Massive scale experiments completed successfully")
        
    except Exception as e:
        print(f"❌ Massive scale experiments failed: {str(e)}")
        print(traceback.format_exc())

async def store_research_artifacts(results):
    """Store research results in Cloud Storage"""
    try:
        bucket = storage_client.bucket("amien-research-artifacts")
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Store paper
        if "paper" in results:
            paper_blob = bucket.blob(f"papers/research_paper_{timestamp}.md")
            paper_blob.upload_from_string(results["paper"])
        
        # Store data
        if "data" in results:
            data_blob = bucket.blob(f"data/research_data_{timestamp}.json")
            data_blob.upload_from_string(json.dumps(results["data"], indent=2))
        
        # Store functions
        if "functions" in results:
            func_blob = bucket.blob(f"functions/discovered_functions_{timestamp}.py")
            func_blob.upload_from_string(results["functions"])
            
        print(f"📁 Research artifacts stored with timestamp {timestamp}")
        
    except Exception as e:
        print(f"❌ Failed to store research artifacts: {str(e)}")

async def store_experiment_results(results, sample_size):
    """Store experiment results in Cloud Storage"""
    try:
        bucket = storage_client.bucket("amien-research-artifacts")
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Store experiment results
        results_blob = bucket.blob(f"experiments/massive_scale_{sample_size}_{timestamp}.json")
        results_blob.upload_from_string(json.dumps(results, indent=2))
        
        print(f"📊 Experiment results stored for {sample_size} samples at {timestamp}")
        
    except Exception as e:
        print(f"❌ Failed to store experiment results: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting AMIEN API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
