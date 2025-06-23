"""
Cloud API for Society Simulation
Provides REST endpoints for running simulations at scale
"""

import os
import json
import asyncio
import subprocess
import time
from datetime import datetime
from flask import Flask, request, jsonify
from google.cloud import storage, secretmanager
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3

app = Flask(__name__)

# Global configuration
PROJECT_ID = os.environ.get('PROJECT_ID', 'amien-research-pipeline')
BUCKET_NAME = f"{PROJECT_ID}-simulation-results"

# Thread pool for async simulations
executor = ThreadPoolExecutor(max_workers=10)

class CloudSimulationManager:
    def __init__(self):
        self.storage_client = storage.Client()
        self.running_simulations = {}
        
    def run_simulation(self, sim_id, agents, steps, workers=4):
        """Run simulation and upload results to cloud storage"""
        try:
            print(f"🚀 Starting simulation {sim_id}: {agents} agents, {steps} steps")
            
            # Run the simulation
            start_time = time.time()
            result = subprocess.run([
                'python', '/app/run_simulation_cloud.py',
                '--agents', str(agents),
                '--steps', str(steps),
                '--save', f'/tmp/sim_{sim_id}_results.json'
            ], capture_output=True, text=True, timeout=3600)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                # Parse results
                try:
                    with open(f'/tmp/sim_{sim_id}_results.json', 'r') as f:
                        sim_data = json.load(f)
                    
                    # Add cloud metadata
                    cloud_metadata = {
                        'simulation_id': sim_id,
                        'cloud_timestamp': datetime.utcnow().isoformat(),
                        'duration_seconds': duration,
                        'stdout': result.stdout,
                        'performance_summary': self.extract_performance_metrics(result.stdout)
                    }
                    
                    sim_data['cloud_metadata'] = cloud_metadata
                    
                    # Upload to cloud storage
                    self.upload_results(sim_id, sim_data)
                    
                    # Update status
                    self.running_simulations[sim_id] = {
                        'status': 'completed',
                        'duration': duration,
                        'results_url': f'gs://{BUCKET_NAME}/simulations/{sim_id}/results.json'
                    }
                    
                    print(f"✅ Simulation {sim_id} completed in {duration:.2f}s")
                    return sim_data
                    
                except Exception as e:
                    print(f"❌ Error processing results for {sim_id}: {e}")
                    self.running_simulations[sim_id] = {'status': 'failed', 'error': str(e)}
                    
            else:
                print(f"❌ Simulation {sim_id} failed: {result.stderr}")
                self.running_simulations[sim_id] = {
                    'status': 'failed', 
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Simulation {sim_id} timed out")
            self.running_simulations[sim_id] = {'status': 'timeout'}
        except Exception as e:
            print(f"❌ Simulation {sim_id} error: {e}")
            self.running_simulations[sim_id] = {'status': 'error', 'error': str(e)}
    
    def extract_performance_metrics(self, stdout):
        """Extract key performance metrics from simulation output"""
        metrics = {}
        try:
            lines = stdout.split('\n')
            for line in lines:
                if 'Average SPS:' in line:
                    metrics['average_sps'] = float(line.split('Average SPS:')[1].strip())
                elif 'Peak SPS:' in line:
                    metrics['peak_sps'] = float(line.split('Peak SPS:')[1].strip())
                elif 'Social Density:' in line:
                    metrics['social_density'] = float(line.split('Social Density:')[1].split()[0])
                elif 'Economic Activity:' in line:
                    metrics['economic_activity'] = float(line.split('Economic Activity:')[1].split()[0])
                elif 'Memory Usage:' in line:
                    metrics['memory_usage'] = line.split('Memory Usage:')[1].strip()
        except:
            pass
        return metrics
    
    def upload_results(self, sim_id, data):
        """Upload simulation results to Cloud Storage"""
        try:
            bucket = self.storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(f'simulations/{sim_id}/results.json')
            blob.upload_from_string(
                json.dumps(data, indent=2),
                content_type='application/json'
            )
        except Exception as e:
            print(f"❌ Failed to upload results for {sim_id}: {e}")

# Initialize simulation manager
sim_manager = CloudSimulationManager()

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'service': 'Society Simulation Cloud API',
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'running_simulations': len(sim_manager.running_simulations)
    })

@app.route('/simulation/run', methods=['POST'])
def run_simulation():
    """Start a new simulation"""
    try:
        data = request.get_json()
        
        # Validate parameters
        agents = data.get('agents', 100)
        steps = data.get('steps', 50)
        workers = data.get('workers', 4)
        
        # Validate limits
        if agents > 5000:
            return jsonify({'error': 'Agent limit is 5000'}), 400
        if steps > 1000:
            return jsonify({'error': 'Step limit is 1000'}), 400
            
        # Generate simulation ID
        sim_id = f"{int(time.time())}_{agents}_{steps}"
        
        # Start simulation asynchronously
        sim_manager.running_simulations[sim_id] = {
            'status': 'running',
            'started_at': datetime.utcnow().isoformat(),
            'parameters': {'agents': agents, 'steps': steps, 'workers': workers}
        }
        
        # Submit to thread pool
        future = executor.submit(sim_manager.run_simulation, sim_id, agents, steps, workers)
        
        return jsonify({
            'simulation_id': sim_id,
            'status': 'started',
            'parameters': {'agents': agents, 'steps': steps, 'workers': workers},
            'check_status_url': f'/simulation/status/{sim_id}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/simulation/status/<sim_id>', methods=['GET'])
def simulation_status(sim_id):
    """Check simulation status"""
    if sim_id in sim_manager.running_simulations:
        return jsonify(sim_manager.running_simulations[sim_id])
    else:
        return jsonify({'error': 'Simulation not found'}), 404

@app.route('/simulation/list', methods=['GET'])
def list_simulations():
    """List all simulations"""
    return jsonify({
        'simulations': sim_manager.running_simulations,
        'total': len(sim_manager.running_simulations)
    })

@app.route('/simulation/benchmark', methods=['POST'])
def run_benchmark():
    """Run performance benchmark"""
    try:
        # Run built-in benchmark
        result = subprocess.run([
            'python', '/app/run_simulation.py', '--benchmark'
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            return jsonify({
                'status': 'completed',
                'output': result.stdout,
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            return jsonify({
                'status': 'failed',
                'error': result.stderr
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/simulation/massive', methods=['POST'])
def run_massive_simulation():
    """Run massive scale simulation (for scheduled jobs)"""
    try:
        data = request.get_json() or {}
        agents = data.get('agents', 2000)
        steps = data.get('steps', 100)
        
        # Generate ID for massive simulation
        sim_id = f"massive_{int(time.time())}_{agents}"
        
        # Start simulation
        sim_manager.running_simulations[sim_id] = {
            'status': 'running',
            'type': 'massive',
            'started_at': datetime.utcnow().isoformat(),
            'parameters': {'agents': agents, 'steps': steps, 'workers': 8}
        }
        
        # Submit to thread pool
        future = executor.submit(sim_manager.run_simulation, sim_id, agents, steps, 8)
        
        return jsonify({
            'simulation_id': sim_id,
            'status': 'massive_simulation_started',
            'parameters': {'agents': agents, 'steps': steps}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)