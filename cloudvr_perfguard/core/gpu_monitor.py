"""
GPU Monitor - Collects GPU performance metrics during VR testing
Monitors GPU utilization, VRAM usage, temperature, and other metrics
"""

import asyncio
import time
import subprocess
import json
import psutil
from typing import Dict, List, Any, Optional
import numpy as np

class GPUMonitor:
    """
    Monitors GPU performance metrics during VR testing
    Uses nvidia-smi and other tools to collect real-time data
    """
    
    def __init__(self):
        self.monitoring = False
        self.metrics_data = []
        
    async def initialize(self):
        """Initialize GPU monitoring"""
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(["nvidia-smi", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("WARNING: nvidia-smi not available, GPU monitoring will be limited")
            
            print("INFO: GPU monitor initialized")
            
        except Exception as e:
            print(f"WARNING: GPU monitor initialization failed: {e}")
    
    async def monitor_performance(self, container_id: str, duration: int) -> Dict[str, Any]:
        """
        Monitor GPU performance for specified duration
        Returns collected metrics data
        """
        
        print(f"📊 Starting GPU monitoring for {duration}s")
        
        self.monitoring = True
        self.metrics_data = []
        
        start_time = time.time()
        end_time = start_time + duration
        
        # Collect metrics every second
        while time.time() < end_time and self.monitoring:
            try:
                metrics = await self._collect_current_metrics(container_id)
                metrics["timestamp"] = time.time() - start_time
                self.metrics_data.append(metrics)
                
                await asyncio.sleep(1.0)  # 1 second interval
                
            except Exception as e:
                print(f"WARNING: Error collecting metrics: {e}")
                await asyncio.sleep(1.0)
        
        self.monitoring = False
        
        # Process and return aggregated metrics
        return self._process_metrics_data()
    
    async def _collect_current_metrics(self, container_id: str) -> Dict[str, Any]:
        """Collect current GPU and system metrics"""
        
        metrics = {
            "gpu_utilization": [],
            "vram_usage_mb": [],
            "vram_total_mb": [],
            "gpu_temperature": [],
            "cpu_utilization": 0,
            "system_ram_usage_mb": 0,
            "system_ram_total_mb": 0
        }
        
        # Collect NVIDIA GPU metrics
        try:
            nvidia_metrics = await self._get_nvidia_metrics()
            if nvidia_metrics:
                metrics.update(nvidia_metrics)
        except Exception as e:
            print(f"WARNING: Failed to collect NVIDIA metrics: {e}")
        
        # Collect system metrics
        try:
            system_metrics = await self._get_system_metrics()
            metrics.update(system_metrics)
        except Exception as e:
            print(f"WARNING: Failed to collect system metrics: {e}")
        
        return metrics
    
    async def _get_nvidia_metrics(self) -> Optional[Dict[str, Any]]:
        """Get NVIDIA GPU metrics using nvidia-smi"""
        
        try:
            # Query nvidia-smi for GPU metrics
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return None
            
            # Parse output
            lines = result.stdout.strip().split('\n')
            gpu_metrics = {
                "gpu_utilization": [],
                "vram_usage_mb": [],
                "vram_total_mb": [],
                "gpu_temperature": []
            }
            
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 4:
                        gpu_util = float(parts[0].strip())
                        vram_used = float(parts[1].strip())
                        vram_total = float(parts[2].strip())
                        gpu_temp = float(parts[3].strip())
                        
                        gpu_metrics["gpu_utilization"].append(gpu_util)
                        gpu_metrics["vram_usage_mb"].append(vram_used)
                        gpu_metrics["vram_total_mb"].append(vram_total)
                        gpu_metrics["gpu_temperature"].append(gpu_temp)
            
            return gpu_metrics
            
        except Exception as e:
            print(f"ERROR getting NVIDIA metrics: {e}")
            return None
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system CPU and RAM metrics"""
        
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            return {
                "cpu_utilization": cpu_percent,
                "system_ram_usage_mb": memory.used / (1024 * 1024),
                "system_ram_total_mb": memory.total / (1024 * 1024),
                "system_ram_percent": memory.percent
            }
            
        except Exception as e:
            print(f"ERROR getting system metrics: {e}")
            return {
                "cpu_utilization": 0,
                "system_ram_usage_mb": 0,
                "system_ram_total_mb": 0,
                "system_ram_percent": 0
            }
    
    def _process_metrics_data(self) -> Dict[str, Any]:
        """Process collected metrics data into summary statistics"""
        
        if not self.metrics_data:
            return {"error": "No metrics data collected"}
        
        # Extract time series data
        gpu_utilization = []
        vram_usage = []
        gpu_temperature = []
        cpu_utilization = []
        
        for metrics in self.metrics_data:
            # GPU metrics (may have multiple GPUs)
            if metrics.get("gpu_utilization"):
                gpu_utilization.extend(metrics["gpu_utilization"])
            
            if metrics.get("vram_usage_mb"):
                vram_usage.extend(metrics["vram_usage_mb"])
            
            if metrics.get("gpu_temperature"):
                gpu_temperature.extend(metrics["gpu_temperature"])
            
            # System metrics
            cpu_utilization.append(metrics.get("cpu_utilization", 0))
        
        # Calculate summary statistics
        processed = {
            "collection_duration": len(self.metrics_data),
            "sample_count": len(self.metrics_data),
            "raw_data": self.metrics_data  # Include raw data for detailed analysis
        }
        
        # GPU utilization stats
        if gpu_utilization:
            processed["gpu_utilization"] = {
                "mean": np.mean(gpu_utilization),
                "min": np.min(gpu_utilization),
                "max": np.max(gpu_utilization),
                "std": np.std(gpu_utilization),
                "p95": np.percentile(gpu_utilization, 95),
                "p99": np.percentile(gpu_utilization, 99),
                "time_series": gpu_utilization
            }
        
        # VRAM usage stats
        if vram_usage:
            processed["vram_usage_mb"] = {
                "mean": np.mean(vram_usage),
                "min": np.min(vram_usage),
                "max": np.max(vram_usage),
                "std": np.std(vram_usage),
                "p95": np.percentile(vram_usage, 95),
                "p99": np.percentile(vram_usage, 99),
                "time_series": vram_usage
            }
        
        # GPU temperature stats
        if gpu_temperature:
            processed["gpu_temperature"] = {
                "mean": np.mean(gpu_temperature),
                "min": np.min(gpu_temperature),
                "max": np.max(gpu_temperature),
                "std": np.std(gpu_temperature),
                "time_series": gpu_temperature
            }
        
        # CPU utilization stats
        if cpu_utilization:
            processed["cpu_utilization"] = {
                "mean": np.mean(cpu_utilization),
                "min": np.min(cpu_utilization),
                "max": np.max(cpu_utilization),
                "std": np.std(cpu_utilization),
                "p95": np.percentile(cpu_utilization, 95),
                "time_series": cpu_utilization
            }
        
        return processed
    
    async def get_gpu_info(self) -> Dict[str, Any]:
        """Get static GPU information"""
        
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return {"error": "Failed to get GPU info"}
            
            gpus = []
            lines = result.stdout.strip().split('\n')
            
            for i, line in enumerate(lines):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpus.append({
                            "gpu_id": i,
                            "name": parts[0],
                            "driver_version": parts[1],
                            "memory_total_mb": parts[2].replace(' MiB', ''),
                            "compute_capability": parts[3]
                        })
            
            return {"gpus": gpus, "gpu_count": len(gpus)}
            
        except Exception as e:
            print(f"ERROR getting GPU info: {e}")
            return {"error": str(e)}
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
    
    async def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        print("INFO: GPU monitor cleanup complete") 