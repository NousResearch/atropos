"""
Performance Profiler for FLAME GPU 2 LLM Society Simulation

This module provides comprehensive performance monitoring and profiling
capabilities for the GPU-accelerated simulation.

Key Features:
- Kernel execution time profiling
- Memory bandwidth monitoring
- Agent throughput analysis
- Performance bottleneck detection
- Detailed performance reports
"""

import time
import logging
import statistics
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

@dataclass
class KernelProfile:
    """Performance profile for a GPU kernel"""
    name: str
    execution_times: List[float] = field(default_factory=list)
    memory_transfers: List[float] = field(default_factory=list)
    throughput_agents_per_sec: List[float] = field(default_factory=list)
    total_executions: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    std_time: float = 0.0

@dataclass
class SystemProfile:
    """Overall system performance profile"""
    total_simulation_time: float = 0.0
    total_steps: int = 0
    avg_step_time: float = 0.0
    peak_memory_usage_mb: float = 0.0
    avg_memory_usage_mb: float = 0.0
    total_memory_transfers_mb: float = 0.0
    avg_agents_per_second: float = 0.0
    cpu_gpu_sync_overhead: float = 0.0
    kernel_profiles: Dict[str, KernelProfile] = field(default_factory=dict)

class FlameGPUProfiler:
    """
    Comprehensive performance profiler for FLAME GPU simulation
    
    Monitors and analyzes:
    - Individual kernel performance
    - Memory transfer efficiency
    - Overall system throughput
    - Performance bottlenecks
    - Temporal performance trends
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        
        # Performance data storage
        self.kernel_profiles = {}
        self.system_profile = SystemProfile()
        
        # Real-time monitoring
        self.current_step_start = None
        self.current_kernel_start = None
        self.current_kernel_name = None
        
        # Historical data
        self.step_times = deque(maxlen=max_history)
        self.memory_usage_history = deque(maxlen=max_history)
        self.throughput_history = deque(maxlen=max_history)
        
        # Performance thresholds for alerts
        self.performance_thresholds = {
            'max_step_time': 1.0,        # seconds
            'max_kernel_time': 0.1,      # seconds
            'min_throughput': 1000.0,    # agents/second
            'max_memory_usage': 8192.0   # MB
        }
        
        # Performance alerts
        self.performance_alerts = []
        
        logger.info("FLAME GPU Performance Profiler initialized")
    
    def start_step_profiling(self, step_number: int, num_agents: int):
        """
        Start profiling a simulation step
        
        Args:
            step_number: Current step number
            num_agents: Number of agents in simulation
        """
        self.current_step_start = time.time()
        self.current_step_number = step_number
        self.current_num_agents = num_agents
        
        logger.debug(f"Started profiling step {step_number} with {num_agents} agents")
    
    def end_step_profiling(self) -> Dict:
        """
        End profiling a simulation step and calculate metrics
        
        Returns:
            Dict: Step performance metrics
        """
        if self.current_step_start is None:
            logger.warning("Step profiling not started")
            return {}
        
        step_time = time.time() - self.current_step_start
        
        # Update system profile
        self.system_profile.total_simulation_time += step_time
        self.system_profile.total_steps += 1
        self.system_profile.avg_step_time = (
            self.system_profile.total_simulation_time / self.system_profile.total_steps
        )
        
        # Calculate throughput
        throughput = self.current_num_agents / step_time if step_time > 0 else 0
        
        # Store historical data
        self.step_times.append(step_time)
        self.throughput_history.append(throughput)
        
        # Check performance thresholds
        self._check_performance_alerts(step_time, throughput)
        
        step_metrics = {
            'step_number': self.current_step_number,
            'step_time': step_time,
            'agents_per_second': throughput,
            'num_agents': self.current_num_agents
        }
        
        self.current_step_start = None
        
        logger.debug(f"Step {self.current_step_number} completed in {step_time:.4f}s "
                    f"({throughput:.0f} agents/sec)")
        
        return step_metrics
    
    def start_kernel_profiling(self, kernel_name: str):
        """
        Start profiling a GPU kernel
        
        Args:
            kernel_name: Name of the kernel being executed
        """
        self.current_kernel_start = time.time()
        self.current_kernel_name = kernel_name
        
        # Initialize kernel profile if first time
        if kernel_name not in self.kernel_profiles:
            self.kernel_profiles[kernel_name] = KernelProfile(name=kernel_name)
        
        logger.debug(f"Started profiling kernel: {kernel_name}")
    
    def end_kernel_profiling(self) -> float:
        """
        End profiling a GPU kernel
        
        Returns:
            float: Kernel execution time
        """
        if self.current_kernel_start is None or self.current_kernel_name is None:
            logger.warning("Kernel profiling not started")
            return 0.0
        
        kernel_time = time.time() - self.current_kernel_start
        kernel_name = self.current_kernel_name
        
        # Update kernel profile
        profile = self.kernel_profiles[kernel_name]
        profile.execution_times.append(kernel_time)
        profile.total_executions += 1
        profile.total_time += kernel_time
        profile.min_time = min(profile.min_time, kernel_time)
        profile.max_time = max(profile.max_time, kernel_time)
        
        # Calculate statistics
        if profile.execution_times:
            profile.avg_time = statistics.mean(profile.execution_times)
            if len(profile.execution_times) > 1:
                profile.std_time = statistics.stdev(profile.execution_times)
        
        # Calculate throughput if we have agent count
        if hasattr(self, 'current_num_agents') and kernel_time > 0:
            throughput = self.current_num_agents / kernel_time
            profile.throughput_agents_per_sec.append(throughput)
        
        # Check kernel performance threshold
        if kernel_time > self.performance_thresholds['max_kernel_time']:
            self._add_performance_alert(
                f"Kernel {kernel_name} exceeded time threshold: {kernel_time:.4f}s"
            )
        
        self.current_kernel_start = None
        self.current_kernel_name = None
        
        logger.debug(f"Kernel {kernel_name} completed in {kernel_time:.4f}s")
        
        return kernel_time
    
    def record_memory_usage(self, memory_usage_mb: float):
        """
        Record current memory usage
        
        Args:
            memory_usage_mb: Memory usage in megabytes
        """
        self.memory_usage_history.append(memory_usage_mb)
        
        # Update system profile
        self.system_profile.peak_memory_usage_mb = max(
            self.system_profile.peak_memory_usage_mb, memory_usage_mb
        )
        
        if self.memory_usage_history:
            self.system_profile.avg_memory_usage_mb = statistics.mean(self.memory_usage_history)
        
        # Check memory threshold
        if memory_usage_mb > self.performance_thresholds['max_memory_usage']:
            self._add_performance_alert(
                f"Memory usage exceeded threshold: {memory_usage_mb:.2f} MB"
            )
    
    def record_memory_transfer(self, kernel_name: str, transfer_size_mb: float, transfer_time: float):
        """
        Record memory transfer performance
        
        Args:
            kernel_name: Name of associated kernel
            transfer_size_mb: Size of transfer in megabytes
            transfer_time: Transfer time in seconds
        """
        # Update system profile
        self.system_profile.total_memory_transfers_mb += transfer_size_mb
        
        # Update kernel profile
        if kernel_name in self.kernel_profiles:
            profile = self.kernel_profiles[kernel_name]
            profile.memory_transfers.append(transfer_time)
        
        # Calculate bandwidth
        bandwidth_gbps = (transfer_size_mb / 1024) / max(transfer_time, 0.001)
        
        logger.debug(f"Memory transfer for {kernel_name}: {transfer_size_mb:.2f} MB "
                    f"in {transfer_time:.4f}s ({bandwidth_gbps:.2f} GB/s)")
    
    def get_kernel_performance(self, kernel_name: str) -> Optional[KernelProfile]:
        """
        Get performance profile for a specific kernel
        
        Args:
            kernel_name: Name of the kernel
            
        Returns:
            Optional[KernelProfile]: Kernel performance profile
        """
        return self.kernel_profiles.get(kernel_name)
    
    def get_system_performance(self) -> SystemProfile:
        """
        Get overall system performance profile
        
        Returns:
            SystemProfile: System performance metrics
        """
        # Update system profile with current kernel data
        self.system_profile.kernel_profiles = self.kernel_profiles.copy()
        
        # Calculate average throughput
        if self.throughput_history:
            self.system_profile.avg_agents_per_second = statistics.mean(self.throughput_history)
        
        return self.system_profile
    
    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary
        
        Returns:
            Dict: Detailed performance summary
        """
        summary = {
            'system_overview': {
                'total_steps': self.system_profile.total_steps,
                'total_simulation_time': self.system_profile.total_simulation_time,
                'avg_step_time': self.system_profile.avg_step_time,
                'avg_agents_per_second': self.system_profile.avg_agents_per_second,
                'peak_memory_usage_mb': self.system_profile.peak_memory_usage_mb,
                'avg_memory_usage_mb': self.system_profile.avg_memory_usage_mb
            },
            'kernel_performance': {},
            'recent_performance': {},
            'performance_alerts': self.performance_alerts[-10:],  # Last 10 alerts
            'bottlenecks': self._identify_bottlenecks()
        }
        
        # Kernel performance details
        for kernel_name, profile in self.kernel_profiles.items():
            summary['kernel_performance'][kernel_name] = {
                'total_executions': profile.total_executions,
                'total_time': profile.total_time,
                'avg_time': profile.avg_time,
                'min_time': profile.min_time,
                'max_time': profile.max_time,
                'std_time': profile.std_time,
                'avg_throughput': statistics.mean(profile.throughput_agents_per_sec) if profile.throughput_agents_per_sec else 0.0,
                'avg_memory_transfer_time': statistics.mean(profile.memory_transfers) if profile.memory_transfers else 0.0
            }
        
        # Recent performance trends
        if len(self.step_times) >= 10:
            recent_steps = list(self.step_times)[-10:]
            recent_throughput = list(self.throughput_history)[-10:]
            
            summary['recent_performance'] = {
                'avg_step_time': statistics.mean(recent_steps),
                'step_time_trend': self._calculate_trend(recent_steps),
                'avg_throughput': statistics.mean(recent_throughput),
                'throughput_trend': self._calculate_trend(recent_throughput),
                'step_time_variance': statistics.variance(recent_steps) if len(recent_steps) > 1 else 0.0
            }
        
        return summary
    
    def _check_performance_alerts(self, step_time: float, throughput: float):
        """
        Check for performance threshold violations
        
        Args:
            step_time: Current step execution time
            throughput: Current throughput (agents/second)
        """
        if step_time > self.performance_thresholds['max_step_time']:
            self._add_performance_alert(
                f"Step time exceeded threshold: {step_time:.4f}s"
            )
        
        if throughput < self.performance_thresholds['min_throughput']:
            self._add_performance_alert(
                f"Throughput below threshold: {throughput:.0f} agents/sec"
            )
    
    def _add_performance_alert(self, message: str):
        """
        Add a performance alert
        
        Args:
            message: Alert message
        """
        alert = {
            'timestamp': time.time(),
            'message': message,
            'step': getattr(self, 'current_step_number', 'unknown')
        }
        
        self.performance_alerts.append(alert)
        logger.warning(f"Performance Alert: {message}")
    
    def _identify_bottlenecks(self) -> List[Dict]:
        """
        Identify performance bottlenecks
        
        Returns:
            List[Dict]: List of identified bottlenecks
        """
        bottlenecks = []
        
        # Kernel time bottlenecks
        total_kernel_time = sum(profile.total_time for profile in self.kernel_profiles.values())
        
        for kernel_name, profile in self.kernel_profiles.items():
            if total_kernel_time > 0:
                time_percentage = (profile.total_time / total_kernel_time) * 100
                
                if time_percentage > 30:  # Kernel takes >30% of total time
                    bottlenecks.append({
                        'type': 'kernel_time',
                        'kernel': kernel_name,
                        'percentage': time_percentage,
                        'avg_time': profile.avg_time,
                        'severity': 'high' if time_percentage > 50 else 'medium'
                    })
        
        # Memory usage bottlenecks
        if self.system_profile.avg_memory_usage_mb > self.performance_thresholds['max_memory_usage'] * 0.8:
            bottlenecks.append({
                'type': 'memory_usage',
                'avg_usage_mb': self.system_profile.avg_memory_usage_mb,
                'peak_usage_mb': self.system_profile.peak_memory_usage_mb,
                'severity': 'high' if self.system_profile.avg_memory_usage_mb > self.performance_thresholds['max_memory_usage'] else 'medium'
            })
        
        # Throughput bottlenecks
        if self.system_profile.avg_agents_per_second < self.performance_thresholds['min_throughput']:
            bottlenecks.append({
                'type': 'throughput',
                'avg_throughput': self.system_profile.avg_agents_per_second,
                'threshold': self.performance_thresholds['min_throughput'],
                'severity': 'high'
            })
        
        return bottlenecks
    
    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction for a series of values
        
        Args:
            values: List of numeric values
            
        Returns:
            str: Trend direction ('improving', 'degrading', 'stable')
        """
        if len(values) < 3:
            return 'stable'
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 'stable'
        
        slope = numerator / denominator
        
        # Determine trend based on slope and magnitude
        if abs(slope) < 0.01:  # Small changes are considered stable
            return 'stable'
        elif slope > 0:
            return 'improving' if 'throughput' in str(values) else 'degrading'  # Higher throughput is better, higher time is worse
        else:
            return 'degrading' if 'throughput' in str(values) else 'improving'
    
    def export_performance_report(self, filename: str = None) -> str:
        """
        Export detailed performance report to file
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            str: Filename of exported report
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"flame_gpu_performance_report_{timestamp}.json"
        
        try:
            performance_data = {
                'metadata': {
                    'timestamp': time.time(),
                    'profiler_version': '1.0',
                    'max_history': self.max_history
                },
                'system_profile': {
                    'total_simulation_time': self.system_profile.total_simulation_time,
                    'total_steps': self.system_profile.total_steps,
                    'avg_step_time': self.system_profile.avg_step_time,
                    'peak_memory_usage_mb': self.system_profile.peak_memory_usage_mb,
                    'avg_memory_usage_mb': self.system_profile.avg_memory_usage_mb,
                    'total_memory_transfers_mb': self.system_profile.total_memory_transfers_mb,
                    'avg_agents_per_second': self.system_profile.avg_agents_per_second
                },
                'kernel_profiles': {},
                'performance_summary': self.get_performance_summary(),
                'historical_data': {
                    'step_times': list(self.step_times),
                    'throughput_history': list(self.throughput_history),
                    'memory_usage_history': list(self.memory_usage_history)
                }
            }
            
            # Export kernel profiles
            for kernel_name, profile in self.kernel_profiles.items():
                performance_data['kernel_profiles'][kernel_name] = {
                    'name': profile.name,
                    'total_executions': profile.total_executions,
                    'total_time': profile.total_time,
                    'avg_time': profile.avg_time,
                    'min_time': profile.min_time,
                    'max_time': profile.max_time,
                    'std_time': profile.std_time,
                    'execution_times': profile.execution_times[-100:],  # Last 100 executions
                    'throughput_agents_per_sec': profile.throughput_agents_per_sec[-100:],
                    'memory_transfers': profile.memory_transfers[-100:]
                }
            
            with open(filename, 'w') as f:
                json.dump(performance_data, f, indent=2, default=str)
            
            logger.info(f"Performance report exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export performance report: {e}")
            raise
    
    def reset_profiling(self):
        """Reset all profiling data"""
        self.kernel_profiles.clear()
        self.system_profile = SystemProfile()
        self.step_times.clear()
        self.memory_usage_history.clear()
        self.throughput_history.clear()
        self.performance_alerts.clear()
        
        logger.info("Performance profiling data reset")
    
    def set_performance_thresholds(self, **thresholds):
        """
        Update performance thresholds
        
        Args:
            **thresholds: Threshold values to update
        """
        for key, value in thresholds.items():
            if key in self.performance_thresholds:
                self.performance_thresholds[key] = value
                logger.info(f"Updated performance threshold {key}: {value}")
            else:
                logger.warning(f"Unknown performance threshold: {key}")
    
    def get_real_time_stats(self) -> Dict:
        """
        Get real-time performance statistics
        
        Returns:
            Dict: Current performance metrics
        """
        recent_window = 10
        
        stats = {
            'current_step': getattr(self, 'current_step_number', 'N/A'),
            'current_kernel': self.current_kernel_name or 'None',
            'total_steps': self.system_profile.total_steps,
            'total_simulation_time': self.system_profile.total_simulation_time
        }
        
        # Recent performance
        if len(self.step_times) > 0:
            recent_steps = list(self.step_times)[-recent_window:]
            recent_throughput = list(self.throughput_history)[-recent_window:]
            
            stats.update({
                'recent_avg_step_time': statistics.mean(recent_steps),
                'recent_avg_throughput': statistics.mean(recent_throughput),
                'last_step_time': self.step_times[-1],
                'last_throughput': self.throughput_history[-1] if self.throughput_history else 0
            })
        
        # Current memory usage
        if self.memory_usage_history:
            stats['current_memory_usage_mb'] = self.memory_usage_history[-1]
        
        # Active alerts
        recent_alerts = [alert for alert in self.performance_alerts 
                        if time.time() - alert['timestamp'] < 60]  # Last minute
        stats['active_alerts'] = len(recent_alerts)
        
        return stats 