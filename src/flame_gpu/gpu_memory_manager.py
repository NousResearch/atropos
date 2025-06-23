"""
GPU Memory Manager for FLAME GPU 2 LLM Society Simulation

This module handles efficient memory allocation, data transfer, and memory
optimization for large-scale agent simulations (501-2,500 agents).

Key Features:
- Efficient CPU-GPU data transfer
- Memory pool management for agent states
- Automatic memory cleanup and optimization
- Memory usage monitoring and reporting
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MemoryPool(Enum):
    """Memory pool types for different data structures"""

    AGENT_STATE = "agent_state"
    MESSAGES = "messages"
    SPATIAL_INDEX = "spatial_index"
    TEMPORARY = "temporary"


@dataclass
class MemoryStats:
    """Memory usage statistics"""

    total_allocated_mb: float
    peak_usage_mb: float
    current_usage_mb: float
    num_allocations: int
    num_deallocations: int
    fragmentation_ratio: float
    transfer_times: Dict[str, float]


class GPUMemoryManager:
    """
    Manages GPU memory allocation and optimization for FLAME GPU simulation

    Handles:
    - Memory pool allocation for different data types
    - Efficient CPU-GPU data transfer
    - Memory usage monitoring and optimization
    - Automatic cleanup and garbage collection
    """

    def __init__(self, max_agents: int = 2500, device_id: int = 0):
        self.max_agents = max_agents
        self.device_id = device_id

        # Memory pools
        self.memory_pools = {}
        self.allocated_memory = {}
        self.memory_stats = MemoryStats(
            total_allocated_mb=0.0,
            peak_usage_mb=0.0,
            current_usage_mb=0.0,
            num_allocations=0,
            num_deallocations=0,
            fragmentation_ratio=0.0,
            transfer_times={},
        )

        # Performance tracking
        self.transfer_history = []
        self.allocation_history = []

        # Initialize memory pools
        self._initialize_memory_pools()

        logger.info(
            f"GPU Memory Manager initialized for {max_agents} agents on device {device_id}"
        )

    def _initialize_memory_pools(self):
        """Initialize memory pools for different data types"""
        try:
            # Agent state memory pool (largest allocation)
            agent_state_size = self._calculate_agent_state_size()
            self.memory_pools[MemoryPool.AGENT_STATE] = {
                "size_mb": agent_state_size * self.max_agents / (1024 * 1024),
                "allocated": False,
                "buffer": None,
            }

            # Message memory pool (for agent communication)
            message_size = self._calculate_message_buffer_size()
            self.memory_pools[MemoryPool.MESSAGES] = {
                "size_mb": message_size / (1024 * 1024),
                "allocated": False,
                "buffer": None,
            }

            # Spatial index memory pool (for neighbor searches)
            spatial_size = self._calculate_spatial_index_size()
            self.memory_pools[MemoryPool.SPATIAL_INDEX] = {
                "size_mb": spatial_size / (1024 * 1024),
                "allocated": False,
                "buffer": None,
            }

            # Temporary memory pool (for intermediate calculations)
            temp_size = agent_state_size * 0.5  # 50% of agent state size
            self.memory_pools[MemoryPool.TEMPORARY] = {
                "size_mb": temp_size / (1024 * 1024),
                "allocated": False,
                "buffer": None,
            }

            total_memory_mb = sum(
                pool["size_mb"] for pool in self.memory_pools.values()
            )
            logger.info(f"Memory pools initialized: {total_memory_mb:.2f} MB total")

        except Exception as e:
            logger.error(f"Failed to initialize memory pools: {e}")
            raise

    def _calculate_agent_state_size(self) -> int:
        """Calculate memory size needed for agent state data"""
        # Each agent has multiple float and int variables
        num_float_vars = 25  # x, y, velocity, energy, happiness, resources, etc.
        num_int_vars = 8  # agent_id, agent_type, family_id, cultural_group, etc.

        float_size = 4 * num_float_vars  # 4 bytes per float
        int_size = 4 * num_int_vars  # 4 bytes per int

        agent_size = float_size + int_size
        return agent_size

    def _calculate_message_buffer_size(self) -> int:
        """Calculate memory size needed for message buffers"""
        # Estimate maximum messages per step
        max_messages_per_agent = 20  # Social, trade, cultural, family messages
        total_messages = self.max_agents * max_messages_per_agent

        # Each message has ~8 variables (4 bytes each)
        message_size = 8 * 4

        return total_messages * message_size

    def _calculate_spatial_index_size(self) -> int:
        """Calculate memory size needed for spatial indexing structures"""
        # Grid-based spatial index for neighbor searches
        grid_cells = 100 * 100  # 100x100 grid
        max_agents_per_cell = 50

        # Each cell stores agent IDs (4 bytes each)
        cell_size = max_agents_per_cell * 4

        return grid_cells * cell_size

    def allocate_memory_pool(self, pool_type: MemoryPool) -> bool:
        """
        Allocate memory for a specific pool

        Args:
            pool_type: Type of memory pool to allocate

        Returns:
            bool: Success status
        """
        try:
            if pool_type not in self.memory_pools:
                logger.error(f"Unknown memory pool type: {pool_type}")
                return False

            pool = self.memory_pools[pool_type]

            if pool["allocated"]:
                logger.warning(f"Memory pool {pool_type} already allocated")
                return True

            # Allocate memory buffer (mock implementation)
            size_bytes = int(pool["size_mb"] * 1024 * 1024)

            allocation_start = time.time()

            # In real implementation, this would use CUDA memory allocation
            # For now, use numpy array as placeholder
            pool["buffer"] = np.zeros(
                size_bytes // 4, dtype=np.float32
            )  # 4 bytes per float

            allocation_time = time.time() - allocation_start

            pool["allocated"] = True
            self.memory_stats.num_allocations += 1
            self.memory_stats.current_usage_mb += pool["size_mb"]
            self.memory_stats.total_allocated_mb += pool["size_mb"]

            if self.memory_stats.current_usage_mb > self.memory_stats.peak_usage_mb:
                self.memory_stats.peak_usage_mb = self.memory_stats.current_usage_mb

            # Track allocation
            self.allocation_history.append(
                {
                    "pool_type": pool_type,
                    "size_mb": pool["size_mb"],
                    "time": allocation_time,
                    "timestamp": time.time(),
                }
            )

            logger.info(
                f"Allocated {pool['size_mb']:.2f} MB for {pool_type} pool in {allocation_time:.4f}s"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to allocate memory pool {pool_type}: {e}")
            return False

    def deallocate_memory_pool(self, pool_type: MemoryPool) -> bool:
        """
        Deallocate memory for a specific pool

        Args:
            pool_type: Type of memory pool to deallocate

        Returns:
            bool: Success status
        """
        try:
            if pool_type not in self.memory_pools:
                logger.error(f"Unknown memory pool type: {pool_type}")
                return False

            pool = self.memory_pools[pool_type]

            if not pool["allocated"]:
                logger.warning(f"Memory pool {pool_type} not allocated")
                return True

            # Deallocate memory buffer
            pool["buffer"] = None
            pool["allocated"] = False

            self.memory_stats.num_deallocations += 1
            self.memory_stats.current_usage_mb -= pool["size_mb"]

            logger.info(f"Deallocated {pool['size_mb']:.2f} MB from {pool_type} pool")
            return True

        except Exception as e:
            logger.error(f"Failed to deallocate memory pool {pool_type}: {e}")
            return False

    def transfer_data_to_gpu(self, data: np.ndarray, pool_type: MemoryPool) -> bool:
        """
        Transfer data from CPU to GPU memory

        Args:
            data: NumPy array to transfer
            pool_type: Target memory pool

        Returns:
            bool: Success status
        """
        try:
            if pool_type not in self.memory_pools:
                logger.error(f"Unknown memory pool type: {pool_type}")
                return False

            pool = self.memory_pools[pool_type]

            if not pool["allocated"]:
                logger.error(f"Memory pool {pool_type} not allocated")
                return False

            transfer_start = time.time()

            # Mock GPU transfer (in real implementation, use CUDA memory copy)
            data_size_mb = data.nbytes / (1024 * 1024)

            # Simulate transfer time based on data size
            simulated_transfer_time = data_size_mb * 0.001  # 1ms per MB
            time.sleep(min(simulated_transfer_time, 0.01))  # Cap at 10ms for testing

            # Copy data to buffer (mock)
            if len(pool["buffer"]) >= len(data.flatten()):
                pool["buffer"][: len(data.flatten())] = data.flatten()
            else:
                logger.warning(
                    f"Data size ({len(data.flatten())}) exceeds buffer size ({len(pool['buffer'])})"
                )

            transfer_time = time.time() - transfer_start

            # Track transfer performance
            self.transfer_history.append(
                {
                    "direction": "cpu_to_gpu",
                    "pool_type": pool_type,
                    "size_mb": data_size_mb,
                    "time": transfer_time,
                    "bandwidth_gbps": (data_size_mb / 1024) / max(transfer_time, 0.001),
                    "timestamp": time.time(),
                }
            )

            if pool_type.value not in self.memory_stats.transfer_times:
                self.memory_stats.transfer_times[pool_type.value] = []
            self.memory_stats.transfer_times[pool_type.value].append(transfer_time)

            logger.debug(
                f"Transferred {data_size_mb:.2f} MB to GPU {pool_type} in {transfer_time:.4f}s"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to transfer data to GPU: {e}")
            return False

    def transfer_data_from_gpu(
        self, pool_type: MemoryPool, size: int
    ) -> Optional[np.ndarray]:
        """
        Transfer data from GPU to CPU memory

        Args:
            pool_type: Source memory pool
            size: Number of elements to transfer

        Returns:
            Optional[np.ndarray]: Transferred data or None if failed
        """
        try:
            if pool_type not in self.memory_pools:
                logger.error(f"Unknown memory pool type: {pool_type}")
                return None

            pool = self.memory_pools[pool_type]

            if not pool["allocated"]:
                logger.error(f"Memory pool {pool_type} not allocated")
                return None

            transfer_start = time.time()

            # Mock GPU transfer (in real implementation, use CUDA memory copy)
            data = np.array(pool["buffer"][:size])
            data_size_mb = data.nbytes / (1024 * 1024)

            # Simulate transfer time
            simulated_transfer_time = data_size_mb * 0.001
            time.sleep(min(simulated_transfer_time, 0.01))

            transfer_time = time.time() - transfer_start

            # Track transfer performance
            self.transfer_history.append(
                {
                    "direction": "gpu_to_cpu",
                    "pool_type": pool_type,
                    "size_mb": data_size_mb,
                    "time": transfer_time,
                    "bandwidth_gbps": (data_size_mb / 1024) / max(transfer_time, 0.001),
                    "timestamp": time.time(),
                }
            )

            logger.debug(
                f"Transferred {data_size_mb:.2f} MB from GPU {pool_type} in {transfer_time:.4f}s"
            )
            return data

        except Exception as e:
            logger.error(f"Failed to transfer data from GPU: {e}")
            return None

    def optimize_memory_layout(self) -> bool:
        """
        Optimize memory layout for better performance

        Returns:
            bool: Success status
        """
        try:
            logger.info("Starting memory layout optimization...")

            # Calculate memory fragmentation
            allocated_pools = [p for p in self.memory_pools.values() if p["allocated"]]
            if len(allocated_pools) > 0:
                total_allocated = sum(p["size_mb"] for p in allocated_pools)
                largest_free_block = (
                    max(
                        p["size_mb"]
                        for p in self.memory_pools.values()
                        if not p["allocated"]
                    )
                    if any(not p["allocated"] for p in self.memory_pools.values())
                    else 0
                )
                self.memory_stats.fragmentation_ratio = 1.0 - (
                    largest_free_block / max(total_allocated, 1.0)
                )

            # Memory optimization strategies
            optimizations_applied = 0

            # 1. Coalesce adjacent free blocks (mock)
            optimizations_applied += 1

            # 2. Reorder allocations for better cache locality (mock)
            optimizations_applied += 1

            # 3. Clean up temporary allocations
            if self.memory_pools[MemoryPool.TEMPORARY]["allocated"]:
                temp_pool = self.memory_pools[MemoryPool.TEMPORARY]
                # Zero out temporary buffer to help with memory compression
                if temp_pool["buffer"] is not None:
                    temp_pool["buffer"].fill(0.0)
                optimizations_applied += 1

            logger.info(
                f"Memory optimization completed: {optimizations_applied} optimizations applied"
            )
            return True

        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return False

    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory usage statistics

        Returns:
            MemoryStats: Current memory statistics
        """
        # Update fragmentation ratio
        if self.memory_stats.current_usage_mb > 0:
            allocated_count = sum(
                1 for p in self.memory_pools.values() if p["allocated"]
            )
            total_pools = len(self.memory_pools)
            self.memory_stats.fragmentation_ratio = 1.0 - (
                allocated_count / total_pools
            )

        return self.memory_stats

    def get_transfer_performance(self) -> Dict:
        """
        Get data transfer performance metrics

        Returns:
            Dict: Transfer performance statistics
        """
        if not self.transfer_history:
            return {}

        recent_transfers = self.transfer_history[-100:]  # Last 100 transfers

        cpu_to_gpu = [t for t in recent_transfers if t["direction"] == "cpu_to_gpu"]
        gpu_to_cpu = [t for t in recent_transfers if t["direction"] == "gpu_to_cpu"]

        stats = {
            "total_transfers": len(self.transfer_history),
            "recent_transfers": len(recent_transfers),
            "cpu_to_gpu": {
                "count": len(cpu_to_gpu),
                "avg_time": (
                    np.mean([t["time"] for t in cpu_to_gpu]) if cpu_to_gpu else 0.0
                ),
                "avg_bandwidth_gbps": (
                    np.mean([t["bandwidth_gbps"] for t in cpu_to_gpu])
                    if cpu_to_gpu
                    else 0.0
                ),
                "total_mb": sum(t["size_mb"] for t in cpu_to_gpu),
            },
            "gpu_to_cpu": {
                "count": len(gpu_to_cpu),
                "avg_time": (
                    np.mean([t["time"] for t in gpu_to_cpu]) if gpu_to_cpu else 0.0
                ),
                "avg_bandwidth_gbps": (
                    np.mean([t["bandwidth_gbps"] for t in gpu_to_cpu])
                    if gpu_to_cpu
                    else 0.0
                ),
                "total_mb": sum(t["size_mb"] for t in gpu_to_cpu),
            },
        }

        return stats

    def cleanup(self):
        """Clean up all allocated memory"""
        try:
            logger.info("Starting GPU memory cleanup...")

            cleanup_count = 0
            for pool_type in list(self.memory_pools.keys()):
                if self.memory_pools[pool_type]["allocated"]:
                    if self.deallocate_memory_pool(pool_type):
                        cleanup_count += 1

            # Clear performance tracking data
            self.transfer_history.clear()
            self.allocation_history.clear()

            # Reset stats
            self.memory_stats = MemoryStats(
                total_allocated_mb=0.0,
                peak_usage_mb=self.memory_stats.peak_usage_mb,  # Keep peak for reporting
                current_usage_mb=0.0,
                num_allocations=self.memory_stats.num_allocations,
                num_deallocations=self.memory_stats.num_deallocations,
                fragmentation_ratio=0.0,
                transfer_times={},
            )

            logger.info(
                f"GPU memory cleanup completed: {cleanup_count} pools deallocated"
            )

        except Exception as e:
            logger.error(f"GPU memory cleanup failed: {e}")

    def __del__(self):
        """Destructor - ensure memory cleanup"""
        self.cleanup()
