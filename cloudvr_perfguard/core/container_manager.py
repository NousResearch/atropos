"""
Container Manager - Orchestrates VR application testing in containerized environments
Manages Docker containers with GPU access for VR performance testing
"""

import asyncio
import json
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

import docker


class ContainerManager:
    """
    Manages containerized VR testing environments
    Handles container lifecycle, GPU allocation, and VR runtime setup
    """

    def __init__(self):
        self.docker_client = None
        self.active_containers = {}

        # Container configurations for different platforms
        self.container_configs = {
            "windows": {
                "base_image": "mcr.microsoft.com/windows/servercore:ltsc2022",
                "vr_runtimes": {
                    "steamvr": {
                        "setup_commands": [
                            "# Install SteamVR runtime",
                            "# This would be implemented with actual SteamVR installation",
                        ]
                    },
                    "oculus": {
                        "setup_commands": [
                            "# Install Oculus runtime",
                            "# This would be implemented with actual Oculus installation",
                        ]
                    },
                },
            },
            "linux": {
                "base_image": "nvidia/opengl:1.2-glvnd-runtime-ubuntu20.04",
                "vr_runtimes": {
                    "openxr": {
                        "setup_commands": [
                            "apt-get update",
                            "apt-get install -y openxr-dev",
                            "# Additional OpenXR setup",
                        ]
                    },
                    "steamvr": {
                        "setup_commands": [
                            "# Install SteamVR for Linux",
                            "# This would be implemented with actual SteamVR installation",
                        ]
                    },
                },
            },
        }

    async def initialize(self):
        """Initialize container manager"""
        try:
            # Initialize Docker client
            self.docker_client = docker.from_env()

            # Test Docker connection
            self.docker_client.ping()

            print("INFO: Container manager initialized successfully")

        except Exception as e:
            print(f"ERROR: Failed to initialize container manager: {e}")
            raise

    async def create_container(self, config: Dict[str, Any]) -> str:
        """
        Create and start a container for VR testing
        Returns container ID
        """

        container_id = str(uuid.uuid4())[:8]

        try:
            print(
                f"🐳 Creating container {container_id} for {config['platform']} + {config['vr_runtime']}"
            )

            # Get platform configuration
            platform = config.get("platform", "linux")
            vr_runtime = config.get("vr_runtime", "openxr")
            gpu_type = config.get("gpu_type", "T4")

            if platform not in self.container_configs:
                raise ValueError(f"Unsupported platform: {platform}")

            platform_config = self.container_configs[platform]

            # Prepare container environment
            environment = {
                "VR_RUNTIME": vr_runtime,
                "GPU_TYPE": gpu_type,
                "DISPLAY": ":0",  # For Linux X11
                "NVIDIA_VISIBLE_DEVICES": "all",
                "NVIDIA_DRIVER_CAPABILITIES": "all",
            }

            # Container runtime configuration
            runtime_config = {
                "image": platform_config["base_image"],
                "name": f"cloudvr-test-{container_id}",
                "environment": environment,
                "detach": True,
                "remove": False,  # Keep container for debugging
                "network_mode": "bridge",
                "shm_size": "2g",  # Shared memory for VR apps
                # GPU access
                "device_requests": [
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ],
                # Volume mounts
                "volumes": {
                    "/tmp/.X11-unix": {"bind": "/tmp/.X11-unix", "mode": "rw"},
                    # Build path will be mounted here
                },
                # Resource limits
                "mem_limit": "8g",
                "cpu_count": 4,
            }

            # Add build path mount if provided
            if "build_path" in config:
                runtime_config["volumes"][config["build_path"]] = {
                    "bind": "/app/build",
                    "mode": "ro",
                }

            # Create and start container
            container = self.docker_client.containers.run(**runtime_config)

            # Store container reference
            self.active_containers[container_id] = {
                "container": container,
                "config": config,
                "created_at": time.time(),
            }

            print(f"✅ Container {container_id} created successfully")
            return container_id

        except Exception as e:
            print(f"❌ Failed to create container {container_id}: {e}")
            raise

    async def wait_for_ready(self, container_id: str, timeout: int = 60) -> bool:
        """Wait for container to be ready for testing"""

        if container_id not in self.active_containers:
            raise ValueError(f"Container {container_id} not found")

        container_info = self.active_containers[container_id]
        container = container_info["container"]

        print(f"⏳ Waiting for container {container_id} to be ready...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check container status
                container.reload()

                if container.status == "running":
                    # Test if VR runtime is accessible
                    if await self._test_vr_runtime(container_id):
                        print(f"✅ Container {container_id} is ready")
                        return True

                await asyncio.sleep(2)

            except Exception as e:
                print(f"WARNING: Error checking container readiness: {e}")
                await asyncio.sleep(2)

        print(f"❌ Container {container_id} failed to become ready within {timeout}s")
        return False

    async def _test_vr_runtime(self, container_id: str) -> bool:
        """Test if VR runtime is accessible in container"""

        try:
            container_info = self.active_containers[container_id]
            container = container_info["container"]
            vr_runtime = container_info["config"].get("vr_runtime", "openxr")

            # Simple test commands for different VR runtimes
            test_commands = {
                "openxr": "ls /usr/lib/x86_64-linux-gnu/openxr/",
                "steamvr": "echo 'SteamVR test'",  # Placeholder
                "oculus": "echo 'Oculus test'",  # Placeholder
            }

            test_cmd = test_commands.get(vr_runtime, "echo 'Unknown VR runtime'")

            # Execute test command
            result = container.exec_run(test_cmd, timeout=10)

            return result.exit_code == 0

        except Exception as e:
            print(f"WARNING: VR runtime test failed: {e}")
            return False

    async def run_vr_app(
        self, container_id: str, scene_name: str, duration: int
    ) -> Dict[str, Any]:
        """
        Run VR application in container and collect performance data
        Returns application metrics
        """

        if container_id not in self.active_containers:
            raise ValueError(f"Container {container_id} not found")

        container_info = self.active_containers[container_id]
        container = container_info["container"]

        print(f"🎮 Running VR app in container {container_id} for {duration}s")

        try:
            # Prepare VR app execution
            app_metrics = await self._execute_vr_app(container, scene_name, duration)

            return app_metrics

        except Exception as e:
            print(f"❌ Failed to run VR app in container {container_id}: {e}")
            raise

    async def _execute_vr_app(
        self, container, scene_name: str, duration: int
    ) -> Dict[str, Any]:
        """Execute VR application and collect metrics"""

        # For MVP, we'll simulate VR app execution and generate synthetic metrics
        # In production, this would:
        # 1. Launch the actual VR application
        # 2. Navigate to the specified scene
        # 3. Collect real frame timing data
        # 4. Monitor VR-specific metrics

        print("    🎯 Executing scene "{scene_name}' for {duration}s")

        # Simulate app execution
        await asyncio.sleep(duration)

        # Generate synthetic but realistic VR metrics
        import random

        import numpy as np

        # Simulate frame times (in milliseconds)
        base_frame_time = 11.11  # Target 90 FPS
        frame_count = duration * 90  # 90 FPS target

        frame_times = []
        for i in range(frame_count):
            # Add some realistic variation and occasional spikes
            variation = random.gauss(0, 0.5)  # Small random variation
            spike_chance = random.random()

            if spike_chance < 0.02:  # 2% chance of frame spike
                spike = random.uniform(5, 15)  # 5-15ms spike
                frame_time = base_frame_time + spike
            else:
                frame_time = max(8.0, base_frame_time + variation)  # Min 8ms (125 FPS)

            frame_times.append(frame_time)

        # Calculate VR-specific metrics
        dropped_frames = sum(1 for ft in frame_times if ft > 20)  # Frames > 20ms
        reprojected_frames = sum(
            1 for ft in frame_times if ft > 13.33
        )  # Frames > 75 FPS

        app_metrics = {
            "scene_name": scene_name,
            "duration": duration,
            "frame_count": len(frame_times),
            "frame_times": frame_times,
            "dropped_frames": dropped_frames,
            "reprojected_frames": reprojected_frames,
            "motion_to_photon_ms": random.uniform(18, 25),  # Typical VR latency
            "tracking_quality": random.uniform(0.95, 1.0),  # Tracking quality score
            "execution_success": True,
        }

        print(
            f"    ✅ Scene execution complete: {len(frame_times)} frames, {dropped_frames} dropped"
        )

        return app_metrics

    async def destroy_container(self, container_id: str) -> bool:
        """Destroy container and cleanup resources"""

        if container_id not in self.active_containers:
            print(f"WARNING: Container {container_id} not found for cleanup")
            return False

        try:
            container_info = self.active_containers[container_id]
            container = container_info["container"]

            print(f"🗑️ Destroying container {container_id}")

            # Stop and remove container
            container.stop(timeout=10)
            container.remove()

            # Remove from active containers
            del self.active_containers[container_id]

            print(f"✅ Container {container_id} destroyed successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to destroy container {container_id}: {e}")
            return False

    async def get_container_logs(self, container_id: str) -> str:
        """Get container logs for debugging"""

        if container_id not in self.active_containers:
            return f"Container {container_id} not found"

        try:
            container = self.active_containers[container_id]["container"]
            logs = container.logs(tail=100).decode("utf-8")
            return logs

        except Exception as e:
            return f"Failed to get logs: {e}"

    async def list_active_containers(self) -> List[Dict[str, Any]]:
        """List all active containers"""

        containers = []

        for container_id, info in self.active_containers.items():
            try:
                container = info["container"]
                container.reload()

                containers.append(
                    {
                        "container_id": container_id,
                        "status": container.status,
                        "config": info["config"],
                        "created_at": info["created_at"],
                        "uptime": time.time() - info["created_at"],
                    }
                )

            except Exception as e:
                print(f"WARNING: Error getting container {container_id} info: {e}")

        return containers

    async def cleanup_all_containers(self):
        """Cleanup all active containers"""

        print("🧹 Cleaning up all active containers...")

        container_ids = list(self.active_containers.keys())

        for container_id in container_ids:
            await self.destroy_container(container_id)

        print(f"✅ Cleaned up {len(container_ids)} containers")

    async def cleanup(self):
        """Cleanup resources"""
        await self.cleanup_all_containers()

        if self.docker_client:
            self.docker_client.close()

        print("INFO: Container manager cleanup complete")
