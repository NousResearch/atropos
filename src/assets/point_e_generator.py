"""
Point-E 3D Asset Generator
Integrates Point-E for generating 3D assets from text descriptions
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logging.warning("trimesh not available, using mock 3D generation")

logger = logging.getLogger(__name__)


@dataclass
class AssetRequest:
    """Request for 3D asset generation"""

    description: str
    agent_id: str
    asset_type: str = "object"  # object, tool, decoration, etc.
    complexity: str = "simple"  # simple, medium, complex
    timestamp: float = 0.0


@dataclass
class Generated3DAsset:
    """Generated 3D asset result"""

    asset_id: str
    description: str
    vertices: np.ndarray
    faces: np.ndarray
    colors: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    file_path: Optional[str] = None
    generation_time: float = 0.0


class PointEGenerator:
    """
    Point-E integration for 3D asset generation
    """

    def __init__(self, config):
        self.config = config
        self.asset_cache: Dict[str, Generated3DAsset] = {}
        self.generation_queue: asyncio.Queue = asyncio.Queue()

        # Statistics
        self.total_generated = 0
        self.cache_hits = 0
        self.generation_time = 0.0

        # Mock primitives for fallback
        self.primitive_assets = self._create_primitive_assets()

        logger.info(
            f"Point-E Generator initialized (trimesh available: {TRIMESH_AVAILABLE})"
        )

    def _create_primitive_assets(self) -> Dict[str, Generated3DAsset]:
        """Create basic primitive 3D shapes for fallback"""
        primitives = {}

        if not TRIMESH_AVAILABLE:
            # Minimal fallback without trimesh
            return {
                "cube": Generated3DAsset(
                    asset_id="primitive_cube",
                    description="A simple cube",
                    vertices=np.array(
                        [
                            [0, 0, 0],
                            [1, 0, 0],
                            [1, 1, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 1],
                            [1, 1, 1],
                            [0, 1, 1],
                        ]
                    ),
                    faces=np.array(
                        [
                            [0, 1, 2],
                            [0, 2, 3],
                            [4, 5, 6],
                            [4, 6, 7],
                            [0, 1, 5],
                            [0, 5, 4],
                            [2, 3, 7],
                            [2, 7, 6],
                            [1, 2, 6],
                            [1, 6, 5],
                            [0, 3, 7],
                            [0, 7, 4],
                        ]
                    ),
                    metadata={"type": "primitive", "complexity": "simple"},
                ),
                "sphere": Generated3DAsset(
                    asset_id="primitive_sphere",
                    description="A simple sphere",
                    vertices=np.random.randn(100, 3),  # Rough sphere approximation
                    faces=np.array([[i, i + 1, i + 2] for i in range(0, 97, 3)]),
                    metadata={"type": "primitive", "complexity": "simple"},
                ),
            }

        try:
            # Create trimesh primitives
            cube = trimesh.creation.box([1, 1, 1])
            sphere = trimesh.creation.icosphere(radius=0.5, subdivisions=2)
            cylinder = trimesh.creation.cylinder(radius=0.3, height=1.0)

            primitives.update(
                {
                    "cube": Generated3DAsset(
                        asset_id="primitive_cube",
                        description="A simple cube",
                        vertices=cube.vertices,
                        faces=cube.faces,
                        metadata={"type": "primitive", "complexity": "simple"},
                    ),
                    "sphere": Generated3DAsset(
                        asset_id="primitive_sphere",
                        description="A simple sphere",
                        vertices=sphere.vertices,
                        faces=sphere.faces,
                        metadata={"type": "primitive", "complexity": "simple"},
                    ),
                    "cylinder": Generated3DAsset(
                        asset_id="primitive_cylinder",
                        description="A simple cylinder",
                        vertices=cylinder.vertices,
                        faces=cylinder.faces,
                        metadata={"type": "primitive", "complexity": "simple"},
                    ),
                }
            )

        except Exception as e:
            logger.warning(f"Failed to create trimesh primitives: {e}")

        return primitives

    async def generate_asset(self, request: AssetRequest) -> Optional[Generated3DAsset]:
        """
        Generate a 3D asset from text description
        """
        self.total_generated += 1
        start_time = time.time()

        # Check cache first
        cache_key = self._get_cache_key(request)
        if cache_key in self.asset_cache:
            self.cache_hits += 1
            logger.debug(f"Asset cache hit for: {request.description}")
            return self.asset_cache[cache_key]

        try:
            # Generate asset (mock implementation for now)
            asset = await self._generate_asset_mock(request)

            generation_time = time.time() - start_time
            self.generation_time += generation_time
            asset.generation_time = generation_time

            # Cache the result
            if self.config.assets.cache_assets:
                self.asset_cache[cache_key] = asset

            # Save to file if configured
            if asset and self.config.output.directory:
                await self._save_asset_file(asset)

            logger.info(
                "Generated asset "{asset.description}' in {generation_time:.2f}s"
            )
            return asset

        except Exception as e:
            logger.error("Asset generation failed for "{request.description}': {e}")
            return None

    async def _generate_asset_mock(self, request: AssetRequest) -> Generated3DAsset:
        """
        Mock asset generation (placeholder for Point-E integration)
        """
        # Simulate generation time
        complexity_time = {"simple": 0.5, "medium": 1.5, "complex": 3.0}
        await asyncio.sleep(complexity_time.get(request.complexity, 1.0))

        # Choose appropriate primitive based on description
        primitive = self._choose_primitive_for_description(request.description)

        if primitive:
            # Create a variant of the primitive
            asset = Generated3DAsset(
                asset_id=f"generated_{request.agent_id}_{int(time.time())}",
                description=request.description,
                vertices=primitive.vertices.copy(),
                faces=primitive.faces.copy(),
                metadata={
                    "type": "generated",
                    "complexity": request.complexity,
                    "agent_id": request.agent_id,
                    "base_primitive": primitive.asset_id,
                },
            )

            # Add some variation to make it unique
            if TRIMESH_AVAILABLE:
                asset.vertices += np.random.normal(0, 0.05, asset.vertices.shape)

            return asset
        else:
            # Fallback to cube if no suitable primitive
            return self.primitive_assets.get(
                "cube", list(self.primitive_assets.values())[0]
            )

    def _choose_primitive_for_description(
        self, description: str
    ) -> Optional[Generated3DAsset]:
        """Choose appropriate primitive based on text description"""
        desc_lower = description.lower()

        # Simple keyword matching
        if any(
            word in desc_lower for word in ["chair", "table", "box", "chest", "crate"]
        ):
            return self.primitive_assets.get("cube")
        elif any(word in desc_lower for word in ["ball", "pot", "bowl", "vase"]):
            return self.primitive_assets.get("sphere")
        elif any(
            word in desc_lower for word in ["stick", "rod", "pole", "tool", "handle"]
        ):
            return self.primitive_assets.get("cylinder")
        else:
            # Default to cube for unknown objects
            return self.primitive_assets.get("cube")

    def _get_cache_key(self, request: AssetRequest) -> str:
        """Generate cache key for asset request"""
        return f"{hash(request.description)}_{request.complexity}_{request.asset_type}"

    async def _save_asset_file(self, asset: Generated3DAsset):
        """Save asset to file"""
        try:
            assets_dir = Path(self.config.output.directory) / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)

            file_path = assets_dir / f"{asset.asset_id}.obj"

            if TRIMESH_AVAILABLE:
                # Create trimesh object and export
                mesh = trimesh.Trimesh(vertices=asset.vertices, faces=asset.faces)
                mesh.export(str(file_path))
            else:
                # Basic OBJ file format
                with open(file_path, "w") as f:
                    f.write(f"# Generated asset: {asset.description}\n")
                    for vertex in asset.vertices:
                        f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                    for face in asset.faces:
                        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

            asset.file_path = str(file_path)
            logger.debug(f"Saved asset to: {file_path}")

        except Exception as e:
            logger.warning(f"Failed to save asset file: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            "total_generated": self.total_generated,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_generated),
            "avg_generation_time": self.generation_time / max(1, self.total_generated),
            "cache_size": len(self.asset_cache),
            "primitive_types": len(self.primitive_assets),
        }

    def list_available_assets(self) -> List[str]:
        """List all available assets"""
        return list(self.asset_cache.keys()) + list(self.primitive_assets.keys())


class AssetManager:
    """
    Manages 3D assets for the simulation
    """

    def __init__(self, config):
        self.config = config
        self.generator = PointEGenerator(config)
        self.active_assets: Dict[str, Generated3DAsset] = {}

        logger.info("Asset Manager initialized")

    async def create_asset_for_agent(
        self,
        agent_id: str,
        description: str,
        asset_type: str = "object",
        complexity: str = "simple",
    ) -> Optional[Generated3DAsset]:
        """Create a 3D asset for an agent"""

        if not self.config.assets.enable_generation:
            logger.debug("Asset generation disabled")
            return None

        request = AssetRequest(
            description=description,
            agent_id=agent_id,
            asset_type=asset_type,
            complexity=complexity,
            timestamp=time.time(),
        )

        asset = await self.generator.generate_asset(request)

        if asset:
            self.active_assets[asset.asset_id] = asset
            logger.info("Created asset "{description}' for agent {agent_id}")

        return asset

    def get_asset(self, asset_id: str) -> Optional[Generated3DAsset]:
        """Get asset by ID"""
        return self.active_assets.get(asset_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get asset manager statistics"""
        generator_stats = self.generator.get_stats()
        return {
            **generator_stats,
            "active_assets": len(self.active_assets),
            "generation_enabled": self.config.assets.enable_generation,
        }
