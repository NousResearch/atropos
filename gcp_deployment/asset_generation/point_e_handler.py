import asyncio
import logging
import os
import random
from typing import Any, Optional, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)


class PointEHandler:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        logger.info(f"PointEHandler: Using device: {self.device}")
        self.base_model = None
        self.upsampler_model = None
        self.sampler = None
        self.models_loaded = False
        self.fallback_mode = False
        self.loop = (
            asyncio.get_event_loop() if hasattr(asyncio, "_get_running_loop") else None
        )

    async def load_models(self):
        """Load Point-E models with fallback to mock generation if models fail"""
        if self.models_loaded:
            return True

        try:
            logger.info("PointEHandler: Attempting to load Point-E models...")

            # Import Point-E modules
            from point_e.diffusion.sampler import PointCloudSampler
            from point_e.models.configs import MODEL_CONFIGS, model_from_config
            from point_e.models.download import load_checkpoint

            # Create event loop if needed
            if self.loop is None:
                self.loop = asyncio.get_event_loop()

            logger.info("PointEHandler: Creating base model...")
            self.base_name = "base40M-textvec"
            self.base_model = await self.loop.run_in_executor(
                None, model_from_config, MODEL_CONFIGS[self.base_name], self.device
            )
            await self.loop.run_in_executor(None, self.base_model.eval)
            self.base_diffusion = MODEL_CONFIGS[self.base_name]["diffusion_kwargs"]

            logger.info("PointEHandler: Creating upsampler model...")
            self.upsampler_model = await self.loop.run_in_executor(
                None, model_from_config, MODEL_CONFIGS["upsample"], self.device
            )
            await self.loop.run_in_executor(None, self.upsampler_model.eval)
            self.upsampler_diffusion = MODEL_CONFIGS["upsample"]["diffusion_kwargs"]

            logger.info("PointEHandler: Downloading/loading base checkpoint...")
            base_model_state_dict = await self.loop.run_in_executor(
                None, load_checkpoint, self.base_name, self.device
            )
            await self.loop.run_in_executor(
                None, self.base_model.load_state_dict, base_model_state_dict
            )

            logger.info("PointEHandler: Downloading/loading upsampler checkpoint...")
            upsampler_model_state_dict = await self.loop.run_in_executor(
                None, load_checkpoint, "upsample", self.device
            )
            await self.loop.run_in_executor(
                None, self.upsampler_model.load_state_dict, upsampler_model_state_dict
            )

            self.sampler = PointCloudSampler(
                device=self.device,
                models=[self.base_model, self.upsampler_model],
                diffusions=[self.base_diffusion, self.upsampler_diffusion],
                num_points=[1024, 4096 - 1024],
                aux_channels=["R", "G", "B"],
                guidance_scale=[3.0, 0.0],
                model_kwargs_key_filter=("texts", ""),
            )

            self.models_loaded = True
            self.fallback_mode = False
            logger.info("PointEHandler: Point-E models loaded successfully!")
            return True

        except Exception as e:
            logger.warning(f"PointEHandler: Failed to load Point-E models: {e}")
            logger.info(
                "PointEHandler: Enabling fallback mode with procedural generation"
            )
            self.fallback_mode = True
            self.models_loaded = True  # Consider fallback as "loaded"
            return True

    async def generate_point_cloud_from_text(
        self, prompt: str, output_path: str = "point_cloud.ply"
    ):
        """Generate point cloud from text, with fallback to procedural generation"""
        if not self.models_loaded:
            logger.warning("PointEHandler: Models not loaded. Attempting to load now.")
            if not await self.load_models():
                logger.error("PointEHandler: Model loading failed completely.")
                return None, None

        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        logger.info("PointEHandler: Generating point cloud for prompt: "{prompt}'")

        if self.fallback_mode or not self.sampler:
            logger.info("PointEHandler: Using fallback procedural generation")
            return await self._generate_fallback_point_cloud(prompt, output_path)
        else:
            logger.info("PointEHandler: Using full Point-E generation")
            return await self._generate_real_point_cloud(prompt, output_path)

    async def _generate_real_point_cloud(self, prompt: str, output_path: str):
        """Generate using real Point-E models"""
        try:
            from point_e.util.plotting import plot_point_cloud

            def _blocking_sample_and_process(sampler, p):
                samples_out = None
                for x_samples in sampler.sample_batch_progressive(
                    batch_size=1, model_kwargs=dict(texts=[p])
                ):
                    samples_out = x_samples
                if samples_out is None:
                    return None, None
                pc_out = sampler.output_to_point_clouds(samples_out)[0]
                fig_out = plot_point_cloud(
                    pc_out,
                    grid_size=3,
                    fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)),
                )
                return pc_out, fig_out

            def _blocking_save(pc_data, out_path):
                coords_cpu = pc_data.coords.cpu().numpy()
                colors_cpu = None
                if pc_data.channels is not None and all(
                    k in pc_data.channels for k in ["R", "G", "B"]
                ):
                    r = pc_data.channels["R"].cpu().numpy()
                    g = pc_data.channels["G"].cpu().numpy()
                    b = pc_data.channels["B"].cpu().numpy()
                    colors_cpu = (
                        np.stack([r.squeeze(), g.squeeze(), b.squeeze()], axis=-1) * 255
                    ).astype(np.uint8)
                    if colors_cpu.shape[0] != coords_cpu.shape[0]:
                        logger.warning("Coord/color mismatch, skipping colors.")
                        colors_cpu = None
                cloud = trimesh.points.PointCloud(coords_cpu, colors=colors_cpu)
                cloud.export(out_path)
                logger.info(f"PointEHandler: Point cloud saved to {out_path}")
                return True

            pc, fig = await self.loop.run_in_executor(
                None, _blocking_sample_and_process, self.sampler, prompt
            )
            if pc is None or fig is None:
                logger.error(
                    f"PointEHandler: Failed to generate point cloud samples for prompt: {prompt}"
                )
                return None, None

            save_success = await self.loop.run_in_executor(
                None, _blocking_save, pc, output_path
            )
            if not save_success:
                logger.error(
                    f"PointEHandler: Failed to save point cloud to {output_path}"
                )
                return pc, fig

            return pc, fig

        except Exception as e:
            logger.error(
                "PointEHandler: Error in real Point-E generation for "{prompt}': {e}"
            )
            logger.info("PointEHandler: Falling back to procedural generation")
            return await self._generate_fallback_point_cloud(prompt, output_path)

    async def _generate_fallback_point_cloud(self, prompt: str, output_path: str):
        """Generate procedural point cloud based on text description"""
        try:
            logger.info(
                f"PointEHandler: Generating procedural point cloud for: {prompt}"
            )

            # Simple keyword-based shape generation
            shape_type = self._determine_shape_from_prompt(prompt)
            color = self._determine_color_from_prompt(prompt)

            # Generate basic shape
            if shape_type == "cube":
                vertices, colors = self._generate_cube_points(color)
            elif shape_type == "sphere":
                vertices, colors = self._generate_sphere_points(color)
            elif shape_type == "cylinder":
                vertices, colors = self._generate_cylinder_points(color)
            else:
                vertices, colors = self._generate_abstract_points(color)

            # Save as PLY file
            cloud = trimesh.points.PointCloud(vertices, colors=colors)
            cloud.export(output_path)

            logger.info(f"PointEHandler: Procedural point cloud saved to {output_path}")

            # Return mock objects for compatibility
            class MockPointCloud:
                def __init__(self, coords, channels=None):
                    self.coords = torch.tensor(coords)
                    self.channels = channels or {}

            mock_pc = MockPointCloud(vertices)
            mock_fig = None  # No matplotlib figure for procedural generation

            return mock_pc, mock_fig

        except Exception as e:
            logger.error(f"PointEHandler: Error in fallback generation: {e}")
            return None, None

    def _determine_shape_from_prompt(self, prompt: str) -> str:
        """Determine basic shape from text prompt"""
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["cube", "box", "block", "square"]):
            return "cube"
        elif any(word in prompt_lower for word in ["ball", "sphere", "round", "orb"]):
            return "sphere"
        elif any(word in prompt_lower for word in ["cylinder", "tube", "pipe", "rod"]):
            return "cylinder"
        else:
            return "abstract"

    def _determine_color_from_prompt(self, prompt: str) -> Tuple[int, int, int]:
        """Determine color from text prompt"""
        prompt_lower = prompt.lower()

        color_map = {
            "red": (255, 100, 100),
            "blue": (100, 100, 255),
            "green": (100, 255, 100),
            "yellow": (255, 255, 100),
            "purple": (255, 100, 255),
            "orange": (255, 165, 0),
            "pink": (255, 192, 203),
            "brown": (165, 42, 42),
            "gray": (128, 128, 128),
            "white": (255, 255, 255),
            "black": (50, 50, 50),
        }

        for color_name, rgb in color_map.items():
            if color_name in prompt_lower:
                return rgb

        # Default to a random pleasant color
        return (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255),
        )

    def _generate_cube_points(
        self, color: Tuple[int, int, int], num_points: int = 2048
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate point cloud for a cube"""
        points = []

        # Generate points on cube faces
        for _ in range(num_points):
            # Choose random face (6 faces)
            face = random.randint(0, 5)

            if face == 0:  # Front face
                x, y, z = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0.5
            elif face == 1:  # Back face
                x, y, z = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), -0.5
            elif face == 2:  # Right face
                x, y, z = 0.5, random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
            elif face == 3:  # Left face
                x, y, z = -0.5, random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
            elif face == 4:  # Top face
                x, y, z = random.uniform(-0.5, 0.5), 0.5, random.uniform(-0.5, 0.5)
            else:  # Bottom face
                x, y, z = random.uniform(-0.5, 0.5), -0.5, random.uniform(-0.5, 0.5)

            points.append([x, y, z])

        vertices = np.array(points)
        colors = np.tile(color, (num_points, 1)).astype(np.uint8)

        return vertices, colors

    def _generate_sphere_points(
        self, color: Tuple[int, int, int], num_points: int = 2048
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate point cloud for a sphere"""
        points = []

        for _ in range(num_points):
            # Generate point on unit sphere surface
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)

            x = 0.5 * np.sin(phi) * np.cos(theta)
            y = 0.5 * np.sin(phi) * np.sin(theta)
            z = 0.5 * np.cos(phi)

            points.append([x, y, z])

        vertices = np.array(points)
        colors = np.tile(color, (num_points, 1)).astype(np.uint8)

        return vertices, colors

    def _generate_cylinder_points(
        self, color: Tuple[int, int, int], num_points: int = 2048
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate point cloud for a cylinder"""
        points = []

        for _ in range(num_points):
            # Generate point on cylinder surface
            theta = random.uniform(0, 2 * np.pi)
            z = random.uniform(-0.5, 0.5)

            x = 0.3 * np.cos(theta)
            y = 0.3 * np.sin(theta)

            points.append([x, y, z])

        vertices = np.array(points)
        colors = np.tile(color, (num_points, 1)).astype(np.uint8)

        return vertices, colors

    def _generate_abstract_points(
        self, color: Tuple[int, int, int], num_points: int = 2048
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate abstract point cloud"""
        # Create interesting abstract shape with some structure
        points = []

        for _ in range(num_points):
            # Create a twisted torus-like shape
            u = random.uniform(0, 2 * np.pi)
            v = random.uniform(0, 2 * np.pi)

            R = 0.4  # Major radius
            r = 0.1  # Minor radius

            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v) + 0.1 * np.sin(3 * u)  # Add twist

            points.append([x, y, z])

        vertices = np.array(points)
        colors = np.tile(color, (num_points, 1)).astype(np.uint8)

        return vertices, colors


async def main_test_point_e():
    """Test function for Point-E handler"""
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing Point-E Handler for async test...")

    handler = PointEHandler()
    if not handler.models_loaded:
        await handler.load_models()

    if handler.models_loaded:
        test_prompts = [
            "a blue car",
            "a red cube",
            "a green sphere",
            "a yellow cylinder",
            "an abstract purple sculpture",
        ]

        for prompt in test_prompts:
            logger.info("Testing generation for: "{prompt}'")
            try:
                output_file = f"test_{prompt.replace(' ', '_').replace(',', '')}.ply"
                point_cloud, figure = await handler.generate_point_cloud_from_text(
                    prompt, output_file
                )
                if point_cloud:
                    logger.info(f"✓ Generation successful for: {prompt}")
                    if os.path.exists(output_file):
                        size = os.path.getsize(output_file)
                        logger.info(f"✓ File saved: {output_file} ({size} bytes)")
                else:
                    logger.error(f"✗ Generation failed for: {prompt}")
            except Exception as e:
                logger.error("✗ Error generating "{prompt}': {e}")
    else:
        logger.error("Models failed to load completely.")


if __name__ == "__main__":
    asyncio.run(main_test_point_e())

    # Example of how to generate an image and then a point cloud
    # print("Loading GLIDE model for text-to-image generation...")
    # glide_model, glide_diffusion, glide_options = load_glide_model_and_diffusion(self.device)
    # img = generate_image_from_text(glide_model, glide_diffusion, glide_options, self.device, prompt)
    # samples = generate_point_cloud_from_image(self.base_model, self.base_diffusion, self.upsampler_model, self.upsampler_diffusion, self.device, img)
    # pc = self.sampler.output_to_point_clouds(samples)[0]
    # fig = plot_point_cloud(pc, grid_size=3)
    # fig.show()
