StarMapCompression: A Reinforcement Learning Approach to Galaxy Data Compression
Project Overview
StarMapCompression is a proof-of-concept project developed for a hackathon to demonstrate how reinforcement learning (RL) can compress 3D galaxy data for efficient visualization in a browser. Our vision is to enable a "universe in a browser" experience, where users can explore vast astronomical datasets interactively. However, large datasets (e.g., millions of stars) are a limiting factor due to memory and bandwidth constraints in web applications. This project uses RL to intelligently compress galaxy point clouds while preserving key features for user viewpoints, making data lightweight for browser rendering.
We kept the approach simple to focus on proving the concept within the hackathon’s time constraints. The result is a functional RL environment that reduces a dataset from 1000 points to as few as 47 points (~95% compression) while maintaining view-relevant information, showcasing the potential for scalable, browser-based astronomy visualizations.
Why Data Compression?
Rendering the universe in a browser requires handling massive 3D datasets (e.g., star positions [x, y, z]). Transmitting and rendering these datasets in real-time is challenging due to:
Memory Limits: Browsers have limited memory, making large datasets impractical.

Bandwidth: Downloading millions of points is slow, degrading user experience.

Rendering: WebGL and other browser technologies struggle with high point counts.

Data compression is critical to overcome these limits. Our project uses RL to learn how to reduce the number of points while prioritizing those visible from user viewpoints, ensuring a visually accurate yet lightweight dataset.
What We Did
We developed StarMapCompressionEnv, an RL environment that compresses 3D galaxy data using a pipeline of spatial algorithms and optimizes the compression with RL. Here’s the process from start to finish:
Input Data:
Galaxy Data: A subset of 1000 star positions (galaxy_subset.npy, shape: (1000, 3)).

User Viewpoints: 10 viewpoints (user_views.npy, shape: (10, 3)) representing where users look in the galaxy.

Compression Pipeline:
Density Sampling: Uses a KD-tree to sample 10% of points (100 points) based on local density, focusing on clustered regions.

PCA: Reduces 3D points to 2D (padded back to 3D) to preserve variance and simplify processing.

Octree: Builds a hierarchical spatial structure, pruning sparse regions with an adaptive density threshold to retain key points.

Quantization: Compresses coordinates to 8-bit integers (dequantized for compatibility).

Mapping to Original: Maps compressed points to the nearest original points using a KD-tree, ensuring fidelity.

RL Optimization:
The RL environment (StarMapCompressionEnv) selects grid sizes (e.g., 26.7, 53.4, 80.1 units) to adjust the compression pipeline.

Actions: Choose a grid size to recompress the data, varying octree depth, minimum points, quantization bits, and sampling fraction.

Reward: Balances data size (-avg_data_size / 1000), point retention (5 * len(self.data) / len(self.original_data)), points within view radius (total_points / len(self.original_data)), and quality (-quality / 1e6).

Steps: Runs multiple steps (e.g., 5) to iteratively refine the dataset, reducing points (e.g., from 79 to 47) while optimizing for views.

Output:
A compressed dataset (e.g., 47 points, shape: (47, 3)) saved as .npy files (compressed_data_step_X.npy).

Metrics: Initial shape (79, 3), final shape (47, 3), Data changed: True, showing successful compression and modification.

View distances (min=30.89, max=365.58) guide view radius (~53.4) to capture relevant points.

How We Built It
Grok for Code Generation: We used Grok, created by xAI, to generate and debug the Python code for StarMapCompressionEnv and run_rl.py. Grok helped design the RL environment, implement spatial algorithms (KD-tree, octree), and fix issues (e.g., empty octree, syntax errors) by providing iterative code improvements based on our feedback and logs.

Atropos Environment: Due to hackathon time constraints, we ran the RL project on a local CPU using the Atropos environment (a custom setup for RL tasks). This avoided the need for GPU setup or cloud resources, enabling rapid development and testing.

Local Execution: The project runs on a local machine (C:\Users\boobs\Projects\space\Atropos\), processing galaxy_subset.npy and user_views.npy with a simple script (run_rl.py).

Why We Kept It Simple
As a proof of concept, we prioritized simplicity to demonstrate RL-driven compression within the hackathon’s tight timeline:
Small Dataset: Used 1000 points to keep computation fast and manageable on a CPU.

Basic RL: Employed a greedy RL approach (run_rl_step) with 5–50 steps instead of a full RL agent (e.g., PPO) to focus on functionality.

Local Setup: Leveraged Atropos and Grok to avoid complex infrastructure, ensuring we could iterate quickly.

Core Features: Focused on a working pipeline (sampling, PCA, octree, quantization) and RL optimization, omitting advanced features (e.g., visualization, multi-agent RL) for future work.

This simplicity allowed us to prove that RL can compress galaxy data effectively, reducing points by ~95% while preserving view-relevant information, setting the stage for scalable browser-based visualizations.
Results
Compression: Reduced dataset from 1000 points to 79 initially, then to 47 after 5 RL steps (~95% compression).

Data Modification: Achieved Data changed: True, showing RL dynamically alters the point set.

View Optimization: View radius (~53.4) covers minimum view distance (30.89), capturing some points, though distant views (max=365.58) suggest alignment improvements.

Proof of Concept: Demonstrated RL’s potential to optimize compression for browser rendering, with logs confirming varied grid sizes and point counts.

How to Run
Setup:
Clone the repository to C:\Users\boobs\Projects\space\Atropos\.

Install dependencies: pip install numpy scipy scikit-learn openai python-dotenv.

Ensure galaxy_subset.npy and user_views.npy are in the project directory.

Set up a local OpenAI-compatible server at http://localhost:9001/v1 (or modify base_url in starmap_compression.py).

Run:
Execute python run_rl.py to run 50 RL steps (~25 seconds on a CPU).

Outputs: Compressed datasets (compressed_data_step_X.npy), final shape, Data changed status, and view distances.

Files:
starmap_compression.py: RL environment with compression pipeline.

run_rl.py: Script to run RL steps and save results.

galaxy_subset.npy: Input galaxy data (1000 points).

user_views.npy: User viewpoints (10 points).

Future Work
View Alignment: Center views around data mean to increase points_in_view.

Advanced RL: Use a PPO agent (e.g., Stable Baselines3) for better policy learning.

Visualization: Plot initial vs. final datasets to showcase compression.

Scalability: Test on larger datasets (e.g., 1M points) with GPU support.

Browser Integration: Export compressed data to WebGL for real-time rendering.

Why It Matters
StarMapCompression shows that RL can address the data compression bottleneck for "universe in a browser" applications. By reducing datasets to a fraction of their size while preserving view-relevant points, we pave the way for interactive, accessible astronomy visualizations. This proof of concept, built with Grok and Atropos on a local CPU, demonstrates the potential for scalable solutions in a short timeframe, making it a compelling hackathon contribution.

Data Explanation
What is the Data?
The data consists of two NumPy arrays used in the StarMapCompression project:
Galaxy Data (galaxy_subset.npy):
Description: A 3D point cloud of 1000 star positions, each with XYZ coordinates derived from a simplified Gaia dataset. The original CSV contained Right Ascension (RA), Declination (Dec), and redshift, representing stars’ celestial coordinates and distances.

Shape: (1000, 3) (1000 points, 3 coordinates: X, Y, Z).

Purpose: Serves as the input dataset to be compressed for efficient rendering in a browser-based Three.js application, simulating a galaxy view.

User Viewpoints (user_views.npy):
Description: 10 viewpoints, each with XYZ coordinates, simulating potential camera positions a user might select in a Three.js visualization (e.g., different angles or zooms within the galaxy).

Shape: (10, 3) (10 viewpoints, 3 coordinates: X, Y, Z).

Purpose: Guides the RL environment to prioritize points visible from these perspectives, ensuring the compressed dataset retains relevant stars.

Where Did It Come From?
Gaia Dataset (CSV):
Source: The original data is a subset of the Gaia Data Release (likely DR2 or DR3), a European Space Agency mission cataloging billions of stars with precise astrometric data (RA, Dec, parallax, etc.).

Simplification: You reduced the CSV to 169 MB by keeping only RA (degrees), Dec (degrees), and redshift (z, related to distance via cosmological models). Other columns (e.g., parallax, magnitudes) were stripped to focus on positional data.

Conversion to XYZ: A Colab script transformed RA, Dec, and redshift into Cartesian XYZ coordinates, accounting for spherical-to-Cartesian conversion and redshift-based distances. This produced galaxy_subset.npy with 1000 points, a small sample for hackathon prototyping.

User Views (user_views.npy):
Source: Generated in Colab to simulate viewpoints a user might encounter in a Three.js application.

Creation: The script likely sampled points within or near the galaxy’s bounding box, representing camera positions or focal points in a 3D visualization. These views mimic how users navigate a galaxy in a browser (e.g., zooming, panning).

Why Sampled?: The 1000-point sample and 10 views were chosen to keep the dataset manageable for RL processing on a local CPU within the hackathon’s time constraints, proving the compression concept before scaling to larger datasets.

Where Does It Live?
Location: Both files are stored in your project directory:
galaxy_subset.npy: C:\Users\boobs\Projects\space\Atropos\galaxy_subset.npy

user_views.npy: C:\Users\boobs\Projects\space\Atropos\user_views.npy

Output Files: The RL process generates compressed datasets:
compressed_data_step_X.npy (e.g., compressed_data_step_1.npy for initial 79 points, compressed_data_step_5.npy for final 47 points), saved in C:\Users\boobs\Projects\space\Atropos\.

Environment: The data is processed in the Atropos environment (a custom RL setup) on your local machine (C:\Users\boobs\Projects\space\Atropos\), using CPU due to hackathon time limits.

How Was It Used?
Input to RL Environment:
galaxy_subset.npy is loaded as the raw dataset to compress, representing a galaxy subset for browser rendering.

user_views.npy defines viewpoints to optimize compression, ensuring the reduced dataset includes points visible from these perspectives.

Compression Pipeline (in starmap_compression.py):
Density Sampling: Uses a KD-tree to select 100 points (10%) based on local density, focusing on star clusters.

PCA: Projects 3D points to 2D (padded to 3D) to reduce dimensionality while preserving variance.

Octree: Constructs a spatial hierarchy, pruning sparse regions with an adaptive density threshold (e.g., ~2.16e-9) to yield ~79 points initially.

Quantization: Compresses coordinates to 8-bit integers (dequantized for compatibility).

Mapping: Maps compressed points to the nearest original points, ensuring they align with the Gaia dataset.

RL Optimization:
The RL environment (StarMapCompressionEnv) selects grid sizes (26.7, 53.4, 80.1) to recompress the data via _recompress_data, adjusting max_depth (3–5), min_points (1–2), bits (6–8), and sample_fraction (0.3–0.6).

Reward: Balances data size (-avg_data_size / 1000), retention (5 * len(self.data) / len(self.original_data)), view-relevant points (total_points / len(self.original_data)), and quality (-quality / 1e6).

Steps: 5 steps reduce points from 79 to 47 (~95% compression from 1000), with Data changed: True.

Output: Compressed datasets are saved as .npy files, with logs tracking shapes, points_in_view, grid sizes, and rewards.

Purpose: Compression reduces the dataset for Three.js rendering, addressing browser constraints (memory, bandwidth, rendering speed) while maintaining visual fidelity for user views.

