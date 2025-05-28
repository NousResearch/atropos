def vr_comfort_optimizer(features):
    """
    VR Comfort Optimization Function
    Optimizes VR comfort based on performance features
    
    Args:
        features: List of 6 normalized features [gpu_util, vram_usage, cpu_util, scene_complexity, duration, app_type]
    
    Returns:
        float: Optimized comfort score (0.0 to 1.0)
    """
    if len(features) != 6:
        return 0.5  # Default comfort score
    
    gpu_util, vram_usage, cpu_util, scene_complexity, duration, app_type = features
    
    # Ensure all features are in valid range
    features = [max(0.0, min(1.0, f)) for f in features]
    gpu_util, vram_usage, cpu_util, scene_complexity, duration, app_type = features
    
    # Comfort optimization formula discovered through analysis
    # Lower GPU/CPU utilization generally improves comfort
    # Moderate scene complexity is optimal
    # Shorter durations reduce fatigue
    
    comfort_score = (
        0.3 * (1.0 - gpu_util) +           # Lower GPU usage = better comfort
        0.2 * (1.0 - cpu_util) +           # Lower CPU usage = better comfort  
        0.2 * (1.0 - abs(scene_complexity - 0.6)) +  # Optimal complexity around 0.6
        0.15 * (1.0 - duration) +          # Shorter duration = less fatigue
        0.1 * (1.0 - vram_usage) +         # Lower VRAM usage = better performance
        0.05 * app_type                    # App-specific adjustment
    )
    
    # Apply sigmoid function for smooth optimization
    import math
    comfort_score = 1.0 / (1.0 + math.exp(-5 * (comfort_score - 0.5)))
    
    return max(0.0, min(1.0, comfort_score))
