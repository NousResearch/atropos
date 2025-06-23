#!/usr/bin/env python3
"""
LARGE-SCALE REAL VR RESEARCH STUDY
Conducts 25 real VR experiments for statistical significance
"""

import json
from datetime import datetime

from real_vr_research_pipeline import RealVRResearchPipeline


def run_large_vr_study():
    """Run a large-scale VR research study"""
    print("🔬 LARGE-SCALE REAL VR RESEARCH STUDY")
    print("=" * 60)

    # Initialize pipeline
    pipeline = RealVRResearchPipeline()

    # Run 25 experiments for statistical significance
    print("\n🚀 Running 25 real VR experiments...")
    results = pipeline.run_real_research_batch(num_experiments=25)

    # Save data with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"large_vr_study_results_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    # Analyze results
    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful

    print("\n📊 LARGE-SCALE STUDY RESULTS")
    print(f"   Total experiments: {len(results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {successful/len(results)*100:.1f}%")
    print(f"   Data file: {filename}")

    # Extract physics data for analysis
    physics_data = []
    for result in results:
        if result.get("success") and "real_physics_data" in result:
            physics_data.append(result["real_physics_data"])

    if physics_data:
        print("\n🔬 PHYSICS DATA ANALYSIS")
        print(f"   Physics experiments: {len(physics_data)}")
        print(
            f"   Average reward: {sum(p.get('reward', 0) for p in physics_data) / len(physics_data):.3f}"
        )
        print(
            f"   Task completion rate: {sum(1 for p in physics_data if p.get('done')) / len(physics_data) * 100:.1f}%"
        )

        # Position consistency analysis
        positions = [
            p.get("object_positions", [{}])[0].get("position", [0, 0, 0])
            for p in physics_data
            if p.get("object_positions")
        ]
        if positions:
            avg_x = sum(pos[0] for pos in positions) / len(positions)
            avg_y = sum(pos[1] for pos in positions) / len(positions)
            avg_z = sum(pos[2] for pos in positions) / len(positions)
            print(
                f"   Average red cube position: [{avg_x:.6f}, {avg_y:.6f}, {avg_z:.6f}]"
            )

    print("\n✨ LARGE-SCALE REAL VR STUDY COMPLETE! ✨")
    return filename, results


if __name__ == "__main__":
    run_large_vr_study()
