#!/usr/bin/env python3
"""
1,000 EXPERIMENT VR RESEARCH STUDY
Streamlined and robust implementation
"""

import json
import random
import statistics
import time
from datetime import datetime

import numpy as np
import requests


class VR1000Study:
    def __init__(self):
        self.url = "https://padres-api-service-312425595703.us-central1.run.app"
        self.results = []
        self.start_time = None

        # Varied target positions for diversity
        self.targets = [
            [-0.4, 0.0, 0.2],
            [-0.3, 0.1, 0.15],
            [-0.5, -0.1, 0.25],
            [-0.2, 0.05, 0.1],
            [-0.6, 0.0, 0.3],
            [-0.4, 0.2, 0.2],
            [-0.35, -0.05, 0.18],
            [-0.45, 0.15, 0.22],
            [-0.25, -0.15, 0.12],
            [-0.55, 0.1, 0.28],
            [-0.15, 0.25, 0.08],
            [-0.65, -0.05, 0.32],
        ]

        print("🚀 VR 1000 EXPERIMENT STUDY INITIALIZED")
        print(f"📊 {len(self.targets)} target positions available")
        print("🎯 Target: 1,000 experiments with population-level data")

    def run_single_experiment(self, exp_num):
        """Run single experiment with error handling"""
        target = random.choice(self.targets)

        try:
            # Setup environment
            setup_response = requests.post(f"{self.url}/setup_environment", timeout=20)
            setup_response.raise_for_status()
            setup_data = setup_response.json()

            # Execute action
            action_response = requests.post(f"{self.url}/execute_action", timeout=20)
            action_response.raise_for_status()
            action_data = action_response.json()

            # Calculate metrics
            object_positions = action_data.get("full_outcome_debug", {}).get(
                "new_state_viz", []
            )
            if object_positions:
                actual_pos = object_positions[0].get("position", [0, 0, 0])
                distance = np.sqrt(
                    sum((a - t) ** 2 for a, t in zip(actual_pos, target))
                )
                accuracy = max(0, 1 - distance)

                result = {
                    "experiment_id": f"exp_{exp_num:04d}",
                    "experiment_number": exp_num,
                    "timestamp": datetime.utcnow().isoformat(),
                    "target_position": target,
                    "actual_position": actual_pos,
                    "distance_error": distance,
                    "positioning_accuracy": accuracy,
                    "reward": action_data.get("reward", 0),
                    "task_completed": action_data.get("done", False),
                    "task_id": action_data.get("task_id"),
                    "success": True,
                }

                return result
            else:
                return {
                    "experiment_id": f"exp_{exp_num:04d}",
                    "experiment_number": exp_num,
                    "error": "No position data",
                    "success": False,
                }

        except Exception as e:
            return {
                "experiment_id": f"exp_{exp_num:04d}",
                "experiment_number": exp_num,
                "error": str(e),
                "success": False,
            }

    def run_batch(self, start_exp, batch_size):
        """Run batch of experiments"""
        print(f"\n--- BATCH {start_exp}-{start_exp + batch_size - 1} ---")

        batch_results = []
        batch_start = time.time()

        for i in range(batch_size):
            exp_num = start_exp + i
            result = self.run_single_experiment(exp_num)
            batch_results.append(result)

            if result.get("success"):
                accuracy = result.get("positioning_accuracy", 0)
                print(f"  ✅ Exp {exp_num}: {accuracy:.3f} accuracy")
            else:
                print(f"  ❌ Exp {exp_num}: {result.get('error', 'Failed')}")

            # Small delay to avoid overwhelming the API
            time.sleep(0.3)

        batch_time = time.time() - batch_start
        batch_success = sum(1 for r in batch_results if r.get("success"))

        print(
            f"  📊 Batch: {batch_success}/{batch_size} success ({batch_success/batch_size*100:.1f}%) in {batch_time:.1f}s"
        )

        return batch_results

    def analyze_progress(self, current_results):
        """Analyze current progress"""
        successful = [r for r in current_results if r.get("success")]

        if not successful:
            return

        success_rate = len(successful) / len(current_results)
        accuracies = [r.get("positioning_accuracy", 0) for r in successful]
        distances = [r.get("distance_error", 0) for r in successful]

        elapsed = time.time() - self.start_time
        rate = len(current_results) / (elapsed / 60)  # experiments per minute

        print("\n📈 PROGRESS ANALYSIS:")
        print(
            f"  Completed: {len(current_results)}/1000 ({len(current_results)/10:.1f}%)"
        )
        print(f"  Success rate: {success_rate*100:.1f}%")
        print(f"  Mean accuracy: {statistics.mean(accuracies):.3f}")
        print(f"  Mean distance error: {statistics.mean(distances):.3f}")
        print(f"  Rate: {rate:.1f} experiments/minute")
        print(
            f"  Estimated completion: {(1000 - len(current_results)) / rate:.1f} minutes"
        )

    def run_1000_experiments(self):
        """Run the full 1,000 experiment study"""
        print("\n🎯 STARTING 1,000 EXPERIMENT STUDY")
        print("=" * 60)

        self.start_time = time.time()
        batch_size = 25  # Smaller batches for better monitoring

        for batch_start in range(1, 1001, batch_size):
            current_batch_size = min(batch_size, 1001 - batch_start)

            # Run batch
            batch_results = self.run_batch(batch_start, current_batch_size)
            self.results.extend(batch_results)

            # Progress analysis every 100 experiments
            if len(self.results) % 100 == 0:
                self.analyze_progress(self.results)

            # Adaptive delay based on success rate
            recent_success = sum(1 for r in batch_results if r.get("success")) / len(
                batch_results
            )
            if recent_success < 0.9:
                print("  ⚠️ Lower success rate - adding delay")
                time.sleep(3)
            else:
                time.sleep(1)

        # Final analysis
        total_time = time.time() - self.start_time
        successful = [r for r in self.results if r.get("success")]

        print("\n🎉 1,000 EXPERIMENT STUDY COMPLETE!")
        print(f"  Total experiments: {len(self.results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Success rate: {len(successful)/len(self.results)*100:.1f}%")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(
            f"  Average rate: {len(self.results)/(total_time/60):.1f} experiments/minute"
        )

        if successful:
            accuracies = [r.get("positioning_accuracy", 0) for r in successful]
            distances = [r.get("distance_error", 0) for r in successful]

            print(f"  Mean accuracy: {statistics.mean(accuracies):.3f}")
            print(f"  Accuracy std: {statistics.stdev(accuracies):.3f}")
            print(f"  Mean distance error: {statistics.mean(distances):.3f}")
            print(f"  Distance std: {statistics.stdev(distances):.3f}")
            print(f"  High accuracy (>0.9): {sum(1 for a in accuracies if a > 0.9)}")
            print(
                f"  Medium accuracy (0.7-0.9): {sum(1 for a in accuracies if 0.7 <= a <= 0.9)}"
            )
            print(f"  Low accuracy (<0.7): {sum(1 for a in accuracies if a < 0.7)}")

        return self.results

    def save_results(self, results):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full results
        filename = f"vr_1000_experiments_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Full results saved to: {filename}")

        # Save summary statistics
        successful = [r for r in results if r.get("success")]
        if successful:
            accuracies = [r.get("positioning_accuracy", 0) for r in successful]
            distances = [r.get("distance_error", 0) for r in successful]

            summary = {
                "study_info": {
                    "total_experiments": len(results),
                    "successful_experiments": len(successful),
                    "success_rate": len(successful) / len(results),
                    "study_date": timestamp,
                },
                "accuracy_stats": {
                    "mean": statistics.mean(accuracies),
                    "median": statistics.median(accuracies),
                    "stdev": statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
                    "min": min(accuracies),
                    "max": max(accuracies),
                },
                "distance_stats": {
                    "mean": statistics.mean(distances),
                    "median": statistics.median(distances),
                    "stdev": statistics.stdev(distances) if len(distances) > 1 else 0,
                    "min": min(distances),
                    "max": max(distances),
                },
                "performance_distribution": {
                    "high_accuracy_count": sum(1 for a in accuracies if a > 0.9),
                    "medium_accuracy_count": sum(
                        1 for a in accuracies if 0.7 <= a <= 0.9
                    ),
                    "low_accuracy_count": sum(1 for a in accuracies if a < 0.7),
                },
            }

            summary_filename = f"vr_1000_summary_{timestamp}.json"
            with open(summary_filename, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"📊 Summary statistics saved to: {summary_filename}")


def main():
    """Main execution"""
    print("🚀 VR RESEARCH: 1,000 EXPERIMENTS IN THE CLOUD")
    print("=" * 60)

    study = VR1000Study()

    # Run the study
    results = study.run_1000_experiments()

    # Save results
    study.save_results(results)

    print("\n✨ 1,000 EXPERIMENT STUDY COMPLETE! ✨")
    print("🔬 Population-level VR research data ready for analysis!")


if __name__ == "__main__":
    main()
