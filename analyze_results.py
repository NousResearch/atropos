import glob
import json
from collections import defaultdict


def analyze_batch_results():
    """Analyze patterns in experiment results"""

    # Find all result files
    result_files = glob.glob("batch_results_*.json") + glob.glob(
        "padres_test_results_*.json"
    )

    if not result_files:
        print("No result files found. Run experiments first!")
        return

    all_experiments = []
    for file in result_files:
        with open(file, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_experiments.extend(data)
            else:
                all_experiments.append(data)

    print(f"Analyzing {len(all_experiments)} experiments...")

    # Basic statistics
    success_count = 0
    reward_sum = 0
    task_completion_count = 0

    for exp in all_experiments:
        action_data = exp.get("action", {})
        if action_data.get("reward", 0) > 0:
            success_count += 1
            reward_sum += action_data.get("reward", 0)

        if action_data.get("done", False):
            task_completion_count += 1

    print("\n📊 Experiment Analysis:")
    print(f"- Total experiments: {len(all_experiments)}")
    print(
        f"- Successful actions: {success_count}/{len(all_experiments)} ({success_count/len(all_experiments)*100:.1f}%)"
    )
    print(f"- Average reward: {reward_sum/len(all_experiments):.2f}")
    print(
        f"- Task completions: {task_completion_count}/{len(all_experiments)} ({task_completion_count/len(all_experiments)*100:.1f}%)"
    )

    # Claude's insights summary
    claude_insights = []
    for exp in all_experiments:
        if exp.get("llm_analysis"):
            claude_insights.append(exp["llm_analysis"])

    print(f"\n🤖 Claude analyzed {len(claude_insights)} experiments")
    print("Sample Claude insights:")
    for i, insight in enumerate(claude_insights[:3]):
        print(f"\nExperiment {i+1} Analysis:")
        print(f"  {insight[:200]}...")


if __name__ == "__main__":
    analyze_batch_results()
