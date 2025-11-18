import datasets

def compute_task_stats(dataset_name: str) -> dict:
    dataset = datasets.load_dataset(dataset_name, split="train")
    total_items = len(dataset)

    task_type_counts = {}
    for item in dataset:
        task_type = item.get("language", "unknown")
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

    task_stats = {
        "total_items": total_items,
        "task_type_distribution": task_type_counts,
    }
    return task_stats

if __name__ == "__main__":
    dataset_name = "NousResearch/swe-agent-13k-2025-06-15"
    stats = compute_task_stats(dataset_name)
    print(f"Dataset: {dataset_name}")
    print(f"Total items: {stats['total_items']}")
    print("Task type distribution:")
    for task_type, count in stats["task_type_distribution"].items():
        print(f"  {task_type}: {count}")