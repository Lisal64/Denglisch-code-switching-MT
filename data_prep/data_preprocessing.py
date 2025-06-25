import csv
import os
from dotenv import load_dotenv
from data_prep.data_prep import prepare_hf_dataset
from datasets import DatasetDict, Dataset

load_dotenv()

def save_all_datasets(input_csv, output_dir="filtered_datasets"):
    os.makedirs(output_dir, exist_ok=True)
    strategies = ["ratio", "threshold", "hybrid"]

    for strategy in strategies:
        print(f"📦 Creating dataset for strategy: {strategy}")
        dataset_dict: DatasetDict = prepare_hf_dataset(
            input_csv_path=input_csv,
            strategy=strategy,
            keep_metadata=True,
            min_ratio=0.3,
            min_count=2
        )

        strategy_dir = os.path.join(output_dir, "dataset", strategy)
        os.makedirs(strategy_dir, exist_ok=True)

        for split in ["train", "validation", "test"]:
            split_path = os.path.join(strategy_dir, f"{split}.json")
            dataset_dict[split].to_json(split_path)
            print(f"Saved {len(dataset_dict[split])} examples to: {split_path}")

if __name__ == "__main__":
    input_csv_path = os.environ.get("DATA_INPUT_PATH")
    save_all_datasets(input_csv_path)

    base_dir = "/home/lisa/Denglisch-code-switching-MT/filtered_datasets/dataset"
    strategies = ["hybrid", "ratio", "threshold"]
    for strategy in strategies:
        for split in ["train", "validation", "test"]:
            path = os.path.join(base_dir, strategy, f"{split}.json")
            if os.path.exists(path):
                dataset = Dataset.from_json(path)
                print(f"{path}: {len(dataset)} examples")
            else:
                print(f"{path}: File not found.")
