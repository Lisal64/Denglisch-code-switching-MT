from model_evaluation.mbart_evaluate import MBARTEvaluator
from dotenv import load_dotenv
import json
import os

if __name__ == "__main__":
    load_dotenv()

    strategies = ["ratio", "threshold", "hybrid"]
    for strategy_name in strategies:
        dataset_base = f"filtered_datasets/mBART_dataset/{strategy_name}"
        model_base = f"models/mBART_{strategy_name}_"
        output_base = f"results_tuned/mBART_{strategy_name}_"

        contexts = ["baseline", "lang", "pos", "lang+pos"]

        evaluators = []

        for context in contexts:
            model_path = os.path.join(model_base + context)
            dataset_path = dataset_base
            output_dir = os.path.join(output_base + context, "test_metrics")
            hyperparam_path = os.path.join(output_base + context, "best_hyperparameters.json")

            evaluator = MBARTEvaluator(
                model_path=model_path,
                dataset_path=dataset_path,
                output_dir=output_dir
            )

            if os.path.exists(hyperparam_path):
                with open(hyperparam_path, "r") as f:
                    best_params = json.load(f)
            else:
                best_params = None

            print(f"\nEvaluating: {context}")
            results, preds, labels = evaluator.evaluate(params=best_params)