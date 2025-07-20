from finetuning.finetune_mbart_all_experiments import MBARTFineTuner
import json
import torch
import gc


if __name__ == "__main__":
    strategies = ["ratio", "hybrid", "threshold"]
    base_path = "filtered_datasets/mBART_dataset"
    model_name = "facebook/mbart-large-50-many-to-one-mmt"
    for strategy in strategies:
        dataset_path = f"{base_path}/{strategy}"

        for context_type in ["baseline", "lang", "pos", "lang+pos"]:
            exp = MBARTFineTuner(
                model_name=model_name,
                dataset_path=dataset_path,
                strategy_name=strategy,
                context_type=context_type
            )
            #best_params = exp.tune_hyperparameters(n_trials=5)
            with open(f"results_tuned/mBART_{strategy}_{context_type}/best_hyperparameters.json", "r") as f: # if already computed
                best_params = json.load(f)
            exp.run(best_params)
            del exp
            torch.cuda.empty_cache()
            gc.collect()