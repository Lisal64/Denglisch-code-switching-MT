from finetuning.finetune_mbart_all_experiments import MBARTFineTuner


if __name__ == "__main__":
    datasets = ["filtered_datasets/mBART_dataset/ratio", "filtered_datasets/mBART_dataset/hybrid", "filtered_datasets/mBART_dataset/threshold"]
    strategies = ["ratio", "hybrid", "threshold"]
    model_name = "facebook/mbart-large-50-many-to-one-mmt"
    for dataset in datasets:
        dataset_path = dataset

        for context_type in ["baseline", "lang", "pos", "lang+pos"]:
            for strategy in strategies:
                exp = MBARTFineTuner(
                    model_name=model_name,
                    dataset_path="filtered_datasets/mBART_dataset/hybrid",
                    strategy_name="hybrid",
                    context_type="lang+pos"
                )
                best_params = exp.tune_hyperparameters(n_trials=5)
                exp.run(best_params)