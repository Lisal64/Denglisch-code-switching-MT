from finetuning.finetune_mbart_all_experiments import MBARTFineTuner


if __name__ == "__main__":
    model_name = "facebook/mbart-large-50-many-to-one-mmt"
    dataset_path = "filtered_datasets/mBART_dataset/threshold"

    for context_type in ["baseline", "lang", "pos", "lang+pos"]:
        exp = MBARTFineTuner(
            model_name=model_name,
            dataset_path=dataset_path,
            strategy_name="threshold",
            context_type=context_type
        )
        exp.run()