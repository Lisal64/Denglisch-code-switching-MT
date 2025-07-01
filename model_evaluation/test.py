from model_evaluation.mbart_evaluate import MBARTEvaluator
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()

    baseline = MBARTEvaluator(
        model_path="facebook/mbart-large-50",
        dataset_path="filtered_datasets/mBART_dataset/ratio",
        output_dir="results/mBART_base/test_metrics"
    )

    finetuned = MBARTEvaluator(
        model_path="models/mBART_ratio_baseline",
        dataset_path="filtered_datasets/mBART_dataset/ratio",
        output_dir="results/mBART_ratio_baseline/test_metrics"
    )

    lang = MBARTEvaluator(
        model_path="models/mBART_ratio_lang",
        dataset_path="filtered_datasets/mBART_dataset/ratio",
        output_dir="results/mBART_ratio_lang/test_metrics"
    )

    pos = MBARTEvaluator(
        model_path="models/mBART_ratio_pos",
        dataset_path="filtered_datasets/mBART_dataset/ratio",
        output_dir="results/mBART_ratio_pos/test_metrics"
    )

    lang_pos = MBARTEvaluator(
        model_path="models/mBART_ratio_lang+pos",
        dataset_path="filtered_datasets/mBART_dataset/ratio",
        output_dir="results/mBART_ratio_lang+pos/test_metrics"
    )

    for evaluator in [baseline, finetuned, lang, pos, lang_pos]:
        results, preds, labels = evaluator.evaluate()
