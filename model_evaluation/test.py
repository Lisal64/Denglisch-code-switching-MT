from model_evaluation.mbart_evaluate import MBARTEvaluator
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()

    baseline = MBARTEvaluator(
        model_path="facebook/mbart-large-50",
        dataset_path="filtered_datasets/mBART_dataset/threshold",
        output_dir="results/mBART_threshold_base/test_metrics"
    )

    finetuned = MBARTEvaluator(
        model_path="models/mBART_threshold_baseline",
        dataset_path="filtered_datasets/mBART_dataset/threshold",
        output_dir="results/mBART_threshold_baseline/test_metrics"
    )

    lang = MBARTEvaluator(
        model_path="models/mBART_threshold_lang",
        dataset_path="filtered_datasets/mBART_dataset/threshold",
        output_dir="results/mBART_threshold_lang/test_metrics"
    )

    pos = MBARTEvaluator(
        model_path="models/mBART_threshold_pos",
        dataset_path="filtered_datasets/mBART_dataset/threshold",
        output_dir="results/mBART_threshold_pos/test_metrics"
    )

    lang_pos = MBARTEvaluator(
        model_path="models/mBART_hybrid_lang+pos",
        dataset_path="filtered_datasets/mBART_dataset/hybrid",
        output_dir="results/mBART_hybrid_lang+pos/test_metrics"
    )

    for evaluator in [baseline, finetuned, lang, pos, lang_pos]:
        results, preds, labels = evaluator.evaluate()
