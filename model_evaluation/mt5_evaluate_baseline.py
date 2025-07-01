import os
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate


def preprocess(example, tokenizer):
    inputs = tokenizer(
        example["input"],
        max_length=128,
        padding="max_length",
        truncation=True
    )
    labels = tokenizer(
        example["target"],
        max_length=128,
        padding="max_length",
        truncation=True
    )["input_ids"]
    labels = [(token if token != tokenizer.pad_token_id else -100) for token in labels]
    inputs["labels"] = labels
    return inputs


def evaluate_mt5(model_dir, dataset_path, output_dir):
    tokenizer = MT5Tokenizer.from_pretrained(model_dir)
    model = MT5ForConditionalGeneration.from_pretrained(model_dir)

    metrics = {
        "bleu": evaluate.load("bleu"),
        "meteor": evaluate.load("meteor"),
        "sacrebleu": evaluate.load("sacrebleu"),
        "rouge": evaluate.load("rouge")
    }

    raw_dataset = load_dataset("json", data_files={"test": f"{dataset_path}/test.json"})["test"]
    dataset = raw_dataset.map(lambda x: preprocess(x, tokenizer))

    args = Seq2SeqTrainingArguments(
        output_dir="./tmp_mt5_eval",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    )

    predictions = trainer.predict(dataset)
    pred_ids = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
    label_ids = np.where(predictions.label_ids != -100, predictions.label_ids, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    results = {
        "bleu": metrics["bleu"].compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])["bleu"],
        "meteor": metrics["meteor"].compute(predictions=decoded_preds, references=decoded_labels)["meteor"],
        "sacrebleu": metrics["sacrebleu"].compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])["score"],
        "rougeL": metrics["rouge"].compute(predictions=decoded_preds, references=decoded_labels)["rougeL"]
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "test_predictions.jsonl"), "w", encoding="utf-8") as f:
        for inp, pred, ref in zip(raw_dataset["input"], decoded_preds, decoded_labels):
            f.write(json.dumps({"input": inp, "label": ref, "prediction": pred}, ensure_ascii=False) + "\n")

    print("Test Set Evaluation:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results, decoded_preds, decoded_labels


if __name__ == "__main__":
    evaluate_mt5(
        model_dir="models/mt5_denglisch_general_5ep_3e-5lr_finetuned",
        dataset_path="filtered_datasets/mT5_dataset/ratio",
        output_dir="results/mT5/test_metrics"
    )
