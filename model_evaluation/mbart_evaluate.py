import os
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    MBartTokenizerFast,
    MBartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate


class MBARTEvaluator:
    def __init__(self, model_path, dataset_path, output_dir,
                 src_lang="de_DE", tgt_lang="en_XX"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.tokenizer = MBartTokenizerFast.from_pretrained(model_path)
        self.model = MBartForConditionalGeneration.from_pretrained(model_path)

        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang
        self.model.config.forced_bos_token_id = self.tokenizer.lang_code_to_id[self.tgt_lang]

        self.metrics = {
            "bleu": evaluate.load("bleu"),
            "meteor": evaluate.load("meteor"),
            "sacrebleu": evaluate.load("sacrebleu"),
            "rouge": evaluate.load("rouge")
        }

    def preprocess(self, example):
        model_input = self.tokenizer(
            example["input"],
            max_length=128,
            padding="max_length",
            truncation=True
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                example["target"],
                max_length=128,
                padding="max_length",
                truncation=True
            )["input_ids"]
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]
        model_input["labels"] = labels
        return model_input

    def evaluate(self, params=None):
        dataset = load_dataset("json", data_files={"test": f"{self.dataset_path}/test.json"})["test"]
        dataset = dataset.map(self.preprocess)

        batch_size = 4
        num_beams = 4
        generation_max_length = 128

        if params:
            batch_size = params.get("batch_size", batch_size)
            num_beams = params.get("num_beams", num_beams)
            generation_max_length = params.get("generation_max_length", generation_max_length)

            if "dropout" in params:
                self.model.config.dropout = params["dropout"]

        args = Seq2SeqTrainingArguments(
            output_dir="./tmp_eval",  # Temporary directory
            per_device_eval_batch_size=batch_size,
            predict_with_generate=True,
            generation_max_length=generation_max_length,
            generation_num_beams=num_beams,
            report_to="none"
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        )

        predictions = trainer.predict(dataset)
        pred_ids = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
        label_ids = predictions.label_ids
        label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        results = {
            "bleu": self.metrics["bleu"].compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])["bleu"],
            "meteor": self.metrics["meteor"].compute(predictions=decoded_preds, references=decoded_labels)["meteor"],
            "sacrebleu": self.metrics["sacrebleu"].compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])["score"],
            "rougeL": self.metrics["rouge"].compute(predictions=decoded_preds, references=decoded_labels)["rougeL"]
        }

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        aligned = []
        for inp, pred, label in zip(dataset["input"], decoded_preds, decoded_labels):
            aligned.append({"input": inp, "label": label, "prediction": pred})

        with open(os.path.join(self.output_dir, "test_predictions.jsonl"), "w", encoding="utf-8") as f:
            for entry in aligned:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print("Test Set Evaluation:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")

        return results, decoded_preds, decoded_labels
