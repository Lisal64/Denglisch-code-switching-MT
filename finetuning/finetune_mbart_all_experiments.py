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
import typing

class MBARTFineTuner:
    def __init__(self, model_name, dataset_path, strategy_name: typing.Literal["ratio", "hybrid", "threshold"], 
                 context_type: typing.Literal["baseline", "lang", "pos", "lang+pos"]):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.strategy_name = strategy_name 
        self.context_type = context_type
        self.output_dir = f"results/mBART_{strategy_name}_{context_type}"
        self.model_dir = f"models/mBART_{strategy_name}_{context_type}"

        self.tokenizer = MBartTokenizerFast.from_pretrained(model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)

        self.SRC_LANG = "de_DE"
        self.TGT_LANG = "en_XX"
        self.tokenizer.src_lang = self.SRC_LANG
        self.tokenizer.tgt_lang = self.TGT_LANG
        self.model.config.forced_bos_token_id = self.tokenizer.lang_code_to_id["en_XX"]

        self.metrics = {
            "bleu": evaluate.load("bleu"),
            "meteor": evaluate.load("meteor"),
            "sacrebleu": evaluate.load("sacrebleu"),
            "rouge": evaluate.load("rouge")
        }

    def build_input(self, example):
        base = example["input"]

        if self.context_type == "baseline":
            return base
        if self.context_type == "lang":
            lang_context = " ".join(example["langs"])
            return f"{lang_context} | {base}"
        if self.context_type == "pos":
            pos_context = " ".join(example["pos_tags"])
            return f"{pos_context} | {base}"
        if self.context_type == "lang+pos":
            lang_context = " ".join(example["langs"])
            pos_context = " ".join(example["pos_tags"])
            return f"{lang_context} <ctx> {pos_context} | {base}"
        else:
            return base
    
    def preprocess(self, examples):
        examples_list = [
            {"input": inp, "tokens": tok, "langs": lang, "pos_tags": pos}
            for inp, tok, lang, pos in zip(
                examples["input"],
                examples.get("tokens", [[]] * len(examples["input"])),
                examples.get("langs", [[]] * len(examples["input"])),
                examples.get("pos_tags", [[]] * len(examples["input"]))
            )
        ]

        inputs = [self.build_input(ex) for ex in examples_list]

        model_inputs = self.tokenizer(
            inputs,
            max_length=128,
            padding="max_length",
            truncation=True
        )

        labels = self.tokenizer(
            text_target=examples["target"],
            max_length=128,
            padding="max_length",
            truncation=True
        )["input_ids"]

        labels = [
            [(label if label != self.tokenizer.pad_token_id else -100) for label in label_seq]
            for label_seq in labels
        ]
        model_inputs["labels"] = labels
        return model_inputs

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        return {
            "bleu": self.metrics["bleu"].compute(predictions=decoded_preds, references=decoded_labels)["bleu"],
            "meteor": self.metrics["meteor"].compute(predictions=decoded_preds, references=[l[0] for l in decoded_labels])["meteor"],
            "sacrebleu": self.metrics["sacrebleu"].compute(predictions=decoded_preds, references=decoded_labels)["score"],
            "rougeL": self.metrics["rouge"].compute(predictions=decoded_preds, references=[l[0] for l in decoded_labels])["rougeL"]
        }

    def run(self):
        dataset = load_dataset("json", data_files={
            "train": f"{self.dataset_path}/train.json",
            "validation": f"{self.dataset_path}/validation.json",
            "test": f"{self.dataset_path}/test.json",
        })

        dataset = dataset.map(lambda x: self.preprocess(x), batched=True)
        dataset = dataset.remove_columns(["tokens", "langs", "pos_tags"])

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.model_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=6,
            predict_with_generate=True,
            generation_max_length=128,
            generation_num_beams=4,
            logging_dir=f"{self.model_dir}/logs",
            logging_steps=10,
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="none",
            seed=42
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        trainer.save_model(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        self.model.config.save_pretrained(self.model_dir)

        # save metrics
        os.makedirs(self.output_dir, exist_ok=True)
        val_metrics = trainer.evaluate()
        with open(f"{self.output_dir}/val_metrics.json", "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2, ensure_ascii=False)

        train_metrics = trainer.evaluate(eval_dataset=dataset["train"])
        with open(f"{self.output_dir}/train_metrics.json", "w", encoding="utf-8") as f:
            json.dump(train_metrics, f, indent=2, ensure_ascii=False)

        print(f"Finished: {self.context_type}")