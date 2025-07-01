import os
import numpy as np
from datasets import load_dataset
from transformers import (
    MBartTokenizerFast,
    MBartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import evaluate
import json

dataset = load_dataset("json", data_files={
    "train": "filtered_datasets/mBART_dataset/ratio/train.json",
    "validation": "filtered_datasets/mBART_dataset/ratio/validation.json"
})
dataset = dataset.remove_columns(["tokens", "langs", "pos_tags"])  # remove extra info
raw_inputs = dataset["validation"]["input"]

model_name = "facebook/mbart-large-50-many-to-one-mmt"
tokenizer = MBartTokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

SRC_LANG = "de_DE"
TGT_LANG = "en_XX"
tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TGT_LANG
model.config.forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]

def preprocess(examples):
    inputs = [inp.replace("$newline$", "\n") for inp in examples["input"]]
    
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        padding="max_length",
        truncation=True
    )

    labels = tokenizer(
        text_target=examples["target"],
        max_length=128,
        padding="max_length",
        truncation=True
    )["input_ids"]

    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels
    ]
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
sacrebleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    results = {
        "bleu": bleu.compute(predictions=decoded_preds, references=decoded_labels)["bleu"],
        "meteor": meteor.compute(predictions=decoded_preds, references=[l[0] for l in decoded_labels])["meteor"],
        "sacrebleu": sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)["score"],
        "rougeL": rouge.compute(predictions=decoded_preds, references=[l[0] for l in decoded_labels])["rougeL"]
    }
    return results

training_args = Seq2SeqTrainingArguments(
    output_dir="./mbart_code_switch",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate = 3e-5, # before 5e-5
    num_train_epochs = 6, # before 10
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,
    logging_dir="./logs",
    logging_steps=10,
    seed=42,
    report_to="none",
)

tokenizer.lang_code_to_id["en_XX"]
model.config.eos_token_id = tokenizer.eos_token_id
model.config.repetition_penalty = 1.2
model.config.length_penalty = 1.1
model.config.no_repeat_ngram_size = 2

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

model_dir = "models/mbart_denglisch_general_5ep_3e-5lr_finetuned"
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)
model.config.save_pretrained(model_dir)



predictions = trainer.predict(tokenized_dataset["validation"])

pred_ids = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
label_ids = predictions.label_ids

label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
pred_ids = np.clip(pred_ids, 0, tokenizer.vocab_size - 1)
label_ids = np.clip(label_ids, 0, tokenizer.vocab_size - 1)

decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)


eval_results = trainer.evaluate()
os.makedirs("results/validation", exist_ok=True)
with open("results/validation/metrics.json", "w", encoding="utf-8") as f:
    json.dump(eval_results, f, indent=2, ensure_ascii=False)

print("Validation metrics:")
for k, v in eval_results.items():
    print(f"{k}: {v:.4f}")

train_results = trainer.evaluate(eval_dataset=tokenized_dataset["train"])
with open("results/validation/metrics.json", "a", encoding="utf-8") as f:
    f.write(",\n")
    f.write('"train_metrics": ')
    json.dump(train_results, f, indent=2, ensure_ascii=False)
    f.write("\n}")

print("Training metrics:")
for k, v in train_results.items():
    print(f"{k}: {v:.4f}")

# show a few samples
for inp, pred, ref in zip(raw_inputs[:5], decoded_preds[:5], decoded_labels[:5]):
    print("INPUT:", inp)
    print("PRED :", pred)
    print("REF  :", ref)
    print("---")