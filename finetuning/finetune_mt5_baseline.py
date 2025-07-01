from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import os
import json

bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")
sacrebleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

model_name = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = [str(x) for x in examples["input"]]
    targets = [str(x) for x in examples["target"]]

    model_inputs = tokenizer(
        inputs,
        max_length=128,
        padding="max_length",
        truncation=True
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            padding="max_length",
            truncation=True
        )["input_ids"]

    model_inputs["labels"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels
    ]

    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):  # defensive check
        preds = preds[0]

    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

    # replace ignored tokens with proper id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds_cleaned, labels_cleaned = postprocess_text(decoded_preds, decoded_labels)

    # remove empty predictions/labels
    filtered_preds, filtered_labels = [], []
    for pred, label in zip(preds_cleaned, labels_cleaned):
        if pred.strip() and label[0].strip():
            filtered_preds.append(pred.strip())
            filtered_labels.append([label[0].strip()])

    if not filtered_preds:
        return {"bleu": 0.0, "meteor": 0.0, "sacrebleu": 0.0, "rogue": 0.0}

    print("Sample predictions:")
    for i in range(min(3, len(filtered_preds))):
        print("PRED:", filtered_preds[i])
        print("REF :", filtered_labels[i][0])
        print("---")

    bleu_result = bleu_metric.compute(predictions=filtered_preds, references=filtered_labels)
    meteor_result = meteor_metric.compute(predictions=filtered_preds, references=[ref[0] for ref in filtered_labels])
    sacrebleu_result = sacrebleu.compute(predictions=filtered_preds, references=filtered_labels)
    rouge_result = rouge.compute(predictions=filtered_preds, references=[ref[0] for ref in filtered_labels])

    return {
        "bleu": bleu_result["bleu"],
        "meteor": meteor_result["meteor"],
        "sacrebleu": sacrebleu_result["score"],
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "rougeLsum": rouge_result["rougeLsum"],
    }


dataset = load_dataset("json", data_files={
    "train": "filtered_datasets/mT5_dataset/ratio/train.json",
    "validation": "filtered_datasets/mT5_dataset/ratio/validation.json"
})
dataset = dataset.remove_columns(["tokens", "langs", "pos_tags"])  # remove extra info
raw_inputs = dataset["validation"]["input"]
print(dataset["train"].features)

print("RAW EXAMPLE:", dataset["train"][0])
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["input", "target"]) # ensures only tokenized input gets passed to model

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5_code_switch_translation",
    save_strategy="epoch",
    eval_strategy = "epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    generation_max_length=128,
    generation_num_beams=4,
    per_device_train_batch_size=4,
    label_smoothing_factor=0.1,
    per_device_eval_batch_size=4,
    remove_unused_columns=False,
    learning_rate=3e-5,
    num_train_epochs=6,
    logging_dir="./logs",
    logging_steps=10,
    predict_with_generate=True,
    seed=42,
)

model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(tokenizer.lang_code_to_id["en"] if hasattr(tokenizer, "lang_code_to_id") else tokenizer.eos_token)
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 128
model.config.num_beams = 4
model.config.repetition_penalty = 1.5
model.config.length_penalty = 1.0
model.config.no_repeat_ngram_size = 2
model.config.dropout = 0.1

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

model_dir = "models/mt5_denglisch_general_5ep_3e-5lr_finetuned"
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)
model.config.save_pretrained(model_dir)

# evaluation
eval_results = trainer.evaluate()
os.makedirs("results/mT5", exist_ok=True)
with open("results/mT5/metrics.json", "w", encoding="utf-8") as f:
    json.dump(eval_results, f, indent=2, ensure_ascii=False)

train_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["train"])
with open(f"results/mT5/train_metrics.json", "w", encoding="utf-8") as f:
    json.dump(train_metrics, f, indent=2, ensure_ascii=False)

predictions = trainer.predict(tokenized_dataset["validation"])
preds = predictions.predictions
if isinstance(preds, tuple):  # edge case: tuple (logits + cache)
    preds = preds[0]

pred_ids = predictions.predictions
if isinstance(pred_ids, tuple):
    pred_ids = pred_ids[0]
pred_ids = np.asarray(pred_ids, dtype=np.int32)
pred_ids = np.clip(pred_ids, 0, tokenizer.vocab_size - 1)

decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

label_ids = predictions.label_ids
label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
label_ids = np.clip(label_ids, 0, tokenizer.vocab_size - 1)
decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

# show a few samples
for inp, pred, ref in zip(raw_inputs[:5], decoded_preds[:5], decoded_labels[:5]):
    print("INPUT:", inp)
    print("PRED :", pred)
    print("REF  :", ref)
    print("---")