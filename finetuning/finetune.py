from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import os 
import json

bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input"], max_length=128, padding="max_length", truncation=True
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"], max_length=128, padding="max_length", truncation=True # change to max_length if bleu isn't better than 0.18
        )["input_ids"]

    # Replace pad token id with -100 for label loss masking
    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label_seq]
        for label_seq in labels
    ]
    print("First label example:", labels[0])
    print("Decoded target:", tokenizer.decode(
        [token if token != -100 else tokenizer.pad_token_id for token in labels[0]],
        skip_special_tokens=True
    ))
    model_inputs["labels"] = labels
    return model_inputs



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Ensure we’re working with token IDs
    if isinstance(preds, tuple):
        preds = preds[0]

    # Check type and clip invalid values before decoding
    preds = np.asarray(preds)
    if np.issubdtype(preds.dtype, np.floating):
        preds = np.round(preds).astype(np.int32)

    # Clip to valid token range
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds, labels = postprocess_text(decoded_preds, decoded_labels)
    bleu_result = bleu_metric.compute(predictions=preds, references=labels)
    meteor_result = meteor_metric.compute(predictions=preds, references=[l[0] for l in labels])

    return {"bleu": bleu_result["bleu"], "meteor": meteor_result["meteor"]}



model_name = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

dataset = load_dataset("json", data_files={
    "train": "filtered_datasets/dataset/ratio/train.json",
    "validation": "filtered_datasets/dataset/ratio/validation.json"
})
dataset = dataset.remove_columns(["tokens", "langs", "pos_tags"]) # generic fine-tuning for now
print("🔥 RAW EXAMPLE:", dataset["train"][0])
tokenized_dataset = dataset.map(preprocess_function, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5_code_switch_translation",
    generation_max_length = 128,
    generation_num_beams = 4,
    label_smoothing_factor = 0.0,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=3e-5, #try 5e-5
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    predict_with_generate=True,
    fp16=False,  # set True if using compatible GPU
    seed=42,
)

trainer = Seq2SeqTrainer(
    model=model,
    data_collator = data_collator,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics = compute_metrics
)

frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
print("Frozen layers:", frozen)

trainer.train()

trainer.save_model("models/mt5_denglisch_general_5ep_3e-5lr_finetuned")
tokenizer.save_pretrained("models/mt5_denglisch_general_5ep_3e-5lr_finetuned")

eval_results = trainer.evaluate()
results_dir = "results/validation"
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(eval_results, f, indent = 2, ensure_ascii=False)

predictions = trainer.predict(tokenized_dataset["validation"])
preds = predictions.predictions
if isinstance(preds, tuple):  # edge case: tuple (logits + cache)
    preds = preds[0]

preds = np.asarray(preds)
if np.issubdtype(preds.dtype, np.floating):
    preds = np.round(preds).astype(np.int32)
preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

labels = predictions.label_ids
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

for pred, ref in zip(decoded_preds[:5], decoded_labels[:5]):
    print("PRED:", pred)
    print("REF: ", ref)
    print("---")