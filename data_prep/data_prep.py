from datasets import Dataset, DatasetDict
from .data_filtering import GermanPresenceFilter
from typing import List, Dict
import csv
from tqdm import tqdm
import ast
from sklearn.model_selection import train_test_split

def clean_tokens(tokens: List[str]) -> List[str]:
    return [
        t.replace("$newline$", "\n") if t == "$newline$" else
        '"' if t == "$quote$" else
        t for t in tokens
        if t not in {"<EOP>", "<punct>"}
    ]

def prepare_hf_mt5_dataset(
    input_csv_path: str,
    strategy: str = "hybrid",
    min_ratio: float = 0.3,
    min_count: int = 2,
    task_prefix: str = "Translate this mixed English-German sentence to English: ",
    keep_metadata: bool = True
) -> Dataset:
    filter = GermanPresenceFilter(strategy, min_ratio, min_count)
    records: List[Dict[str, str]] = []

    with open(input_csv_path, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in tqdm(reader, desc="Filtering rows"):
            tokens = ast.literal_eval(row["tokens"])
            lang_ids = ast.literal_eval(row["langs"])
            pos_tags = ast.literal_eval(row["pos_tags"])
            target = row["target_sentence"]

            # get DE token count and ratio
            de_count = filter._count_de_tokens(lang_ids)
            ratio = de_count / len(lang_ids) if lang_ids else 0.0
            passes_filter = filter.apply(lang_ids)

            if passes_filter:
                input_text = task_prefix + " ".join(tokens)
                record = {
                    "input": input_text,
                    "target": target,
                }
                print(f"Accepted row: langs={lang_ids}, de_count={de_count}, ratio={ratio:.2f}")
                if keep_metadata:
                    record.update({
                        "tokens": tokens,
                        "langs": lang_ids,
                        "pos_tags": pos_tags
                    })
                records.append(record)
            else:
                print(f"Rejected row: langs={lang_ids}, de_count={de_count}, ratio={ratio:.2f}")
    test_size = 0.1
    val_size = 0.1
    seed = 42
    train_val, test = train_test_split(records, test_size=test_size, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=seed)

    return DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test)
    })


def prepare_hf_mbart_dataset(
    input_csv_path: str,
    strategy: str = "hybrid",
    min_ratio: float = 0.3,
    min_count: int = 2,
    keep_metadata: bool = True
) -> Dataset:
    filter = GermanPresenceFilter(strategy, min_ratio, min_count)
    records: List[Dict[str, str]] = []

    with open(input_csv_path, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in tqdm(reader, desc="Filtering rows"):
            tokens = ast.literal_eval(row["tokens"])
            tokens = clean_tokens(tokens)
            lang_ids = ast.literal_eval(row["langs"])
            pos_tags = ast.literal_eval(row["pos_tags"])
            target = row["target_sentence"]

            # get DE token count and ratio
            de_count = filter._count_de_tokens(lang_ids)
            ratio = de_count / len(lang_ids) if lang_ids else 0.0
            passes_filter = filter.apply(lang_ids)

            if passes_filter:
                input_text = " ".join(tokens)
                record = {
                    "input": input_text,
                    "target": target,
                }
                print(f"Accepted row: langs={lang_ids}, de_count={de_count}, ratio={ratio:.2f}")
                if keep_metadata:
                    record.update({
                        "tokens": tokens,
                        "langs": lang_ids,
                        "pos_tags": pos_tags
                    })
                records.append(record)
            else:
                print(f"Rejected row: langs={lang_ids}, de_count={de_count}, ratio={ratio:.2f}")
    test_size = 0.1
    val_size = 0.1
    seed = 42
    train_val, test = train_test_split(records, test_size=test_size, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=seed)

    return DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test)
    })