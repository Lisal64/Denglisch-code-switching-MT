# Denglisch-code-switching-MT

This repository contains all code and data processing steps used in my master's thesis on code-switched machine translation. It includes:

- Preprocessing scripts
- Fine-tuning scripts for mT5 and other models
- Evaluation utilities
- The adapted Denglisch dataset


### Installing

This Python project utilizes modern packaging standards defined in the [pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) configuration file. Project management tasks like environment setup and dependency installation are handled using [UV](https://github.com/astral-sh/uv).

### Using UV

UV is a command-line tool that needs to be installed first. You can typically install it using pip, pipx, or your system's package manager if available.

Refer to the [official UV installation guide](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) for the most up-to-date methods. A common way is:

```bash
# Using pip (ensure pip is up-to-date)
pip install uv

# Or using pipx (recommended for CLI tools)
pipx install uv
```

To install project dependencies, use:

```bash
uv sync
```

To enable the environment, use:

```bash
source .venv/bin/activate
```

### Dataset

The original dataset used as a base for the adapted dataset found in this repository, can be found under <https://github.com/HaifaCLG/Denglisch>.


### Running the scripts

To apply data filtering, navigate to the root of the project and run:
```bash
python -m data_prep.data_preprocessing
```

To conduct finetuning, navigate to the root of the project and run:

```bash
python -m finetuning.train
```

To test the finetuned models, navigate to the root of the project and run: 

```bash
python -m model_evaluation.test
```

To run mT5 finetuning, navigate to the root of the project and run: 
```bash
python -m finetuning.finetune_mt5_baseline
```

To test the finetuned mT5, naviage to the root of the project and run:
```bash
python -m model_evaluation.mt5_evaluate_baseline
```