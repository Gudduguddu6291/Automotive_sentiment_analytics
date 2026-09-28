# Automotive Sentiment Analytics

A small Python project for generating synthetic automotive reviews and analyzing sentiment for selected aspects. The example pipeline generates a dataset, splits a sample review into clauses, matches clauses to requested aspects, and classifies each matching clause as positive or negative with a pretrained Hugging Face model.

## Requirements

- Python 3.10 or newer
- PyTorch
- pandas
- Transformers

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch pandas transformers
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Run

From the project root:

```powershell
python main.py
```

The first run downloads the pretrained sentiment model and tokenizer. The pipeline writes 50 synthetic reviews to `data/raw/automotive_reviews_synthetic.json`, then prints aspect-level sentiment and confidence for a sample review. Running it again replaces that generated JSON file.

## Project layout

```text
.
├── config.py                  # Paths and model/training settings
├── main.py                    # Dataset generation and inference example
├── data/
│   └── raw/                    # Generated synthetic review data
└── src/
    ├── dataset_generator.py   # Synthetic review generation
    ├── inference.py           # Aspect matching and sentiment inference
    ├── model.py               # Custom PyTorch classifier
    ├── preprocessing.py       # Text cleanup, clause splitting, tokenization
    └── train.py               # Model training components
```

The example inference engine uses `distilbert-base-uncased-finetuned-sst-2-english`. The model name and other training settings are defined in `config.py` and the relevant source modules.