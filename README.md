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

## Run the demo pipeline

From the project root:

```powershell
python main.py
```

The pipeline writes 50 synthetic reviews to `data/raw/automotive_reviews_synthetic.json`, then analyzes a sample review for the explicitly supplied aspects `battery range` and `charging time`. It prints each aspect's sentiment and confidence. Running it again replaces the generated JSON file.

## Run the interactive predictor

To enter your own review and let the predictor identify matching automotive aspects:

```powershell
python predict.py
```

The predictor prints aspect-level sentiment and confidence, or a message if it cannot identify any aspects. Both inference entry points download the pretrained `distilbert-base-uncased-finetuned-sst-2-english` model and tokenizer on first use. An internet connection is required for that initial download.

## Training

`src/train.py` contains a fine-tuning function for labeled aspect/sentiment samples, but there is no training command or saved-checkpoint workflow yet. `config.py` defines the training model (`bert-base-uncased`) and its hyperparameters; the demo and interactive predictor use the pretrained sentiment model described above.

## Project layout

```text
.
├── config.py                  # Paths and model/training settings
├── main.py                    # Dataset generation and inference example
├── predict.py                 # Interactive aspect sentiment predictor
├── data/
│   └── raw/                   # Generated synthetic review data
└── src/
    ├── __init__.py
    ├── dataset_generator.py   # Synthetic review generation
    ├── inference.py           # Aspect matching and sentiment inference
    ├── model.py               # Custom PyTorch classifier
    ├── preprocessing.py       # Text cleanup, clause splitting, tokenization
    └── train.py               # Model training components
```
