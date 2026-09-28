from pathlib import Path

# Repository Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "automotive_reviews_synthetic.json"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Core Model Hyperparameters & Architecture Choices
# Set to "xlm-roberta-base" for multilingual support or "bert-base-uncased" for monolingual English
MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5

# Specified Aspect Categories
ASPECT_CATEGORIES = [
    "Vehicle",
    "Engine",
    "Battery",
    "Mileage",
    "Safety",
    "Comfort",
    "Service",
    "Infotainment",
    "Price",
]