import re
from typing import Dict, Any, List
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_SEQ_LENGTH

class TextPreprocessor:
    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def clean_text(self, text: str) -> str:
        """Removes extra spaces and normalizes text."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize_aspect_sentence_pair(self, text: str, aspect: str) -> Dict[str, Any]:
        """
        Tokenizes review text paired with a target aspect for Sentence-Pair classification.
        Format: [CLS] aspect [SEP] context text [SEP]
        """
        cleaned_text = self.clean_text(text)
        encoding = self.tokenizer(
            text=aspect,
            text_pair=cleaned_text,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        return encoding

    def split_into_clauses(self, text: str) -> List[str]:
        """Splits sentences across coordinate conjunctions and punctuation."""
        cleaned = self.clean_text(text)
        clauses = [
            c.strip() 
            for c in re.split(r"[,;]|\bbut\b|\band\b", cleaned, flags=re.IGNORECASE) 
            if c.strip()
        ]
        return clauses