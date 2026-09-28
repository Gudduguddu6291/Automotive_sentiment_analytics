import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from config import MODEL_NAME
from src.preprocessing import TextPreprocessor

class ABSAInferenceEngine:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initializes sequence classification pipeline for aspect clause evaluation.
        Can be switched to 'xlm-roberta-base' for multilingual datasets.
        """
        self.preprocessor = TextPreprocessor(model_name=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def analyze_review(self, review_text: str, target_aspects: list = None) -> pd.DataFrame:
        """
        Extracts sub-clauses, connects them to target aspects, and predicts sentiment.
        """
        cleaned_text = self.preprocessor.clean_text(review_text)
        clauses = self.preprocessor.split_into_clauses(cleaned_text)
        
        results = []
        for clause in clauses:
            if target_aspects:
                matched_aspects = [a for a in target_aspects if a.lower() in clause.lower()]
            else:
                matched_aspects = [clause.split()[0]]

            for aspect in matched_aspects:
                pred = self.classifier(clause)[0]
                label_map = {"POSITIVE": "Positive", "NEGATIVE": "Negative"}
                sentiment = label_map.get(pred["label"], pred["label"])
                
                results.append({
                    "Aspect": aspect,
                    "Sentiment": sentiment,
                    "Confidence": round(pred["score"], 4),
                    "Clause Context": clause
                })

        return pd.DataFrame(results)