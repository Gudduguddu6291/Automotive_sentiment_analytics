import sys
import torch
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing import TextPreprocessor

class ABSAPredictor:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initializes tokenizer, model, and preprocessor.
        Automates target aspect extraction using noun-phrase heuristics.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = TextPreprocessor(model_name=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Known domain aspect keywords to match automatically
        self.domain_aspects = [
            "battery range", "charging time", "charging speed", "battery",
            "engine power", "engine", "acceleration", "fuel economy", "mileage",
            "cabin comfort", "comfort", "safety features", "safety", "brakes",
            "touchscreen", "infotainment", "audio system", "price", "cost",
            "dealer service", "service", "build quality", "vehicle"
        ]

    def _auto_extract_aspects(self, clause: str) -> list:
        """
        Automatically identifies target aspects within a clause context.
        """
        clause_lower = clause.lower()
        extracted = [aspect for aspect in self.domain_aspects if aspect in clause_lower]
        
        # Fallback: If no predefined aspect matches, extract the primary subject/noun
        if not extracted:
            words = [w.strip() for w in clause.split() if len(w) > 3]
            if words:
                extracted = [f"{words[0]} {words[1]}" if len(words) > 1 else words[0]]
                
        return extracted

    def predict(self, review_text: str) -> pd.DataFrame:
        cleaned_text = self.preprocessor.clean_text(review_text)
        clauses = self.preprocessor.split_into_clauses(cleaned_text)
        
        results = []
        for clause in clauses:
            # Auto-extract target aspects directly from the clause
            extracted_aspects = self._auto_extract_aspects(clause)
            
            for aspect in extracted_aspects:
                inputs = self.tokenizer(
                    clause, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = F.softmax(logits, dim=-1)[0]
                
                pred_idx = torch.argmax(probabilities).item()
                confidence = probabilities[pred_idx].item()
                
                labels = ["Negative", "Positive"] if self.model.config.num_labels == 2 else ["Negative", "Neutral", "Positive"]
                predicted_label = labels[pred_idx]
                
                results.append({
                    "Aspect": aspect.title(),
                    "Sentiment": predicted_label,
                    "Confidence": round(confidence, 4)
                })

        return pd.DataFrame(results)

if __name__ == "__main__":
    predictor = ABSAPredictor()
    
    print("\n" + "="*50)
    print("   AI Automotive Review Sentiment Predictor")
    print("="*50 + "\n")

    # 1. Input sentence only
    review_input = input("Enter your review sentence:\n> ")

    if not review_input.strip():
        print("\n[Error] Review sentence cannot be empty!")
        sys.exit(1)

    # 2. Automatically extract aspects & predict sentiment
    df_result = predictor.predict(review_input)

    print("\n--- Aspect-Level Sentiment Output ---")
    if not df_result.empty:
        print(df_result.to_string(index=False))
    else:
        print("No aspect targets could be identified from the review input.")