from src.dataset_generator import generate_dataset
from src.inference import ABSAInferenceEngine
from config import RAW_DATA_PATH

def run_pipeline():
    # 1. Generate Synthetic Dataset
    print("==================================================")
    print("Stage 1: Generating Synthetic Automotive Dataset")
    print("==================================================")
    generate_dataset(num_samples=50, output_path=RAW_DATA_PATH)

    # 2. Initialize Inference Engine & Run Example
    print("\n==================================================")
    print("Stage 2: Running Aspect Extraction & Sentiment Analytics")
    print("==================================================")
    engine = ABSAInferenceEngine()
    
    sample_review = "The battery range is excellent, but the charging time is too long."
    target_aspects = ["battery range", "charging time"]
    
    print(f"\nRaw Review Input:\n\"{sample_review}\"\n")
    output_df = engine.analyze_review(sample_review, target_aspects=target_aspects)
    
    print("--- Aspect-Level Sentiment Output ---")
    print(output_df[["Aspect", "Sentiment", "Confidence"]].to_string(index=False))

if __name__ == "__main__":
    run_pipeline()