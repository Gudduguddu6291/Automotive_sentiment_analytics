import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from src.model import ABSAClassifier
from config import MODEL_NAME, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE

class ABSADataset(Dataset):
    def __init__(self, data_samples, preprocessor):
        self.samples = data_samples
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.preprocessor.tokenize_aspect_sentence_pair(item["text"], item["aspect"])
        label = 1 if item["sentiment"] == "Positive" else 0
        
        data_dict = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        if "token_type_ids" in encoding:
            data_dict["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        return data_dict

def train_absa_model(train_samples):
    """Fine-tuning loop template for downstream BERT/XLM-R ABSA classification."""
    from src.preprocessing import TextPreprocessor
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = TextPreprocessor(MODEL_NAME)
    dataset = ABSADataset(train_samples, preprocessor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = ABSAClassifier(model_name=MODEL_NAME, num_classes=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    print(f"Starting training on device: {device}")
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
                
            logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {total_loss/len(dataloader):.4f}")
        
    return model