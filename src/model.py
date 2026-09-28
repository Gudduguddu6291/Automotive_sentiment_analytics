import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ABSAClassifier(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 2, dropout_rate: float = 0.2):
        """
        Custom PyTorch architecture wrapper for BERT / XLM-R Aspect-Based Sentiment Classification.
        """
        super(ABSAClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] pooled representation
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits