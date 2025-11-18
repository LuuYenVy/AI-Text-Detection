import torch
import torch.nn as nn
from transformers import AutoModel

class RobertaMeanPoolLayerMix(nn.Module):
    def __init__(self, model_name="roberta-base", num_labels=2, layer_start=-4, dropout=0.3):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels
        self.layer_start = layer_start
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.roberta.config.hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        all_layer_embeddings = torch.stack(hidden_states[-4:], dim=0)
        mean_hidden = all_layer_embeddings.mean(dim=0)
        mask_expanded = attention_mask.unsqueeze(-1).expand(mean_hidden.size()).float()
        sum_embeddings = torch.sum(mean_hidden * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        logits = self.classifier(self.dropout(mean_pooled))

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
