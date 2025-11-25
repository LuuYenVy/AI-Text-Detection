import torch
import torch.nn as nn
from transformers import AutoModel

class RobertaMeanPoolLayerMix(nn.Module):
    def __init__(self, model_name="roberta-base", layer_start=-4, dropout=0.4):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_start = layer_start
        self.hidden_size = self.roberta.config.hidden_size

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # tuple: (emb0, emb1, ... embN)
        # stack last layers from layer_start to end then mean
        all_layer_embeds = torch.stack(hidden_states[self.layer_start:], dim=0)  # (L, B, T, H)
        mean_hidden = all_layer_embeds.mean(dim=0)  # (B, T, H)
        # attention mask pooling
        mask_exp = attention_mask.unsqueeze(-1).expand(mean_hidden.size()).float()
        sum_embeds = torch.sum(mean_hidden * mask_exp, dim=1)  # (B, H)
        sum_mask = torch.clamp(mask_exp.sum(1), min=1e-9)
        pooled = sum_embeds / sum_mask
        pooled = self.dropout(pooled)
        return pooled
