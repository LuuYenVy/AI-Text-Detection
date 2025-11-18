import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import random
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

batch_size = 16
model_names_multi = ["roberta-base", "microsoft/deberta-base", "distilbert-base-uncased"]
roberta_name = "roberta-base"

# ===================== STATISTICAL FEATURES =====================
stopwords = set(["a","the","and","is","in","of","to","it","for"])
def text_features(texts):
    features = []
    for t in texts:
        n_chars = len(t)
        words = t.split()
        n_words = len(words)
        n_sentences = t.count(".") + t.count("!") + t.count("?")
        n_punct = sum(1 for c in t if c in ".,!?;:")
        upper_ratio = sum(1 for c in t if c.isupper()) / max(n_chars,1)
        stopword_ratio = sum(1 for w in t.lower().split() if w in stopwords) / max(n_words,1)
        digit_ratio = sum(1 for c in t if c.isdigit()) / max(n_chars,1)
        features.append([n_chars, n_words, n_sentences, n_punct, upper_ratio, stopword_ratio, digit_ratio])
    return np.array(features, dtype=np.float32)

# ===================== RobertaMeanPoolLayerMix =====================
class RobertaMeanPoolLayerMix(nn.Module):
    def __init__(self, model_name="roberta-base", layer_start=-4, dropout=0.4):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_start = layer_start
        self.hidden_size = self.roberta.config.hidden_size

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        all_layer_embeds = torch.stack(hidden_states[self.layer_start:], dim=0)
        mean_hidden = all_layer_embeds.mean(dim=0)
        mask_exp = attention_mask.unsqueeze(-1).expand(mean_hidden.size()).float()
        sum_embeds = torch.sum(mean_hidden * mask_exp, dim=1)
        sum_mask = torch.clamp(mask_exp.sum(1), min=1e-9)
        pooled = sum_embeds / sum_mask
        return self.dropout(pooled)

# ===================== EMBEDDINGS =====================
def compute_embeddings(texts, tokenizers, models, batch_size=16):
    """Multi-base frozen embeddings"""
    n = len(texts)
    all_model_embeddings = [[] for _ in range(len(models))]
    for i in tqdm(range(0, n, batch_size), desc="Embedding batches"):
        batch_texts = texts[i:i+batch_size]
        for idx, (tokenizer, model) in enumerate(zip(tokenizers, models)):
            enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:,0,:]
                all_model_embeddings[idx].append(cls.cpu().numpy())
    per_model_arrays = [np.vstack(l) for l in all_model_embeddings]
    combined = np.concatenate(per_model_arrays, axis=1)
    return combined

def compute_layer_mix_embeddings(texts, model, tokenizer, batch_size=16):
    model.eval()
    all_embeds = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            embeds = model(input_ids=input_ids, attention_mask=attention_mask)
            all_embeds.append(embeds.detach().cpu().numpy())
    return np.vstack(all_embeds)

# ===================== LOAD DATA =====================
# train_df and test_df phải có sẵn
# train_df: columns ["id", "topic", "answer", "is_cheating"]
# test_df: columns ["id", "topic", "answer"]
train_texts_full = (train_df["topic"].fillna("") + " " + train_df["answer"].fillna("")).tolist()
y = np.array(train_df["is_cheating"].astype(int).tolist())
test_texts = (test_df["topic"].fillna("") + " " + test_df["answer"].fillna("")).tolist()

# ===================== Multi-base frozen models =====================
tokenizers_multi = [AutoTokenizer.from_pretrained(name) for name in model_names_multi]
base_models_multi = []
for name in model_names_multi:
    m = AutoModel.from_pretrained(name)
    m.eval()
    for p in m.parameters():
        p.requires_grad=False
    m.to(device)
    base_models_multi.append(m)

X_train_multi = compute_embeddings(train_texts_full, tokenizers_multi, base_models_multi, batch_size=batch_size)
X_test_multi = compute_embeddings(test_texts, tokenizers_multi, base_models_multi, batch_size=batch_size)

# ===================== K-Fold Stacking Multi-base =====================
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
oof_preds_multi_lr = np.zeros((len(y), len(base_models_multi)))
oof_preds_multi_xgb = np.zeros((len(y), len(base_models_multi)))
test_preds_multi_lr = np.zeros((len(test_texts), len(base_models_multi)))
test_preds_multi_xgb = np.zeros((len(test_texts), len(base_models_multi)))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_multi)):
    print(f"--- Fold {fold+1} ---")
    X_tr, X_val = X_train_multi[train_idx], X_train_multi[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    for m_idx in range(len(base_models_multi)):
        input_dim = base_models_multi[m_idx].config.hidden_size
        X_tr_slice = X_tr[:, m_idx*input_dim:(m_idx+1)*input_dim]
        X_val_slice = X_val[:, m_idx*input_dim:(m_idx+1)*input_dim]
        X_test_slice = X_test_multi[:, m_idx*input_dim:(m_idx+1)*input_dim]

        # Logistic Regression
        clf_lr = LogisticRegression(max_iter=1000)
        clf_lr.fit(X_tr_slice, y_tr)
        oof_preds_multi_lr[val_idx, m_idx] = clf_lr.predict_proba(X_val_slice)[:,1]
        test_preds_multi_lr[:, m_idx] += clf_lr.predict_proba(X_test_slice)[:,1]/kf.n_splits

        # XGBoost
        clf_xgb = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            use_label_encoder=False, eval_metric='logloss'
        )
        clf_xgb.fit(X_tr_slice, y_tr, verbose=False)
        oof_preds_multi_xgb[val_idx, m_idx] = clf_xgb.predict_proba(X_val_slice)[:,1]
        test_preds_multi_xgb[:, m_idx] += clf_xgb.predict_proba(X_test_slice)[:,1]/kf.n_splits

# Meta model
meta_model_multi = LogisticRegression(max_iter=1000)
meta_model_multi.fit(oof_preds_multi_lr, y)
oof_meta_multi = meta_model_multi.predict_proba(oof_preds_multi_lr)[:,1]
test_meta_multi = meta_model_multi.predict_proba(test_preds_multi_lr)[:,1]

# ===================== Roberta Layer-Mix models =====================
tokenizer_roberta = AutoTokenizer.from_pretrained(roberta_name)
roberta_a = RobertaMeanPoolLayerMix(model_name=roberta_name, layer_start=-4, dropout=0.3).to(device)
roberta_b = RobertaMeanPoolLayerMix(model_name=roberta_name, layer_start=-8, dropout=0.3).to(device)

X_train_embed_a = compute_layer_mix_embeddings(train_texts_full, roberta_a, tokenizer_roberta, batch_size=batch_size)
X_test_embed_a = compute_layer_mix_embeddings(test_texts, roberta_a, tokenizer_roberta, batch_size=batch_size)
X_train_embed_b = compute_layer_mix_embeddings(train_texts_full, roberta_b, tokenizer_roberta, batch_size=batch_size)
X_test_embed_b = compute_layer_mix_embeddings(test_texts, roberta_b, tokenizer_roberta, batch_size=batch_size)

X_train_feat = text_features(train_texts_full)
X_test_feat = text_features(test_texts)

X_train_mix = np.hstack([X_train_embed_a, X_train_embed_b, X_train_feat])
X_test_mix = np.hstack([X_test_embed_a, X_test_embed_b, X_test_feat])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_mix)
X_test_scaled = scaler.transform(X_test_mix)

# K-Fold stacking for Layer-Mix
kf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
oof_preds_lr = np.zeros(len(y))
test_preds_lr = np.zeros(len(test_texts))
oof_preds_xgb = np.zeros(len(y))
test_preds_xgb = np.zeros(len(test_texts))

for fold, (train_idx, val_idx) in enumerate(kf2.split(X_train_scaled, y)):
    print(f"--- Fold {fold+1} ---")
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Logistic
    clf_lr = LogisticRegression(max_iter=2000)
    clf_lr.fit(X_tr, y_tr)
    oof_preds_lr[val_idx] = clf_lr.predict_proba(X_val)[:,1]
    test_preds_lr += clf_lr.predict_proba(X_test_scaled)[:,1]/kf2.n_splits

    # XGB
    clf_xgb = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric='auc', random_state=seed
    )
    clf_xgb.fit(X_tr, y_tr, verbose=False)
    oof_preds_xgb[val_idx] = clf_xgb.predict_proba(X_val)[:,1]
    test_preds_xgb += clf_xgb.predict_proba(X_test_scaled)[:,1]/kf2.n_splits

# Meta-level stacking + Isotonic Calibration
X_meta_oof = np.vstack([oof_preds_lr, oof_preds_xgb]).T
X_meta_test = np.vstack([test_preds_lr, test_preds_xgb]).T
meta_clf = LogisticRegression(max_iter=2000)
meta_clf.fit(X_meta_oof, y)
oof_meta_pred = meta_clf.predict_proba(X_meta_oof)[:,1]
test_meta_pred = meta_clf.predict_proba(X_meta_test)[:,1]

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(oof_meta_pred, y)
test_meta_cal = iso.predict(test_meta_pred)

pd.DataFrame({"id": test_df["id"], "is_cheating": test_meta_cal}).to_csv("submission_stacked_calibrated.csv", index=False)
print("Saved submission_stacked_calibrated.csv ✅")

# ===================== SIMPLE BLENDING =====================
sub1 = pd.DataFrame({"id": test_df["id"], "is_cheating": test_meta_multi})
sub2 = pd.DataFrame({"id": test_df["id"], "is_cheating": test_meta_cal})

p1 = sub1["is_cheating"].values
p2 = sub2["is_cheating"].values

final_pred = 0.5*p1 + 0.5*p2

pd.DataFrame({"id": test_df["id"], "is_cheating": final_pred}).to_csv("submission_blend_simple.csv", index=False)
print("Saved submission_blend_simple.csv ✅")
