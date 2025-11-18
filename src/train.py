import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from src.models.roberta_layer_mix import RobertaMeanPoolLayerMix
from src.data_utils import Dataset
from src.metrics import compute_metrics

def train_model(train_dataset, val_dataset, model_name="roberta-base", output_dir="./results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaMeanPoolLayerMix(model_name=model_name).to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.1,
        report_to="none",
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    return trainer, model
