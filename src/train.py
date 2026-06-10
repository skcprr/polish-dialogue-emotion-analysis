import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score
from data_preparation import get_prepared_datasets

MODEL_NAME = "allegro/herbert-base-cased"

class MultiLabelWeightedTrainer(Trainer):
    def __init__(self, pos_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.pos_weight is not None:
            device = logits.device
            self.pos_weight = self.pos_weight.to(device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            loss_fct = nn.BCEWithLogitsLoss()
            
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probs = 1 / (1 + np.exp(-predictions))
    preds = (probs > 0.5).astype(float)
    
    f1_macro = f1_score(y_true=labels, y_pred=preds, average='macro', zero_division=0)
    f1_micro = f1_score(y_true=labels, y_pred=preds, average='micro', zero_division=0)
    
    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro
    }

def run_training():
    train_ds, val_ds, test_ds, label_cols = get_prepared_datasets()
    num_labels = len(label_cols)
    
    labels_matrix = np.array(train_ds['labels'])
    pos_counts = labels_matrix.sum(axis=0)
    neg_counts = len(labels_matrix) - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-6)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32)
    
    print(f"Wagi obliczone dla kolejnych klas:\n{pos_weights}\n")

    print(f"Ładowanie modelu bazowego: {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

    print("Hyperparameters config")
    training_args = TrainingArguments(
        output_dir="./models/herbert-twitteremo-checkpoints", 
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
    )

    print("Training process inicialized:")
    trainer = MultiLabelWeightedTrainer(
        pos_weight=pos_weight_tensor,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("Saving best model")
    trainer.save_model("./models/best_emotion_model")
    print("Model saved")

if __name__ == "__main__":
    run_training()