import numpy as np
from datasets import load_dataset
from evaluate import load as evaluate_load
from typing import Dict
from torchvision import transforms as tv_transforms
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)
import wandb
from torch.utils.data import DataLoader, Dataset
import random
import torch
import matplotlib.pyplot as plt

wandb.init(project="vit_baseline_dog")

def load_and_preprocess_data():
    dog = load_dataset('amaye15/stanford-dogs')
    
    labels = dog['train'].features['label'].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}
    
    return dog, labels, label2id, id2label

def get_image_transforms():
    return tv_transforms.Compose([
        tv_transforms.Resize((224, 224)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    predictions = predictions[0] if isinstance(predictions, tuple) else predictions
    labels = labels[0] if isinstance(labels, tuple) else labels
    
    predictions = torch.from_numpy(predictions)
    labels = torch.from_numpy(labels)
    
    # Get the predicted class
    predicted_classes = predictions.argmax(dim=-1)
    
    # Compute accuracy
    correct = (predicted_classes == labels).float()
    accuracy = correct.mean().item()
    
    # Compute top-5 accuracy
    top5_pred = predictions.topk(5, dim=-1)[1]
    top5_correct = top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).float()
    top5_accuracy = top5_correct.sum(dim=-1).mean().item()
    
    # Compute F1 score
    metric_f1 = evaluate_load("f1")
    f1_score = metric_f1.compute(predictions=predicted_classes.numpy(), references=labels.numpy(), average="weighted")['f1']
    
    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "f1_score": f1_score
    }
    
class DOG100Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['pixel_values'].convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return {
            'pixel_values': image,
            'label': item['label']
        }
    
def main():
    dog, labels, label2id, id2label = load_and_preprocess_data()
    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    transform = get_image_transforms()
    train_dataset = DOG100Dataset(dog['train'], transform=transform)
    test_dataset = DOG100Dataset(dog['test'], transform=transform)

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    
    training_args = TrainingArguments(
        output_dir="vit_dog_cutmix",
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=28,
        save_steps=28,
        learning_rate=5e-4,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to='wandb',
        label_names=["label"] 
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()