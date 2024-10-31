import numpy as np
from datasets import load_dataset
from evaluate import load as evaluate_load
from typing import Dict
import os
from torch.utils.data import ConcatDataset
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

wandb.init(project="dog_augumentation")

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

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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

class MixUpCollator(DefaultDataCollator):
    def __init__(self, mixup_prob=0.5, mixup_alpha=1.0):
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha

    def __call__(self, features, is_train=True):
        batch = {}
        pixel_values = torch.stack([f['pixel_values'] for f in features])
        labels = torch.tensor([f['label'] for f in features])

        if is_train and random.random() < self.mixup_prob:
            pixel_values, labels_a, labels_b, lam = mixup_data(pixel_values, labels, self.mixup_alpha)
            batch['pixel_values'] = pixel_values
            batch['labels_a'] = labels_a
            batch['labels_b'] = labels_b
            batch['lam'] = lam
        else:
            batch['pixel_values'] = pixel_values
            batch['label'] = labels
        
        return batch

class MixUpTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        pixel_values = inputs['pixel_values']
        labels = inputs.get('label', None)
        
        outputs = model(pixel_values)
        loss_fct = torch.nn.CrossEntropyLoss()
        
        if self.model.training and 'labels_a' in inputs and 'labels_b' in inputs and 'lam' in inputs:
            labels_a = inputs['labels_a']
            labels_b = inputs['labels_b']
            lam = inputs['lam']
            loss = mixup_criterion(loss_fct, outputs.logits, labels_a, labels_b, lam)
        else:
            loss = loss_fct(outputs.logits, labels)
        
        return (loss, outputs) if return_outputs else loss


class GeneratedDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images, self.labels = self._load_generated_data()

    def _load_generated_data(self):
        images, labels = [], []
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            label = int(class_dir.split("_")[-1]) 
            for npy_file in os.listdir(class_path):
                if '.png' in npy_file:
                    continue
                npy_path = os.path.join(class_path, npy_file)
                img = np.load(npy_path, allow_pickle=True)  
                images.append(img)
                labels.append(label)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = torch.tensor(img).float()  # Convert to tensor

        if self.transform:
            img = self.transform(img)

        return {
            'pixel_values': img,
            'label': label
        }

def load_generated_data(transform=None, generated_data_dir="./gan_images"):
    return GeneratedDataset(generated_data_dir, transform=transform)

def main():
    dog, labels, label2id, id2label = load_and_preprocess_data()
    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    transform = get_image_transforms()
    train_dataset = DOG100Dataset(dog['train'], transform=transform)
    test_dataset = DOG100Dataset(dog['test'], transform=transform)
    generated_dataset = load_generated_data()
    train_dataset = ConcatDataset([train_dataset, generated_dataset])

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    
    training_args = TrainingArguments(
        output_dir="vit_dog_gan_mixup",
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
    
    trainer = MixUpTrainer(
        model=model,
        args=training_args,
        data_collator=lambda features: MixUpCollator(mixup_prob=0.5, mixup_alpha=1.0)(features, is_train=model.training),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()