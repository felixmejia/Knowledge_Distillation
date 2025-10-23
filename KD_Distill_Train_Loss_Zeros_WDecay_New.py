from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from torch.utils.data import Subset

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import time
from tqdm import tqdm
import psutil
import argparse
import sys
from io import StringIO

from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as F

from losses import AsymmetricLoss


from torchmetrics.classification import MultilabelAccuracy


from torchinfo import summary

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("CodeCarbon not available. Falling back to estimated energy consumption.")

# Custom console logger
class ConsoleLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
#{
# PATHOL7OGIES = ['Atelectasis', 'Cardiomegaly']

#### python3 KD_Distill_Train_Loss_Ones_WDecay.py --resume --benchmark --checkpoint_dir KD_Main_Thrreshold_Ones_20250902-023228/checkpoints/

def multihot(labels_df, classes):
    """
    Convierte un DataFrame de etiquetas de CheXpert a formato multi-hot.

    Esta función implementa la política "U-Ones", donde las etiquetas
    "Incierto" (-1.0) se tratan como positivas (1), al igual que las
    "Presente" (1.0). "Ausente" (0.0) y "No Mencionado" (NaN) se
    tratan como negativas (0).

    Args:
        labels_df (pd.DataFrame or pd.Series): DataFrame o Serie con las
                                               etiquetas originales.
        classes (list): Lista de los nombres de las columnas de patologías.

    Returns:
        np.array: Un array de NumPy con las etiquetas en formato multi-hot
                  (0s y 1s).
    """
    # Asegurarse de que estamos trabajando con una copia
    y = labels_df[classes].copy()

    # Llenar NaNs (Not Mentioned) con 0 (Negativo)
    y = y.fillna(0.0)

    # Reemplazar -1.0 (Uncertain) con 1 (Positivo) - Política U-Ones
    y = y.replace(-1.0, 0.0)

    # Asegurarse de que todo sea entero (0 o 1)
    y = y.astype(float)

    print(f'multihot={labels_df}')

    # Convertir a un array de NumPy
    return y.values


class CheXpertDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, ones_strategy=True, fraction=1.0):
        self.data = pd.read_csv(csv_file)
        if fraction < 1.0:
            self.data = self.data.sample(frac=fraction, random_state=42).reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.ones_strategy = ones_strategy
        self.labels = self._process_labels()
        self.pos_weight = self._compute_pos_weight()

    def _process_labels(self):
        labels = self.data[PATHOLOGIES].copy()
        if self.ones_strategy:
            labels = labels.fillna(0)
            labels = labels.replace(-1, 0)

        if (len(np.unique(labels)) > 1):
            self.labels = multihot(labels, PATHOLOGIES)
        else:
            self.labels = labels
        return labels.values.astype(np.float32)

    def _compute_pos_weight(self):
        labels = self._process_labels()
        pos_weight = []
        for i, pathology in enumerate(PATHOLOGIES):
            positive = np.sum(labels[:, i] == 1)
            negative = len(labels) - positive
            if positive == 0:
                pos_weight.append(1.0)
                logging.warning(f"No positive samples for {pathology}. Using pos_weight=1.0.")
            else:
                pos_weight.append(negative / positive)
        return np.array(pos_weight, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]['Path'])
        image = plt.imread(img_path)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def _replace_first_conv_to_1ch(mobilenet_v2_model: nn.Module):
    """Convierte la primera conv 3→1 canal promediando pesos RGB."""
    for parent_name, parent in mobilenet_v2_model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                new_conv = nn.Conv2d(
                    1, child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    padding_mode=child.padding_mode
                )
                with torch.no_grad():
                    new_conv.weight.copy_(child.weight.mean(dim=1, keepdim=True))
                    if child.bias is not None:
                        new_conv.bias.copy_(child.bias)
                setattr(parent, child_name, new_conv)
                return
    raise RuntimeError("No se encontró una Conv2d 3→? para convertir a 1 canal.")

class StudentMobileNetV2(nn.Module):
    """
    MobileNetV2 sin congelar (toda la red entrenable), multilabel para 5 patologías.
    - width_mult: 0.35, 0.5, 0.75, 1.0... (más bajo = más ligero).
    - grayscale_mode:
        "replicate": DataLoader replica a 3 canales (mantiene la primera conv en 3ch).
        "1ch": convierte la primera conv a 1 canal (si alimentas tensores [B,1,H,W]).
    """
    def __init__(self, num_classes=5, dropout=0.2, width_mult=1.0,
                 grayscale_mode="replicate", pretrained=True):
        super().__init__()
        try:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.mobilenet_v2(weights=weights, width_mult=width_mult)
        except Exception:
            base = models.mobilenet_v2(pretrained=pretrained)  # compat antiguas

        if grayscale_mode.lower() == "1ch":
            _replace_first_conv_to_1ch(base)
        elif grayscale_mode.lower() == "replicate":
            pass
        else:
            raise ValueError("grayscale_mode debe ser 'replicate' o '1ch'.")

        # Reemplaza la cabeza: Dropout + Linear → num_classes (SIN Sigmoid; usar BCEWithLogitsLoss)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

        # Nada congelado: todo entrenable
        for p in base.parameters():
            p.requires_grad = True

        self.model = base

    def forward(self, x):
        return self.model(x)
    

class LightweightCNNStudent(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


# =============== LOSS FUNCIONES ===============
def kd_loss_fn(student_logits, teacher_logits, targets, alpha=0.7, temperature=4.0):
    kd_loss = nn.KLDivLoss(reduction='batchmean')( 
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1)
    ) * (alpha * temperature * temperature)

    ce_loss = F.binary_cross_entropy_with_logits(student_logits, targets)
    return kd_loss + (1. - alpha) * ce_loss


class WeightedFocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=2.0, reduction='mean', eps=1e-8):
        """
        Implementa Weighted Focal Loss para clasificación binaria o multilabel.
        
        Args:
            weights (Tensor, opcional): tensor de pesos por clase, tamaño [n_labels].
            gamma (float): parámetro de focal loss, controla el enfoque en ejemplos difíciles.
            reduction (str): 'mean', 'sum' o 'none'.
            eps (float): constante para estabilidad numérica.
        """
        super(WeightedFocalLoss, self).__init__()
        self.register_buffer('weights', None)
        if weights is not None:
            self.register_buffer('weights', torch.tensor(weights, dtype=torch.float))
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits sin activar (shape [batch_size, n_labels])
            targets: etiquetas binarias (shape [batch_size, n_labels])
        """
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)   # prob. predicha correcta
        focal_term = (1 - pt) ** self.gamma
        log_pt = torch.log(pt + self.eps)

        loss = -focal_term * log_pt  # focal loss básica

        if self.weights is not None:
            loss = loss * self.weights  # aplica pesos por clase

        # Reducción
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# Ejemplo de uso:
# logits = modelo(x)   # salida [batch, n_labels]
# labels = ...         # ground truth [batch, n_labels]
# class_weights = torch.tensor([1.0, 2.0, 0.5])  # pesos por clase
# loss = weighted_focal_loss(logits, labels, weights=class_weights, gamma=2)


class DistillationLoss_NN(nn.Module):
    def __init__(self, alpha=0.5, distillation_mode='soft'):
        """
        Knowledge Distillation Loss for multi-label classification.
        
        Args:
            alpha (float): Weight for balancing BCE loss and distillation loss (0 <= alpha <= 1).
            distillation_mode (str): Type of distillation, either 'soft' or 'hard'.
        """
        super(DistillationLoss_NN, self).__init__()
        self.alpha = alpha
        self.distillation_mode = distillation_mode
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, teacher_logits, true_labels):
        """
        Compute the total loss combining BCEWithLogitsLoss and distillation loss.
        
        Args:
            student_logits (torch.Tensor): Logits from the student model (batch_size, num_classes).
            teacher_logits (torch.Tensor): Logits from the teacher model (batch_size, num_classes).
            true_labels (torch.Tensor): Ground truth labels (batch_size, num_classes).
        
        Returns:
            torch.Tensor: Total loss.
        """
        # Base BCE loss with true labels
        bce_loss = self.bce_with_logits(student_logits, true_labels)

        # Teacher probabilities (sigmoid of teacher logits)
        teacher_probs = torch.sigmoid(teacher_logits)

        if self.distillation_mode == 'soft':
            # Soft distillation: BCEWithLogitsLoss between student logits and teacher probabilities
            distill_loss = self.bce_with_logits(student_logits, teacher_probs)
        else:  # hard distillation
            # Hard distillation: Binarize teacher probabilities at 0.5
            teacher_pseudo_labels = (teacher_probs > 0.5).float()
            distill_loss = self.bce_with_logits(student_logits, teacher_pseudo_labels)

        # Total loss: weighted combination of BCE loss and distillation loss
        total_loss = (1 - self.alpha) * bce_loss + self.alpha * distill_loss
        return total_loss
    

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.3, temperature=4.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        hard_loss = self.ce_loss(student_logits, labels)
        teacher_probs = torch.sigmoid(teacher_logits / self.temperature)
        student_probs = torch.sigmoid(student_logits / self.temperature)
        soft_loss = self.kl_div(torch.log_softmax(student_probs, dim=1), 
                               torch.softmax(teacher_probs, dim=1)) * (self.temperature ** 2)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss




class MultiLabelDistillLoss(nn.Module):
    """
    KD para multilabel: BCEWithLogits (dura) + KL Bernoulli entre sigmoides templadas (suave).
    hard: BCEWithLogitsLoss(pos_weight=...)
    soft: mean over classes of KL(q_t || q_s), con q = sigmoid(logits / T), escalado por T^2.
    """
    def __init__(self, alpha=0.3, temperature=4.0, pos_weight=None, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # balanceo de positivos, si se desea

    @staticmethod
    def _binary_kl(q_t, q_s, eps=1e-7):
        # KL Bernoulli por elemento; promediaremos luego sobre batch y clases
        q_t = q_t.clamp(eps, 1 - eps)
        q_s = q_s.clamp(eps, 1 - eps)
        return q_t * (q_t / q_s).log() + (1 - q_t) * ((1 - q_t) / (1 - q_s)).log()

    def forward(self, student_logits, teacher_logits, labels):
        # 1) Pérdida dura (BCE con logits)
        hard = self.bce(student_logits, labels)

        # 2) Pérdida suave (sigmoide templada + KL Bernoulli por clase)
        q_t = torch.sigmoid(teacher_logits / self.T)
        q_s = torch.sigmoid(student_logits / self.T)
        kl_per_elem = self._binary_kl(q_t, q_s, self.eps)
        # promedio sobre batch y clases
        soft = kl_per_elem.mean() * (self.T ** 2)

        # 3) Combinación
        return self.alpha * hard + (1 - self.alpha) * soft
    

class MultiLabelDistillLoss_old(nn.Module):
    def __init__(self, alpha=0.3, temperature=4.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.hard_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def binary_kl(self, q_t, q_s, eps=1e-7):
        q_t = q_t.clamp(eps, 1-eps); q_s = q_s.clamp(eps, 1-eps)
        return (q_t*torch.log(q_t/q_s) + (1-q_t)*torch.log((1-q_t)/(1-q_s))).mean()

    def forward(self, student_logits, teacher_logits, labels):
        hard = self.hard_bce(student_logits, labels)
        q_t = torch.sigmoid(teacher_logits / self.T)
        q_s = torch.sigmoid(student_logits / self.T)
        soft = (self.T**2) * self.binary_kl(q_t, q_s)
        return self.alpha * hard + (1 - self.alpha) * soft
    

def set_binary_crossentropy_weighted_loss(positive_weights, negative_weights, epsilon=1e-7):
    """
    Note: Imported from the AI for Medicine Specialization course on Coursera: Assignment 1 Week 1.
    Returns weighted binary cross entropy loss function given negative weights and positive weights.

    Args:
      positive_weights (np.array): array of positive weights for each class, size (num_classes)
      negative_weights (np.array): array of negative weights for each class, size (num_classes)

    Returns:
      weighted_loss (function): weighted loss function
    """
    def binary_crossentropy_weighted_loss(y_true, y_pred):
        """
        Returns weighted binary cross entropy loss value.

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)

        Returns:
            loss (Tensor): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0

        for i in range(len(positive_weights)):
            # for each class, add average weighted loss for that class
            loss += -1 * K.mean((positive_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon) +
                                 negative_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon)))
        return loss

    return binary_crossentropy_weighted_loss


def get_teacher_model_Old():
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, len(PATHOLOGIES))
    return model

def get_teacher_model(num_classes: int, dropout: float = 0.0):
    """
    Build a DenseNet-121 teacher model with optional dropout in the classifier.
    
    Args:
        num_classes (int): Number of output classes (len(PATHOLOGIES))
        dropout (float): Dropout probability, 0.0 means no dropout
    """
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

    in_features = model.classifier.in_features
    if dropout > 0:
        # Replace classifier with Dropout + Linear
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        # Simple linear classifier (default)
        model.classifier = nn.Linear(in_features, num_classes)

    return model


def get_student_model():
    model = StudentMobileNetV2(len(PATHOLOGIES))
    return model

def compute_optimal_thresholds(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            preds = torch.sigmoid(model(images)).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    thresholds = []
    for i in range(len(PATHOLOGIES)):
        precision, recall, thresh = precision_recall_curve(all_labels[:, i], all_preds[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        thresholds.append(thresh[optimal_idx] if optimal_idx < len(thresh) else 0.5)
    return np.array(thresholds)

def plot_learning_curves_real_time(history, model_name, epoch, output_dir='plots'):
    Path(output_dir).mkdir(exist_ok=True)
    metrics = ['loss', 'acc', 'f1', 'auc']
    titles = ['Loss', 'Accuracy', 'F1 Score', 'AUC-ROC']
    extra_metrics = [('energy_kwh', 'Energy Consumption (kWh)'), ('emissions_kgco2eq', 'Carbon Emissions (kgCO₂eq)'), ('memory_mb', 'Memory Usage (MB)')]
    
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(10, 6))
        plt.plot(history[f'train_{metric}'], label=f'{model_name} Train')
        plt.plot(history[f'val_{metric}'], label=f'{model_name} Val')
        plt.title(f'{title} - {model_name} (Up to Epoch {epoch+1})')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_dir, f'{model_name}_{metric}_curves_epoch_{epoch+1}.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f'Saved real-time plot: {plot_path}')

    for metric, title in extra_metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(history[metric], label=f'{model_name} {title}')
        plt.title(f'{title} - {model_name} (Up to Epoch {epoch+1})')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_dir, f'{model_name}_{metric}_curves_epoch_{epoch+1}.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f'Saved real-time plot: {plot_path}')

def plot_learning_curves(history, model_name, output_dir='plots'):
    Path(output_dir).mkdir(exist_ok=True)
    metrics = ['loss', 'acc', 'f1', 'auc']
    titles = ['Loss', 'Accuracy', 'F1 Score', 'AUC-ROC']
    extra_metrics = [('energy_kwh', 'Energy Consumption (kWh)'), ('emissions_kgco2eq', 'Carbon Emissions (kgCO₂eq)'), ('memory_mb', 'Memory Usage (MB)')]
    
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(10, 6))
        plt.plot(history[f'train_{metric}'], label=f'{model_name} Train')
        plt.plot(history[f'val_{metric}'], label=f'{model_name} Val')
        plt.title(f'{title} - {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_dir, f'{model_name}_{metric}_curves_final.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f'Saved final plot: {plot_path}')

    for metric, title in extra_metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(history[metric], label=f'{model_name} {title}')
        plt.title(f'{title} - {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_dir, f'{model_name}_{metric}_curves_final.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f'Saved final plot: {plot_path}')

def plot_roc_pr_curves(models, model_names, val_loader, device, progress_file, output_dir='plots', thresholds=None, optimal = False):
    Path(output_dir).mkdir(exist_ok=True)
    all_labels, all_preds = {name: [] for name in model_names}, {name: [] for name in model_names}
    
    for name, model in zip(model_names, models):
        model.eval()
        val_pbar = tqdm(val_loader, desc=f'{name} ROC/PR Evaluation')
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                preds = torch.sigmoid(model(images)).cpu().numpy()
                all_preds[name].extend(preds)
                all_labels[name].extend(labels.numpy())
    
    for name in model_names:
        thresholds = []
        for i, pathology in enumerate(PATHOLOGIES + ['Overall']):
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            labels = np.array(all_labels[name])[:, i] if pathology != 'Overall' else np.array(all_labels[name])
            preds = np.array(all_preds[name])[:, i] if pathology != 'Overall' else np.array(all_preds[name])
            fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
            plt.title(f'ROC Curve - {pathology} - {name}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            precision, recall, thresh = precision_recall_curve(labels.flatten(), preds.flatten())
            pr_auc = auc(recall, precision)


            plt.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.2f})')

            #######

            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)

            thresholds.append(thresh[optimal_idx] if optimal_idx < len(thresh) else 0.5)
            
            # if thresholds is not None and pathology != 'Overall':
            #     plt.axvline(x=thresholds[i], color='r', linestyle='--', label=f'Threshold = {thresholds[i]:.2f}')

            

            if optimal and thresh[optimal_idx] is not None and pathology != 'Overall':
                 plt.axvline(x=thresh[optimal_idx], color='r', linestyle='--', label=f'Threshold = {thresh[optimal_idx]:.2f}')

            plt.title(f'Precision-Recall Curve - {pathology} - {name}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{name}_{pathology}_roc_pr_curves.png')
            plt.savefig(plot_path)
            plt.close()
            logging.info(f'Saved ROC/PR plot: {plot_path}')
        
        print(f"{name} Optimal Thresholds: {thresholds}")
        logging.info(f"{name} Optimal Thresholds: {thresholds}")
        with open(progress_file, 'a') as f:
            f.write(f"{name} Optimal Thresholds: {thresholds}\n")



def plot_combined_curves(models, model_names, val_loader, device, output_dir='plots'):
    """
    Genera y guarda gráficas ROC y PR combinadas para cada modelo, 
    mostrando las curvas de todas las patologías en una sola figura por modelo.

    Args:
        models (list): Lista de modelos de PyTorch a evaluar.
        model_names (list): Lista de nombres para los modelos.
        val_loader (DataLoader): DataLoader con los datos de validación.
        device (torch.device): Dispositivo para la ejecución (e.g., 'cuda' o 'cpu').
        output_dir (str): Directorio para guardar las gráficas.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Diccionarios para almacenar predicciones y etiquetas para cada modelo
    all_labels, all_preds = {name: [] for name in model_names}, {name: [] for name in model_names}
    
    # 1. Obtener todas las predicciones y etiquetas de los modelos
    for name, model in zip(model_names, models):
        model.eval()
        val_pbar = tqdm(val_loader, desc=f'Evaluando {name}')
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                # Aplicar sigmoide para obtener probabilidades y mover a CPU
                preds = torch.sigmoid(model(images)).cpu().numpy()
                all_preds[name].extend(preds)
                all_labels[name].extend(labels.numpy())

    # 2. Generar una gráfica combinada por cada modelo
    for name in model_names:
        # --- Configuración de la Gráfica ROC ---
        plt.figure(figsize=(11, 9))
        ax_roc = plt.gca()
        ax_roc.plot([0, 1], [0, 1], 'r--', label='Azar') # Línea de referencia
        
        # --- Configuración de la Gráfica Precision-Recall ---
        plt.figure(figsize=(11, 9))
        ax_pr = plt.gca()

        # Iterar sobre cada patología para dibujarla en la misma gráfica
        for i, pathology in enumerate(PATHOLOGIES + ['Overall']):
            # Preparar datos para la patología actual o para el caso "Overall"
            if pathology != 'Overall':
                labels = np.array(all_labels[name])[:, i]
                preds = np.array(all_preds[name])[:, i]
            else:
                # Para "Overall", aplanamos todas las etiquetas y predicciones
                labels = np.array(all_labels[name]).flatten()
                preds = np.array(all_preds[name]).flatten()

            # --- Calcular y dibujar Curva ROC ---
            fpr, tpr, _ = roc_curve(labels, preds)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f'{pathology} (AUC = {roc_auc:.2f})')

            # --- Calcular y dibujar Curva Precision-Recall ---
            precision, recall, _ = precision_recall_curve(labels, preds)
            pr_auc = auc(recall, precision)
            ax_pr.plot(recall, precision, label=f'{pathology} (AUC = {pr_auc:.2f})')

        # --- Finalizar y Guardar Gráfica ROC ---
        ax_roc.set_title(f'ROC Curves for All Pathologies - Model: {name}', fontsize=16)
        ax_roc.set_xlabel('False Positive Rate (FPR)', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate (TPR)', fontsize=12)
        ax_roc.legend()
        ax_roc.grid(True)
        roc_plot_path = os.path.join(output_dir, f'{name}_ALL_PATHOLOGIES_roc_curve.png')
       
       
        
        # Guardamos la figura correspondiente al eje ROC
        plt.figure(ax_roc.figure.number)
        plt.savefig(roc_plot_path)
        plt.close(ax_roc.figure)
        logging.info(f'ROC plot saved at: {roc_plot_path}')

        # --- Finalize and Save Precision-Recall Plot ---
        ax_pr.set_title(f'Precision-Recall Curves - Model: {name}', fontsize=16)
        ax_pr.set_xlabel('Recall', fontsize=12)
        ax_pr.set_ylabel('Precision', fontsize=12)
        ax_pr.legend()
        ax_pr.grid(True)
        pr_plot_path = os.path.join(output_dir, f'{name}_ALL_PATHOLOGIES_pr_curve.png')
        # Save the figure corresponding to the PR axis
        plt.figure(ax_pr.figure.number)
        plt.savefig(pr_plot_path)
        plt.close(ax_pr.figure)
        logging.info(f'PR plot saved at: {pr_plot_path}')




def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, 
                device, model_name, checkpoint_dir='checkpoints', best_metric='auc_roc', 
                patience=3, start_epoch=0, history=None, progress_file=None, plot_dir='plots', 
                threshold=0.5, compute_thresholds=False):
    Path(checkpoint_dir).mkdir(exist_ok=True)
    scaler = GradScaler()
    best_score = 1000000.0
    best_model_path = None
    patience_counter = 0
    history = history if history is not None else {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'train_acc_multilabel': [], 'val_acc': [], 'val_acc_multilabel': [], 
        'train_f1': [], 'val_f1': [], 'train_auc': [], 'val_auc': [], 
        'train_time': [], 'lr': [], 'energy_kwh': [], 'emissions_kgco2eq': [], 'memory_mb': []
    }
    thresholds = np.array([threshold] * len(PATHOLOGIES))
    total_start_time = time.time()
    multilabel_accuracy = MultilabelAccuracy(num_labels=len(PATHOLOGIES))
        

    def train_epoch():
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []
        train_pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{num_epochs} Training')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_pbar.set_postfix({'loss': loss.item()})
        return train_loss / len(train_loader), train_preds, train_labels

    def validate_epoch():
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        val_pbar = tqdm(val_loader, desc=f'{model_name} Epoch {epoch+1}/{num_epochs} Validation')
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs)
                val_preds.extend(preds.detach().cpu().numpy())

                # val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_pbar.set_postfix({'val_loss': loss.item()})
        return val_loss / len(val_loader), val_preds, val_labels

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024
        energy_kwh, emissions_kgco2eq = 0.0, 0.0

        # if compute_thresholds and epoch == start_epoch:
        #     thresholds = compute_optimal_thresholds(model, val_loader, device)
        #     print(f"{model_name} Optimal Thresholds: {thresholds}")
        #     logging.info(f"{model_name} Optimal Thresholds: {thresholds}")
        #     with open(progress_file, 'a') as f:
        #         f.write(f"{model_name} Optimal Thresholds: {thresholds}\n")

        if compute_thresholds:
            thresholds = compute_optimal_thresholds(model, val_loader, device)
            print(f"{model_name} Optimal Thresholds: {thresholds}")
            logging.info(f"{model_name} Optimal Thresholds: {thresholds}")
            with open(progress_file, 'a') as f:
                f.write(f"{model_name} Optimal Thresholds: {thresholds}\n")

        if CODECARBON_AVAILABLE:
            tracker = EmissionsTracker(output_dir=checkpoint_dir, output_file=f"{model_name}_emissions.csv")
            tracker.start()
            train_loss, train_preds, train_labels = train_epoch()
            val_loss, val_preds, val_labels = validate_epoch()
            emissions_kgco2eq = tracker.stop()
            if emissions_kgco2eq is None:
                print("WARNING: CodeCarbon was unable to measure emissions. It will be recorded as 0.")
                logging.info(f'WARNING: CodeCarbon was unable to measure emissions. It will be recorded as 0.')
                emissions_kgco2eq = 0.0
            energy_kwh = 0.0
            emissions_file_path = os.path.join(checkpoint_dir, f"{model_name}_emissions.csv")
            try:
                if os.path.exists(emissions_file_path):
                    # Usamos pandas para leer el archivo CSV fácilmente
                    results_df = pd.read_csv(emissions_file_path)
                    # Obtenemos el valor de la última fila (que corresponde a esta ejecución)
                    energy_kwh = results_df.iloc[-1]['energy_consumed']
            except Exception as e:
                print(f"WARNING: Could not read energy from CSV file: {e}")
                logging.info(f"WARNING: Could not read energy from CSV file: {e}")
                    
            logging.info(f'Saved emissions data: {os.path.join(checkpoint_dir, f"{model_name}_emissions.csv")}')
        else:
            train_loss, train_preds, train_labels = train_epoch()
            val_loss, val_preds, val_labels = validate_epoch()
            energy_kwh = (300 * (time.time() - start_time)) / (3600 * 1000)
            emissions_kgco2eq = 0.0  # No emissions data without CodeCarbon

        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        train_preds_bin = np.array([train_preds[:, i] > thresholds[i] for i in range(len(PATHOLOGIES))]).T
        val_preds_bin = np.array([val_preds[:, i] > thresholds[i] for i in range(len(PATHOLOGIES))]).T

        

        train_acc = accuracy_score(train_labels, train_preds_bin)
        train_preds_bin_tensor = torch.from_numpy(train_preds_bin)
        train_labels_tensor = torch.from_numpy(train_labels)
        train_acc_multilabel = multilabel_accuracy(train_preds_bin_tensor, train_labels_tensor)
        train_f1 = f1_score(train_labels, train_preds_bin, average='macro')
        train_auc = roc_auc_score(train_labels, train_preds, average='macro')

        val_acc = accuracy_score(val_labels, val_preds_bin)
        val_preds_bin_tensor = torch.from_numpy(val_preds_bin)
        val_labels_tensor = torch.from_numpy(val_labels)
        val_acc_multilabel = multilabel_accuracy(val_preds_bin_tensor, val_labels_tensor)
        val_f1 = f1_score(val_labels, val_preds_bin, average='macro')
        val_auc = roc_auc_score(val_labels, val_preds, average='macro')

        epoch_time = time.time() - start_time
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['train_acc_multilabel'].append(train_acc_multilabel)
        
        history['val_acc'].append(val_acc)
        history['val_acc_multilabel'].append(val_acc_multilabel)
        
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_time'].append(epoch_time)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['energy_kwh'].append(energy_kwh)
        history['emissions_kgco2eq'].append(emissions_kgco2eq)
        history['memory_mb'].append((memory_start + memory_end) / 2)

        metrics_str = (f'{model_name} Epoch {epoch+1}/{num_epochs}:\n'
                      f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Acc Multilabel: {train_acc_multilabel:.4f}, Train F1: {train_f1:.4f}, Train AUC: {train_auc:.4f}\n'
                      f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f},  Val Acc Multilabel: {val_acc_multilabel:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}\n'
                      f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}, Epoch Time: {epoch_time:.2f}s\n'
                      f'  Energy Consumption: {energy_kwh:.6f} kWh, Emissions: {emissions_kgco2eq:.6f} kgCO₂eq, Avg Memory: {(memory_start + memory_end)/2:.2f} MB\n')
        print(metrics_str)
        with open(progress_file, 'a') as f:
            f.write(metrics_str)

        scheduler.step(val_loss)

        score = val_loss
        if score < best_score:
            best_score = score
            best_model_path = os.path.join(checkpoint_dir, f'best_{model_name}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_score,
                'history': history,
                'thresholds': thresholds
            }, best_model_path)
            logging.info(f'Saved best model: {best_model_path}')
            print(f'Saved best model: {best_model_path}')
            with open(progress_file, 'a') as f:
                f.write(f'Saved best model: {best_model_path}\n')
            patience_counter = 0
        else:
            patience_counter += 1

        latest_model_path = os.path.join(checkpoint_dir, f'latest_{model_name}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'thresholds': thresholds
        }, latest_model_path)
        logging.info(f'Saved latest model: {latest_model_path}')

        logging.info(f'Epoch {epoch+1}/{num_epochs} - {model_name} - '
                     f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                     f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                      f'Train Acc Multilabel: {train_acc_multilabel:.4f}, Val Acc Multilabel: {val_acc_multilabel:.4f}, '
                     f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, '
                     f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, '
                     f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}, '
                     f'Epoch Time: {epoch_time:.2f}s, Energy: {energy_kwh:.6f} kWh, '
                     f'Emissions: {emissions_kgco2eq:.6f} kgCO₂eq, Avg Memory: {(memory_start + memory_end)/2:.2f} MB')
        plot_learning_curves_real_time(history, model_name, epoch, output_dir=plot_dir)
        if patience_counter >= patience:
            print(f'Early stopping triggered for {model_name} after {patience} epochs without improvement.')
            logging.info(f'Early stopping triggered for {model_name} at epoch {epoch+1}')
            with open(progress_file, 'a') as f:
                f.write(f'Early stopping triggered for {model_name} at epoch {epoch+1}\n')
            break

        

    total_time = time.time() - total_start_time
    total_energy_kwh = sum(history['energy_kwh'])
    total_emissions_kgco2eq = sum(history['emissions_kgco2eq'])
    avg_memory_mb = np.mean(history['memory_mb'])
    summary_str = (f'\n{model_name} Training Summary:\n'
                   f'  Total Training Time: {total_time:.2f}s\n'
                   f'  Total Energy Consumption: {total_energy_kwh:.6f} kWh\n'
                   f'  Total Carbon Emissions: {total_emissions_kgco2eq:.6f} kgCO₂eq\n'
                   f'  Average Memory Usage: {avg_memory_mb:.2f} MB\n')
    print(summary_str)
    with open(progress_file, 'a') as f:
        f.write(summary_str)
    logging.info(f'{model_name} - Total Training Time: {total_time:.2f}s, '
                 f'Energy: {total_energy_kwh:.6f} kWh, '
                 f'Emissions: {total_emissions_kgco2eq:.6f} kgCO₂eq, '
                 f'Avg Memory: {avg_memory_mb:.2f} MB')

    return history, best_model_path, total_time, total_energy_kwh, total_emissions_kgco2eq, thresholds

def train_distillation(teacher_model, student_model, train_loader, val_loader, criterion, 
                      optimizer, scheduler, num_epochs, device, checkpoint_dir='checkpoints', 
                      patience=3, start_epoch=0, history=None, progress_file=None, plot_dir='plots',
                      threshold=0.5, compute_thresholds=False, alpha=0.7, temperature = 4.0):
    Path(checkpoint_dir).mkdir(exist_ok=True)
    scaler = GradScaler()
    best_score = 1000000.0
    best_model_path = None
    patience_counter = 0
    history = history if history is not None else {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'train_acc_multilabel': [], 'val_acc': [], 'val_acc_multilabel': [], 
        'train_f1': [], 'val_f1': [], 'train_auc': [], 'val_auc': [], 
        'train_time': [], 'lr': [], 'energy_kwh': [], 'emissions_kgco2eq': [], 'memory_mb': []
    }
    
    thresholds = np.array([threshold] * len(PATHOLOGIES))
    total_start_time = time.time()
    multilabel_accuracy = MultilabelAccuracy(num_labels=len(PATHOLOGIES))
    
    def train_epoch():
        student_model.train()
        teacher_model.eval()
        train_loss, train_preds, train_labels = 0.0, [], []
        train_pbar = tqdm(train_loader, desc=f'Distilled Student Epoch {epoch+1}/{num_epochs} Training')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            with autocast():
                student_logits = student_model(images)
                loss = criterion(student_logits, teacher_logits, labels)
                # loss = kd_loss_fn(student_logits, teacher_logits, labels, alpha, temperature)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(student_logits).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_pbar.set_postfix({'loss': loss.item()})
        return train_loss / len(train_loader), train_preds, train_labels

    def validate_epoch():
        student_model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        val_pbar = tqdm(val_loader, desc=f'Distilled Student Epoch {epoch+1}/{num_epochs} Validation')
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    student_logits = student_model(images)
                    teacher_logits = teacher_model(images)
                    loss = criterion(student_logits, teacher_logits, labels)
                    # loss = kd_loss_fn(student_logits, teacher_logits, labels, alpha, temperature)

                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(student_logits).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_pbar.set_postfix({'val_loss': loss.item()})
        return val_loss / len(val_loader), val_preds, val_labels

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024
        energy_kwh, emissions_kgco2eq = 0.0, 0.0

        # if compute_thresholds and epoch == start_epoch:
        #     thresholds = compute_optimal_thresholds(student_model, val_loader, device)
        #     print(f"Distilled Student Optimal Thresholds: {thresholds}")
        #     logging.info(f"Distilled Student Optimal Thresholds: {thresholds}")
        #     with open(progress_file, 'a') as f:
        #         f.write(f"Distilled Student Optimal Thresholds: {thresholds}\n")

        
        if compute_thresholds:
            thresholds = compute_optimal_thresholds(student_model, val_loader, device)
            print(f"Distilled Student Optimal Thresholds: {thresholds}")
            logging.info(f"Distilled Student Optimal Thresholds: {thresholds}")
            with open(progress_file, 'a') as f:
                f.write(f"Distilled Student Optimal Thresholds: {thresholds}\n")


        if CODECARBON_AVAILABLE:
            tracker = EmissionsTracker(output_dir=checkpoint_dir, output_file=f"distilled_student_emissions.csv")
            tracker.start()
            train_loss, train_preds, train_labels = train_epoch()
            val_loss, val_preds, val_labels = validate_epoch()
            emissions_kgco2eq = tracker.stop()
            if emissions_kgco2eq is None:
                print("WARNING: CodeCarbon was unable to measure emissions. It will be recorded as 0.")
                logging.info(f'WARNING: CodeCarbon was unable to measure emissions. It will be recorded as 0.')
                emissions_kgco2eq = 0.0
            energy_kwh = 0.0
            emissions_file_path = os.path.join(checkpoint_dir, f"distilled_student_emissions.csv")
            try:
                if os.path.exists(emissions_file_path):
                    # Usamos pandas para leer el archivo CSV fácilmente
                    results_df = pd.read_csv(emissions_file_path)
                    # Obtenemos el valor de la última fila (que corresponde a esta ejecución)
                    energy_kwh = results_df.iloc[-1]['energy_consumed']
            except Exception as e:
                print(f"WARNING: Could not read energy from CSV file: {e}")
                logging.info(f"WARNING: Could not read energy from CSV file: {e}")
              
            logging.info(f'Saved emissions data: {os.path.join(checkpoint_dir, f"distilled_student_emissions.csv")}')
        else:
            train_loss, train_preds, train_labels = train_epoch()
            val_loss, val_preds, val_labels = validate_epoch()
            energy_kwh = (300 * (time.time() - start_time)) / (3600 * 1000)
            emissions_kgco2eq = 0.0

        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        train_preds_bin = np.array([train_preds[:, i] > thresholds[i] for i in range(len(PATHOLOGIES))]).T
        val_preds_bin = np.array([val_preds[:, i] > thresholds[i] for i in range(len(PATHOLOGIES))]).T

        print (f"train_preds_bin={train_preds_bin}")
        print (f"train_labels={train_labels}")

        print (f"val_preds_bin={val_preds_bin}")
        print (f"val_labels={val_labels}")

        train_acc = accuracy_score(train_labels, train_preds_bin)

        train_preds_bin_tensor = torch.from_numpy(train_preds_bin)
        train_labels_tensor = torch.from_numpy(train_labels)
        train_acc_multilabel = multilabel_accuracy(train_preds_bin_tensor, train_labels_tensor)
        
        
        train_f1 = f1_score(train_labels, train_preds_bin, average='macro')
        train_auc = roc_auc_score(train_labels, train_preds, average='macro')

        val_acc = accuracy_score(val_labels, val_preds_bin)
        val_preds_bin_tensor = torch.from_numpy(val_preds_bin)
        val_labels_tensor = torch.from_numpy(val_labels)
        val_acc_multilabel = multilabel_accuracy(val_preds_bin_tensor, val_labels_tensor)
        
        val_f1 = f1_score(val_labels, val_preds_bin, average='macro')
        val_auc = roc_auc_score(val_labels, val_preds, average='macro')

        epoch_time = time.time() - start_time
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['train_acc_multilabel'].append(train_acc_multilabel)

        
        history['val_acc'].append(val_acc)
        history['val_acc_multilabel'].append(val_acc_multilabel)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_time'].append(epoch_time)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['energy_kwh'].append(energy_kwh)
        history['emissions_kgco2eq'].append(emissions_kgco2eq)
        history['memory_mb'].append((memory_start + memory_end) / 2)

        metrics_str = (f'Distilled Student Epoch {epoch+1}/{num_epochs}:\n'
                      f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Acc Multilabel: {train_acc_multilabel:.4f}, Train F1: {train_f1:.4f}, Train AUC: {train_auc:.4f}\n'
                      f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Acc Multilabel: {val_acc_multilabel:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}\n'
                      f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}, Epoch Time: {epoch_time:.2f}s\n'
                      f'  Energy Consumption: {energy_kwh:.6f} kWh, Emissions: {emissions_kgco2eq:.6f} kgCO₂eq, Avg Memory: {(memory_start + memory_end)/2:.2f} MB\n')
        print(metrics_str)
        with open(progress_file, 'a') as f:
            f.write(metrics_str)

        scheduler.step(val_loss)
        score = val_loss
        if score < best_score:
            best_score = score
            best_model_path = os.path.join(checkpoint_dir, f'best_distilled_student.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_score,
                'history': history,
                'thresholds': thresholds
            }, best_model_path)
            logging.info(f'Saved best model: {best_model_path}')
            print(f'Saved best model: {best_model_path}')
            with open(progress_file, 'a') as f:
                f.write(f'Saved best model: {best_model_path}\n')
            patience_counter = 0
        else:
            patience_counter += 1

        latest_model_path = os.path.join(checkpoint_dir, f'latest_distilled_student.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'thresholds': thresholds
        }, latest_model_path)
        logging.info(f'Saved latest model: {latest_model_path}')

        logging.info(f'Epoch {epoch+1}/{num_epochs} - Distilled Student - '
                     f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                     f'Train Acc: {train_acc:.4f}, Train Acc Multilabel: {train_acc_multilabel:.4f}, '
                     f'Val Acc: {val_acc:.4f}, Val Acc Multilabel: {val_acc_multilabel:.4f}, '
                     f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, '
                     f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, '
                     f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}, '
                     f'Epoch Time: {epoch_time:.2f}s, Energy: {energy_kwh:.6f} kWh, '
                     f'Emissions: {emissions_kgco2eq:.6f} kgCO₂eq, Avg Memory: {(memory_start + memory_end)/2:.2f} MB')
        plot_learning_curves_real_time(history, 'Distilled Student', epoch, output_dir=plot_dir)
        if patience_counter >= patience:
            print(f'Early stopping triggered for Distilled Student after {patience} epochs without improvement.')
            logging.info(f'Early stopping triggered for Distilled Student at epoch {epoch+1}')
            with open(progress_file, 'a') as f:
                f.write(f'Early stopping triggered for Distilled Student at epoch {epoch+1}\n')
            break

        
    total_time = time.time() - total_start_time
    total_energy_kwh = sum(history['energy_kwh'])
    total_emissions_kgco2eq = sum(history['emissions_kgco2eq'])
    avg_memory_mb = np.mean(history['memory_mb'])
    summary_str = (f'\nDistilled Student Training Summary:\n'
                   f'  Total Training Time: {total_time:.2f}s\n'
                   f'  Total Energy Consumption: {total_energy_kwh:.6f} kWh\n'
                   f'  Total Carbon Emissions: {total_emissions_kgco2eq:.6f} kgCO₂eq\n'
                   f'  Average Memory Usage: {avg_memory_mb:.2f} MB\n')
    print(summary_str)
    with open(progress_file, 'a') as f:
        f.write(summary_str)
    logging.info(f'Distilled Student - Total Training Time: {total_time:.2f}s, '
                 f'Energy: {total_energy_kwh:.6f} kWh, '
                 f'Emissions: {total_emissions_kgco2eq:.6f} kgCO₂eq, '
                 f'Avg Memory: {avg_memory_mb:.2f} MB')

    return history, best_model_path, total_time, total_energy_kwh, total_emissions_kgco2eq, thresholds

def evaluate_models(models, model_names, val_loader, device, output_dir='plots', progress_file=None, thresholds=None):
    results = []
    # if thresholds is None:
    #     thresholds = np.array([0.5] * len(PATHOLOGIES))
    
    for name, model, threshold in zip(model_names, models, thresholds):

        # if thresholds == None:
        #     thresholds = compute_optimal_thresholds(model, val_loader, device)
        #     print(f"{name} Optimal Thresholds: {thresholds}")
        #     logging.info(f"{name} Optimal Thresholds: {thresholds}")
        #     with open(progress_file, 'a') as f:
        #         f.write(f"{name} Optimal Thresholds: {thresholds}\n")

        model.eval()
        all_preds, all_labels = [], []
        val_pbar = tqdm(val_loader, desc=f'{name} Final Evaluation')
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                preds = torch.sigmoid(model(images)).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_preds_bin = np.array([all_preds[:, i] > threshold[i] for i in range(len(PATHOLOGIES))]).T
        
        multilabel_accuracy = MultilabelAccuracy(num_labels=len(PATHOLOGIES))
        metrics = {'Model': name}
        for i, pathology in enumerate(PATHOLOGIES):
            acc = accuracy_score(all_labels[:, i], all_preds_bin[:, i])
            f1 = f1_score(all_labels[:, i], all_preds_bin[:, i])
            auc_roc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            sensitivity = recall_score(all_labels[:, i], all_preds_bin[:, i])
            specificity = recall_score(1 - all_labels[:, i], 1 - all_preds_bin[:, i])
            metrics.update({
                f'{pathology}_Acc': acc,
                f'{pathology}_F1': f1,
                f'{pathology}_AUC': auc_roc,
                f'{pathology}_Sensitivity': sensitivity,
                f'{pathology}_Specificity': specificity
            })
        acc = accuracy_score(all_labels, all_preds_bin)
        
        preds_bin_tensor = torch.from_numpy(all_preds_bin)
        labels_tensor = torch.from_numpy(all_labels)
        acc_multilabel = multilabel_accuracy(preds_bin_tensor, labels_tensor)
        
        f1 = f1_score(all_labels, all_preds_bin, average='macro')
        auc_roc = roc_auc_score(all_labels, all_preds, average='macro')
        sensitivity = recall_score(all_labels, all_preds_bin, average='macro')
        specificity = recall_score(1 - all_labels, 1 - all_preds_bin, average='macro')
        metrics.update({
            'Overall_Acc': acc,
            'Overall_Acc_MultiLabel': acc_multilabel, 
            'Overall_F1': f1,
            'Overall_AUC': auc_roc,
            'Overall_Sensitivity': sensitivity,
            'Overall_Specificity': specificity
        })
        results.append(metrics)
    
    summary_table = pd.DataFrame(results)
    summary_table_path = os.path.join(output_dir, 'summary_metrics.csv')
    summary_table.to_csv(summary_table_path)
    logging.info(f'Saved summary table: {summary_table_path}')

    compute_confusion_matrices(models, model_names, val_loader, device, output_dir, thresholds)
    
    return summary_table, thresholds

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint.get('epoch', -1) + 1
            history = checkpoint.get('history', {
                'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 
                'train_f1': [], 'val_f1': [], 'train_auc': [], 'val_auc': [], 
                'train_time': [], 'lr': [], 'energy_kwh': [], 'emissions_kgco2eq': [], 'memory_mb': []
            })
            thresholds = checkpoint.get('thresholds', np.array([0.5] * len(PATHOLOGIES)))
            logging.info(f'Loaded checkpoint: {checkpoint_path} at epoch {epoch}')
            print(f'Loaded checkpoint: {checkpoint_path} at epoch {epoch}')
            return model, optimizer, scheduler, epoch, history, thresholds
        except Exception as e:
            logging.error(f'Failed to load checkpoint {checkpoint_path}: {e}')
            print(f'Failed to load checkpoint {checkpoint_path}: {e}. Starting from scratch.')
            return model, optimizer, scheduler, 0, None, np.array([0.5] * len(PATHOLOGIES))
    else:
        logging.info(f'No checkpoint found at {checkpoint_path}. Starting from scratch.')
        print(f'No checkpoint found at {checkpoint_path}. Starting from scratch.')
        return model, optimizer, scheduler, 0, None, np.array([0.5] * len(PATHOLOGIES))


def create_final_train_val_files(pathFile, results_dir, data_fraction=1.0, val_size=0.1, random_state=42):
    """
    Crea DataLoaders para el re-entrenamiento final, dividiendo train.csv
    en un nuevo conjunto de entrenamiento (ej. 90%) y de validación (ej. 10%).
    """
    print("Creando DataLoaders para el re-entrenamiento final (división 90/10 de train.csv)...")
    
    full_train_df = pd.read_csv(pathFile)
    if data_fraction < 1.0:
        full_train_df = full_train_df.sample(frac=data_fraction, random_state=random_state).reset_index(drop=True)
        
    # Extraer los IDs de paciente para una división por grupos
    patient_ids = full_train_df['Path'].apply(lambda x: x.split('/')[2])
    groups = patient_ids.values

    # Usar GroupShuffleSplit para crear una única división train/val
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(full_train_df, groups=groups))

    final_train_df = full_train_df.iloc[train_idx]
    final_val_df = full_train_df.iloc[val_idx]

    filePath_Train = os.path.join(results_dir, "train_1.csv")
    filePath_Valid = os.path.join(results_dir, "valid_1.csv")
    final_train_df.to_csv(filePath_Train,  index=False)
    final_val_df.to_csv(filePath_Valid,  index=False)

    
    print(f"División para entrenamiento final: {len(final_train_df)} muestras de entrenamiento, {len(final_val_df)} muestras de validación.")


    return filePath_Train, filePath_Valid


def compute_confusion_matrices(models, model_names, val_loader, device, output_dir='plots', thresholds=None):
    """
    Compute confusion matrices for each pathology and an overall matrix for each model.
    Save matrices as PNG plots and numerical values in a CSV file.
    
    Args:
        models: List of PyTorch models (Teacher, Student, Distilled Student).
        model_names: List of model names.
        val_loader: DataLoader for validation set.
        device: Device to run models (cuda or cpu).
        output_dir: Directory to save plots and CSV.
        thresholds: Array of thresholds per pathology (default: 0.5 for each).
    
    Returns:
        confusion_data: List of dictionaries with confusion matrix data.
    """
    Path(output_dir).mkdir(exist_ok=True)
    # if thresholds is None:
    #     thresholds = np.array([0.5] * len(PATHOLOGIES))
    
    confusion_data = []
    
    for name, model, threshold in zip(model_names, models, thresholds):
        if threshold is None:
            threshold = np.array([0.5] * len(PATHOLOGIES))
        model.eval()
        all_preds, all_labels = [], []
        val_pbar = tqdm(val_loader, desc=f'{name} Confusion Matrix Evaluation')
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                preds = torch.sigmoid(model(images)).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_preds_bin = np.array([all_preds[:, i] > threshold[i] for i in range(len(PATHOLOGIES))]).T
        
        # Compute per-pathology confusion matrices
        for i, pathology in enumerate(PATHOLOGIES):
            cm = confusion_matrix(all_labels[:, i], all_preds_bin[:, i])
            tn, fp, fn, tp = cm.ravel()
            
            # Save numerical data
            confusion_data.append({
                'Model': name,
                'Pathology': pathology,
                'True Positives (TP)': tp,
                'True Negatives (TN)': tn,
                'False Positives (FP)': fp,
                'False Negatives (FN)': fn
            })
            
            # Plot confusion matrix
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'])
            plt.title(f'Confusion Matrix - {pathology} - {name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plot_path = os.path.join(output_dir, f'{name}_{pathology}_confusion_matrix.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            logging.info(f'Saved confusion matrix plot: {plot_path}')
        
        # Compute overall confusion matrix (aggregated across pathologies)
        cm_overall = confusion_matrix(all_labels.flatten(), all_preds_bin.flatten())
        tn, fp, fn, tp = cm_overall.ravel()
        
        # Save overall numerical data
        confusion_data.append({
            'Model': name,
            'Pathology': 'Overall',
            'True Positives (TP)': tp,
            'True Negatives (TN)': tn,
            'False Positives (FP)': fp,
            'False Negatives (FN)': fn
        })
        
        # Plot overall confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title(f'Overall Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plot_path = os.path.join(output_dir, f'{name}_overall_confusion_matrix.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logging.info(f'Saved overall confusion matrix plot: {plot_path}')
    
    # Save confusion matrix data to CSV
    confusion_df = pd.DataFrame(confusion_data)
    confusion_csv_path = os.path.join(output_dir, 'confusion_matrices.csv')
    confusion_df.to_csv(confusion_csv_path, index=False)
    logging.info(f'Saved confusion matrix data: {confusion_csv_path}')
    print(f'Saved confusion matrix data: {confusion_csv_path}')
    
    return confusion_data


#### Energy Calculation


@dataclass
class TechConstants:
    eps_mac: float         # J / MAC
    eps_dram_byte: float   # J / byte
    eps_sram_byte: float = 0.0  # reserved for extensions


def get_tech_constants(precision: str = "fp16") -> TechConstants:
    p = precision.lower()
    if p == "fp32":
        return TechConstants(eps_mac=8e-12, eps_dram_byte=40e-12)
    elif p == "fp16":
        return TechConstants(eps_mac=4e-12, eps_dram_byte=25e-12)
    elif p == "int8":
        return TechConstants(eps_mac=1e-12, eps_dram_byte=15e-12)
    return TechConstants(eps_mac=4e-12, eps_dram_byte=25e-12)


@dataclass
class LayerStats:
    macs: float
    flops: float
    dram_bytes: float


@dataclass
class ModelStats:
    macs_total: float
    flops_total: float
    dram_bytes_total: float


def _tensor_nbytes(x: Tensor, assume_dtype_bytes: Optional[int] = None) -> int:
    if x is None:
        return 0
    if assume_dtype_bytes is not None:
        return int(np.prod(list(x.shape)) * assume_dtype_bytes)
    return x.element_size() * x.numel()


def _conv2d_macs(in_c, out_c, out_h, out_w, k_h, k_w, groups=1) -> float:
    return float(out_h * out_w * out_c * (in_c / groups) * k_h * k_w)


def _linear_macs(in_f, out_f) -> float:
    return float(in_f * out_f)


class EnergyHooks:
    """Forward-hooks collector to approximate per-layer MACs and DRAM bytes."""

    def __init__(self, precision_bytes: int = 2):
        self.layer_stats: List[LayerStats] = []
        self.precision_bytes = precision_bytes
        self.handles = []

    def register(self, model: nn.Module):
        def hook(module, inputs, output):
            macs = 0.0
            bytes_dram = 0.0

            inp = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            out = output

            in_bytes = _tensor_nbytes(inp, assume_dtype_bytes=self.precision_bytes)
            out_bytes = _tensor_nbytes(out, assume_dtype_bytes=self.precision_bytes)

            w_bytes = 0
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                w = module.weight
                w_bytes = _tensor_nbytes(w)

            if isinstance(module, nn.Conv2d):
                B = out.shape[0] if isinstance(out, torch.Tensor) else 1
                Cin = module.in_channels
                Cout = module.out_channels
                Kh, Kw = module.kernel_size
                Hout = out.shape[-2]
                Wout = out.shape[-1]
                macs = _conv2d_macs(Cin, Cout, Hout, Wout, Kh, Kw, groups=module.groups)
                bytes_dram = in_bytes / max(1, B) + out_bytes / max(1, B) + w_bytes
            elif isinstance(module, nn.Linear):
                macs = _linear_macs(module.in_features, module.out_features)
                bytes_dram = in_bytes + out_bytes + w_bytes
            elif isinstance(module, (nn.BatchNorm2d, nn.ReLU, nn.SiLU, nn.GELU, nn.ReLU6)):
                macs = 0.0
                bytes_dram = in_bytes + out_bytes
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                macs = 0.0
                bytes_dram = in_bytes + out_bytes
            else:
                macs = 0.0
                bytes_dram = in_bytes + out_bytes

            self.layer_stats.append(LayerStats(
                macs=float(macs),
                flops=float(2.0 * macs),
                dram_bytes=float(bytes_dram)
            ))

        for m in model.modules():
            if len(list(m.children())) == 0:
                self.handles.append(m.register_forward_hook(hook))

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def summarize(self) -> ModelStats:
        macs = sum(s.macs for s in self.layer_stats)
        flops = sum(s.flops for s in self.layer_stats)
        dram = sum(s.dram_bytes for s in self.layer_stats)
        return ModelStats(macs_total=macs, flops_total=flops, dram_bytes_total=dram)


def profile_model_once(model: nn.Module, sample: torch.Tensor, precision: str = "fp16") -> ModelStats:
    model.eval()
    hooks = EnergyHooks(precision_bytes=2 if precision.lower() == "fp16" else 4)
    hooks.register(model)
    with torch.inference_mode():
        _ = model(sample)
    stats = hooks.summarize()
    hooks.clear()
    return stats


def estimate_energy_theoretical(stats: ModelStats, consts: TechConstants) -> float:
    e_comp = stats.macs_total * consts.eps_mac
    e_data = stats.dram_bytes_total * consts.eps_dram_byte
    return float(e_comp + e_data)


def calibrate_energy_linear_model(
    bench_df: pd.DataFrame,
    flops_per_img: float,
    bytes_per_img: float
):
    df = bench_df.copy()
    df = df[(df["samples"] > 0) & (df["energy_kwh"] >= 0)]
    if len(df) < 2:
        return 0.0, 0.0, 0.0

    y = (df["energy_kwh"].values * 3_600_000.0) / df["samples"].values  # J/img
    X = np.stack([
        np.full_like(y, flops_per_img, dtype=float),
        np.full_like(y, bytes_per_img, dtype=float),
        np.ones_like(y, dtype=float)
    ], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coef.tolist()
    return float(a), float(b), float(c)


def energy_linear_predict(flops_per_img: float, bytes_per_img: float, a: float, b: float, c: float) -> float:
    return float(a * flops_per_img + b * bytes_per_img + c)


def make_energy_report(
    models: Dict[str, nn.Module],
    device: torch.device,
    sample_input_shape=(1, 3, 224, 224),
    precision: str = "fp16",
    benchmark_results: Dict[str, pd.DataFrame] | None = None,
    out_dir: str = "energy_report"
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    with torch.inference_mode():
        sample = torch.randn(*sample_input_shape, device=device)
    consts = get_tech_constants(precision)

    rows = {}
    per_model_stats = {}

    for name, model in models.items():
        model = model.to(device)
        stats = profile_model_once(model, sample, precision=precision)
        per_model_stats[name] = stats
        e_theory = estimate_energy_theoretical(stats, consts)
        rows[name] = {
            "model": name,
            "flops_per_img": stats.flops_total,
            "macs_per_img": stats.macs_total,
            "bytes_per_img": stats.dram_bytes_total,
            "E_theoretical_J": e_theory
        }

    df_profile = pd.DataFrame(list(rows.values()))
    df_profile.to_csv(os.path.join(out_dir, "model_profile.csv"), index=False)

    calib_rows = []
    if benchmark_results:
        for name, df_bench in benchmark_results.items():
            stats = per_model_stats.get(name)
            if stats is None:
                continue
            a, b, c = calibrate_energy_linear_model(
                bench_df=df_bench,
                flops_per_img=stats.flops_total,
                bytes_per_img=stats.dram_bytes_total
            )
            e_calib = energy_linear_predict(stats.flops_total, stats.dram_bytes_total, a, b, c)

            df_b = df_bench[(df_bench["samples"] > 0) & (df_bench["energy_kwh"] >= 0)].copy()
            if len(df_b) > 0:
                df_b["J_per_img"] = (df_b["energy_kwh"] * 3_600_000.0) / df_b["samples"]
                measured_median = float(df_b["J_per_img"].median())
                best_imgs_per_J = float((df_b["samples"] / (df_b["energy_kwh"] * 3_600_000.0 + 1e-12)).max())
            else:
                measured_median = np.nan
                best_imgs_per_J = np.nan

            calib_rows.append({
                "model": name,
                "a_J_per_FLOP": a,
                "b_J_per_byte": b,
                "c_J_const": c,
                "E_calibrated_J": e_calib,
                "E_measured_J_img_median": measured_median,
                "imgs_per_J_best": best_imgs_per_J
            })

    df_calib = pd.DataFrame(calib_rows) if calib_rows else pd.DataFrame(
        columns=["model","a_J_per_FLOP","b_J_per_byte","c_J_const","E_calibrated_J","E_measured_J_img_median","imgs_per_J_best"]
    )
    df_summary = df_profile.merge(df_calib, on="model", how="left")
    df_summary.to_csv(os.path.join(out_dir, "energy_summary.csv"), index=False)

    # Per-model bars
    for name in models.keys():
        sub = df_summary[df_summary["model"] == name]
        if sub.empty:
            continue
        plt.figure(figsize=(7.5, 5.5))
        vals, labels = [], []
        if "E_measured_J_img_median" in sub and not np.isnan(sub["E_measured_J_img_median"].values[0]):
            vals.append(sub["E_measured_J_img_median"].values[0]); labels.append("Measured (median)")
        if "E_calibrated_J" in sub and not np.isnan(sub["E_calibrated_J"].values[0]):
            vals.append(sub["E_calibrated_J"].values[0]); labels.append("Calibrated model")
        vals.append(sub["E_theoretical_J"].values[0]); labels.append("Theoretical model")
        plt.bar(labels, vals)
        plt.ylabel("Energy per image (J)")
        plt.title(f"Energy per image - {name}")
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(os.path.join(out_dir, f"{name}_energy_bars.png"), bbox_inches="tight")
        plt.close()

    # Comparative best images/kWh
    if benchmark_results:
        comp = []
        for name, df_bench in benchmark_results.items():
            df_b = df_bench[(df_bench["samples"] > 0) & (df_bench["energy_kwh"] > 0)].copy()
            if len(df_b) == 0: 
                continue
            df_b["imgs_per_kWh"] = df_b["samples"] / df_b["energy_kwh"]
            best = float(df_b["imgs_per_kWh"].max())
            comp.append({"model": name, "best_imgs_per_kWh": best})
        if comp:
            dfc = pd.DataFrame(comp)
            plt.figure(figsize=(7.5, 5.5))
            plt.bar(dfc["model"], dfc["best_imgs_per_kWh"])
            plt.ylabel("Images per kWh (higher is better)")
            plt.title("Best Images/kWh across benchmark sizes")
            plt.grid(axis="y", alpha=0.3)
            plt.savefig(os.path.join(out_dir, "COMPARE_best_imgs_per_kWh.png"), bbox_inches="tight")
            plt.close()

    # Roofline-style scatter
    roof_rows = []
    for name, df_bench in (benchmark_results or {}).items():
        stats = per_model_stats.get(name)
        if stats is None:
            continue
        intensity = stats.flops_total / max(1.0, stats.dram_bytes_total)  # FLOPs/byte
        tmp = df_bench.copy()
        tmp["imgs_per_s"] = tmp["samples"] / tmp["time_s"].clip(lower=1e-9)
        tmp["achieved_flops_s"] = tmp["imgs_per_s"] * stats.flops_total
        roof_rows.append({
            "model": name,
            "intensity_flops_per_byte": intensity,
            "achieved_flops_s_max": float(tmp["achieved_flops_s"].max())
        })
    if roof_rows:
        dfr = pd.DataFrame(roof_rows)
        plt.figure(figsize=(7.5, 5.5))
        for _, r in dfr.iterrows():
            plt.scatter(r["intensity_flops_per_byte"], r["achieved_flops_s_max"], label=r["model"])
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Operational intensity (FLOPs/byte)")
        plt.ylabel("Achieved performance (FLOPs/s)")
        plt.title("Roofline-style scatter (per model)")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(out_dir, "roofline_scatter.png"), bbox_inches="tight")
        plt.close()

    # LaTeX table
    latex_df = df_summary[[
        "model","flops_per_img","bytes_per_img","E_theoretical_J","E_calibrated_J","E_measured_J_img_median","imgs_per_J_best"
    ]].copy()
    latex_df.columns = [
        "Model","FLOPs/img","Bytes/img","E_theory (J/img)","E_calib (J/img)","E_measured (J/img)","imgs/J (best)"
    ]
    for c in ["FLOPs/img","Bytes/img"]:
        latex_df[c] = latex_df[c].map(lambda x: f"{x:,.2e}")
    for c in ["E_theory (J/img)","E_calib (J/img)","E_measured (J/img)","imgs/J (best)"]:
        latex_df[c] = latex_df[c].map(lambda x: ("-" if pd.isna(x) else f"{x:,.4g}"))
    with open(os.path.join(out_dir, "energy_summary_table.tex"), "w") as f:
        f.write(latex_df.to_latex(index=False, escape=False))

    return df_summary


def run_inference_efficiency_benchmarks(
    models_dict,
    dataset,
    device,
    sizes=(1000, 5000, 10000, 20000, 30000, 40000),
    batch_size=128,
    num_workers=4,
    output_dir='benchmarks_efficiency',
    assumed_power_watts=300.0,
    emissions_factor_kg_per_kwh=0.5,
    make_comparative_plots=True
):
    """
    Measure inference efficiency for each model and generate per-model plots.
    Optionally, generate comparative plots across models.

    Parameters
    ----------
    models_dict : dict[str, torch.nn.Module]
        Mapping model names -> models (already on device).
    dataset : torch.utils.data.Dataset
        Validation dataset used for benchmarking.
    device : torch.device
        'cuda' or 'cpu'.
    sizes : tuple[int]
        Requested sizes (samples to evaluate). Clipped by len(dataset).
    batch_size : int
        Inference batch size.
    num_workers : int
        DataLoader workers.
    output_dir : str
        Output directory for CSVs and plots.
    assumed_power_watts : float
        Average power if CodeCarbon is unavailable.
    emissions_factor_kg_per_kwh : float
        Grid factor if CodeCarbon is unavailable.
    make_comparative_plots : bool
        If True, produce model-comparison plots.

    Returns
    -------
    dict[str, pd.DataFrame]
        model_name -> DataFrame with results.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    is_cuda = (device.type == 'cuda')
    proc = psutil.Process()
    results_by_model = {}

    def _benchmark_one_model(model_name, model):
        model.eval()
        rows = []
        total_len = len(dataset)
        model_dir = os.path.join(output_dir, model_name)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        for N in sizes:
            n_eff = min(N, total_len)
            if n_eff <= 0:
                continue

            subset = Subset(dataset, range(n_eff))
            loader = DataLoader(
                subset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=is_cuda
            )

            # CPU memory before
            cpu_mb_before = proc.memory_info().rss / (1024.0 * 1024.0)

            # Reset CUDA peak memory
            gpu_mb_peak_alloc = 0.0
            gpu_mb_peak_reserved = 0.0
            if is_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            # Energy/CO2 tracking
            emissions_kgco2eq = 0.0
            energy_kwh = 0.0
            tracker = None
            emissions_csv_path = None
            used_codecarbon = False

            if CODECARBON_AVAILABLE:
                used_codecarbon = True
                emissions_csv_path = os.path.join(
                    model_dir, f"{model_name}_inference_{n_eff}.csv"
                )
                tracker = EmissionsTracker(
                    output_dir=model_dir,
                    output_file=f"{model_name}_inference_{n_eff}.csv"
                )
                tracker.start()

            # Inference timing
            start = time.time()
            with torch.inference_mode():
                for images, _ in loader:
                    images = images.to(device, non_blocking=True)
                    with autocast():
                        _ = model(images)
                    if is_cuda:
                        torch.cuda.synchronize()
            elapsed_s = time.time() - start

            # Energy/CO2
            if used_codecarbon and tracker is not None:
                emissions_kgco2eq = tracker.stop() or 0.0
                try:
                    if os.path.exists(emissions_csv_path):
                        df_cc = pd.read_csv(emissions_csv_path)
                        energy_kwh = float(df_cc.iloc[-1]['energy_consumed'])
                except Exception as e:
                    print(f"[WARN][{model_name}] Could not read CodeCarbon energy: {e}")
                    energy_kwh = 0.0
            else:
                energy_kwh = (assumed_power_watts * elapsed_s) / (3600.0 * 1000.0)
                emissions_kgco2eq = energy_kwh * emissions_factor_kg_per_kwh

            # CPU memory after
            cpu_mb_after = proc.memory_info().rss / (1024.0 * 1024.0)
            cpu_mb_delta = max(0.0, cpu_mb_after - cpu_mb_before)

            if is_cuda:
                try:
                    gpu_mb_peak_alloc = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                    gpu_mb_peak_reserved = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
                except Exception:
                    pass

            # Efficiency metrics
            throughput = n_eff / max(1e-9, elapsed_s)
            time_ms_per_img = 1000.0 * elapsed_s / max(1, n_eff)
            energy_joules = energy_kwh * 3_600_000.0

            imgs_per_kwh = (n_eff / energy_kwh) if energy_kwh > 0 else float('nan')
            imgs_per_joule = (n_eff / energy_joules) if energy_joules > 0 else float('nan')
            kwh_per_img = energy_kwh / max(1, n_eff)
            kgco2_per_img = emissions_kgco2eq / max(1, n_eff)
            avg_power_watts = (energy_kwh * 3_600_000.0) / max(1e-9, elapsed_s)

            vram_gb_peak = (gpu_mb_peak_reserved or gpu_mb_peak_alloc) / 1024.0
            imgs_per_s_per_gb_vram = throughput / vram_gb_peak if vram_gb_peak > 0 else float('nan')

            rows.append({
                'model': model_name,
                'samples': n_eff,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'time_s': elapsed_s,
                'imgs_per_s': throughput,
                'time_ms_per_img': time_ms_per_img,
                'energy_kwh': energy_kwh,
                'emissions_kgco2eq': emissions_kgco2eq,
                'kwh_per_img': kwh_per_img,
                'kgco2_per_img': kgco2_per_img,
                'imgs_per_kwh': imgs_per_kwh,
                'energy_joules': energy_joules,
                'imgs_per_joule': imgs_per_joule,
                'avg_power_watts': avg_power_watts,
                'cpu_mb_before': cpu_mb_before,
                'cpu_mb_after': cpu_mb_after,
                'cpu_mb_delta': cpu_mb_delta,
                'gpu_mb_peak_alloc': gpu_mb_peak_alloc,
                'gpu_mb_peak_reserved': gpu_mb_peak_reserved,
                'imgs_per_s_per_gb_vram': imgs_per_s_per_gb_vram,
                'energy_source': 'codecarbon' if used_codecarbon else 'estimated',
                'assumed_power_watts_if_estimated': None if used_codecarbon else assumed_power_watts,
                'emissions_factor_kg_per_kwh_if_estimated': None if used_codecarbon else emissions_factor_kg_per_kwh
            })

            print(f"[{model_name}] N={n_eff} | {elapsed_s:.2f}s | {throughput:.2f} img/s | "
                  f"{time_ms_per_img:.2f} ms/img | {energy_kwh:.6f} kWh | "
                  f"{emissions_kgco2eq:.6f} kgCO2 | CPU +{cpu_mb_delta:.1f} MB | "
                  f"GPU peak {(gpu_mb_peak_reserved or gpu_mb_peak_alloc):.1f} MB")

        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, f"{model_name}_validation_inference_benchmark.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved inference benchmark CSV: {csv_path}")
        print(f"Saved inference benchmark CSV: {csv_path}")

        # --- Per-model plots ---
        def _plot_metric(metric_key, ylabel, filename, ylog=False):
            plt.figure(figsize=(9, 6))
            plt.plot(df['samples'], df[metric_key], marker='o')
            plt.title(f'{model_name} - {ylabel} vs Samples')
            plt.xlabel('Samples')
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            if ylog:
                plt.yscale('log')
            out_path = os.path.join(output_dir, f"{model_name}_{filename}.png")
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved plot: {out_path}")

        _plot_metric('time_s', 'Total Time (s)', 'time_s')
        _plot_metric('imgs_per_s', 'Throughput (img/s)', 'throughput')
        _plot_metric('time_ms_per_img', 'Latency (ms/img)', 'latency_ms_per_img')
        _plot_metric('energy_kwh', 'Energy (kWh)', 'energy_kwh')
        _plot_metric('kwh_per_img', 'Energy per Image (kWh/img)', 'kwh_per_img', ylog=True)
        _plot_metric('emissions_kgco2eq', 'CO2 (kg)', 'co2_kg')
        _plot_metric('kgco2_per_img', 'CO2 per Image (kg/img)', 'co2_per_img', ylog=True)
        _plot_metric('imgs_per_kwh', 'Images per kWh (img/kWh)', 'imgs_per_kwh', ylog=True)
        _plot_metric('imgs_per_joule', 'Images per Joule (img/J)', 'imgs_per_joule', ylog=True)
        _plot_metric('avg_power_watts', 'Average Power (W)', 'avg_power_watts')
        _plot_metric('cpu_mb_delta', 'CPU Memory Delta (MB)', 'cpu_mem_delta')
        _plot_metric('gpu_mb_peak_reserved', 'GPU Peak Reserved (MB)', 'gpu_mem_peak_reserved')
        _plot_metric('gpu_mb_peak_alloc', 'GPU Peak Allocated (MB)', 'gpu_mem_peak_alloc')
        _plot_metric('imgs_per_s_per_gb_vram', 'img/s per GB VRAM', 'imgs_per_s_per_gb_vram', ylog=True)

        return df

    for name, model in models_dict.items():
        results_by_model[name] = _benchmark_one_model(name, model)

    if make_comparative_plots and len(results_by_model) > 1:
        def _comparative_plot(metric_key, ylabel, filename, ylog=False):
            plt.figure(figsize=(10.5, 7))
            for name, df in results_by_model.items():
                plt.plot(df['samples'], df[metric_key], marker='o', label=name)
            plt.title(f'Comparison - {ylabel} vs Samples')
            plt.xlabel('Samples')
            plt.ylabel(ylabel)
            if ylog:
                plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()
            out_path = os.path.join(output_dir, f"COMPARE_{filename}.png")
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved comparative plot: {out_path}")

        _comparative_plot('time_s', 'Total Time (s)', 'time_s')
        _comparative_plot('imgs_per_s', 'Throughput (img/s)', 'throughput')
        _comparative_plot('time_ms_per_img', 'Latency (ms/img)', 'latency_ms_per_img')
        _comparative_plot('energy_kwh', 'Energy (kWh)', 'energy_kwh')
        _comparative_plot('kwh_per_img', 'Energy per Image (kWh/img)', 'kwh_per_img', ylog=True)
        _comparative_plot('emissions_kgco2eq', 'CO2 (kg)', 'co2_kg')
        _comparative_plot('kgco2_per_img', 'CO2 per Image (kg/img)', 'co2_per_img', ylog=True)
        _comparative_plot('imgs_per_kwh', 'Images per kWh (img/kWh)', 'imgs_per_kwh', ylog=True)
        _comparative_plot('imgs_per_joule', 'Images per Joule (img/J)', 'imgs_per_joule', ylog=True)
        _comparative_plot('avg_power_watts', 'Average Power (W)', 'avg_power_watts')
        _comparative_plot('cpu_mb_delta', 'CPU Memory Delta (MB)', 'cpu_mem_delta')
        _comparative_plot('gpu_mb_peak_reserved', 'GPU Peak Reserved (MB)', 'gpu_mem_peak_reserved')
        _comparative_plot('gpu_mb_peak_alloc', 'GPU Peak Allocated (MB)', 'gpu_mem_peak_alloc')
        _comparative_plot('imgs_per_s_per_gb_vram', 'img/s per GB VRAM', 'imgs_per_s_per_gb_vram', ylog=True)

    return results_by_model
# =========================================================================== #




def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoints in --checkpoint_dir')
    parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of training data to use (0.0 to 1.0)')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory containing checkpoints for resuming training')
    parser.add_argument('--use_optimal_thresholds', action='store_true', help='Compute optimal thresholds for metrics')
    parser.add_argument('--benchmark', action='store_true', help='Resume training from checkpoints in --checkpoint_dir')
    args = parser.parse_args()

    if args.resume and not args.checkpoint_dir:
        print("Error: --checkpoint_dir must be specified when --resume is used.")
        logging.error("No --checkpoint_dir specified with --resume.")
        sys.exit(1)
    if args.checkpoint_dir and not os.path.isdir(args.checkpoint_dir):
        print(f"Error: --checkpoint_dir {args.checkpoint_dir} does not exist or is not a directory.")
        logging.error(f"--checkpoint_dir {args.checkpoint_dir} does not exist or is not a directory.")
        sys.exit(1)

    hyperparams = {
        'general': {
            'main_python' : 'KD_Distill_Train_Loss_Ones_WDecay.py',
            'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            'batch_size': 128,
            'num_epochs': 100,
            'checkpoint_dir': 'checkpoints',
            'plot_dir': 'plots',
            'fraction': args.fraction,
            'threshold': 0.5,
            'use_optimal_thresholds': args.use_optimal_thresholds,
            'val_fraction': 0.25
        },
        'teacher': {
            'model' : 'densenet121',
            'loss_function': 'BCEWithLogitsLoss',
            'batch_size': 128,
            'learning_rate': 0.0001,
            'optimizer': 'AdamW',
            'weight_decay': 0.01,
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_factor': 0.1,
            'scheduler_patience': 4,
            'patience': 6
        },
        'student': {
            'model' : 'StudentMobileNetV2',
            'loss_function': 'BCEWithLogitsLoss',
            'batch_size': 128,
            'learning_rate': 0.001,
            'optimizer': 'AdamW',
            'weight_decay': 0.01,
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_factor': 0.1,
            'scheduler_patience': 4,
            'patience': 5
        },
        'distilled_student': {
            'learning_rate': 0.001,
            'batch_size': 128,
            'optimizer': 'AdamW',
            'weight_decay': 0.01,
            'loss_function': 'MultiLabelDistillLoss',
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_factor': 0.1,
            'scheduler_patience': 4,
            'patience': 5,
            'distillation_alpha': 0.7,
            'distillation_temperature': 6.0
        }
    }

    results_dir = f"OutPut_Zeros_{time.strftime('%Y%m%d-%H%M%S')}"
    checkpoint_save_dir = os.path.join(results_dir, hyperparams['general']['checkpoint_dir'])
    plot_dir = os.path.join(results_dir, hyperparams['general']['plot_dir'])
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    checkpoint_load_dir = args.checkpoint_dir if args.resume else checkpoint_save_dir

    data_dir = "/workspace/WORKS/DATA/"
    csv_train = os.path.join(data_dir, "CheXpert-v1.0-small", "train.csv")
    csv_valid = os.path.join(data_dir, "CheXpert-v1.0-small", "valid.csv")


    pathFileTrain , pathFileValid = create_final_train_val_files(csv_train, results_dir, hyperparams['general']['fraction'], hyperparams['general']['val_fraction'], 42)


    # train_dataset = CheXpertDataset(csv_train, data_dir, train_transform, fraction=hyperparams['general']['fraction']) 
    #val_dataset = CheXpertDataset(csv_valid, data_dir, val_transform)
    # train_loader = DataLoader(train_dataset, batch_size=hyperparams['general']['batch_size'], shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=hyperparams['general']['batch_size'], shuffle=False, num_workers=4)

    
    train_dataset = CheXpertDataset(pathFileTrain, data_dir, train_transform, fraction=hyperparams['general']['fraction'])
    val_dataset = CheXpertDataset(pathFileValid, data_dir, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['general']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['general']['batch_size'], shuffle=False, num_workers=4)


    hyperparams['general']['pos_weight'] = train_dataset.pos_weight.tolist()
    print(f"Computed pos_weight: {hyperparams['general']['pos_weight']}")
    logging.info(f"Computed pos_weight: {hyperparams['general']['pos_weight']}")

    params_path = os.path.join(results_dir, 'parameters.txt')
    with open(params_path, 'w') as f:
        f.write('General Parameters:\n')
        for key, value in hyperparams['general'].items():
            f.write(f'  {key}: {value}\n')
        if args.resume:
            f.write(f'  checkpoint_load_dir: {checkpoint_load_dir}\n')
        f.write('\nTeacher Parameters:\n')
        for key, value in hyperparams['teacher'].items():
            f.write(f'  {key}: {value}\n')
        f.write('\nStudent Parameters:\n')
        for key, value in hyperparams['student'].items():
            f.write(f'  {key}: {value}\n')
        f.write('\nDistilled Student Parameters:\n')
        for key, value in hyperparams['distilled_student'].items():
            f.write(f'  {key}: {value}\n')
    logging.info(f'Saved parameters: {params_path}')
    print(f'Saved parameters: {params_path}')


    print('General Parameters:')
    for key, value in hyperparams['general'].items():
        print(f'  {key}: {value}')
    if args.resume:
        print(f'  checkpoint_load_dir: {checkpoint_load_dir}')
    print('\nTeacher Parameters:')
    for key, value in hyperparams['teacher'].items():
        print(f'  {key}: {value}')
    print('\nStudent Parameters:')
    for key, value in hyperparams['student'].items():
        print(f'  {key}: {value}')
    print('\nDistilled Student Parameters:\n')
    for key, value in hyperparams['distilled_student'].items():
        print(f'  {key}: {value}')


    output_path = os.path.join(results_dir, 'training_output.txt')
    sys.stdout = ConsoleLogger(output_path)
    progress_file = os.path.join(results_dir, 'training_progress.txt')
    with open(progress_file, 'w') as f:
        f.write(f'Training started at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Using {hyperparams["general"]["fraction"]*100:.1f}% of training data\n')
        f.write(f'Computed pos_weight: {hyperparams["general"]["pos_weight"]}\n')
        if args.resume:
            f.write(f'Resuming from checkpoint directory: {checkpoint_load_dir}\n')

    logging.info(f'Checkpoint save directory: {checkpoint_save_dir}')
    logging.info(f'Checkpoint load directory: {checkpoint_load_dir}')
    logging.info(f'Plot directory: {plot_dir}')
    logging.info(f'Progress file: {progress_file}')
    print(f'Checkpoint save directory: {checkpoint_save_dir}')
    print(f'Checkpoint load directory: {checkpoint_load_dir}')
    print(f'Plot directory: {plot_dir}')
    print(f'Progress file: {progress_file}')

    device = torch.device(hyperparams['general']['device'])
    teacher_model = get_teacher_model(num_classes=len(PATHOLOGIES), dropout=0.39408860826430314).to(device)

    print("---------------------------  MODELO TEACHER -------------------------------")
    summary(teacher_model, (1, 3, 224, 224))
    print("----------------------------------------------------------------------------")
   
    print("----------------------------------------------------------------------------")
   

    student_model = get_student_model().to(device)
    print("---------------------------  MODELO STUDENT -------------------------------")
    summary(student_model, (1, 3, 224, 224))
    print("----------------------------------------------------------------------------")
   
    print("----------------------------------------------------------------------------")
   
    
    distilled_student_model = get_student_model().to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hyperparams['general']['pos_weight']).to(device))


    # criterion = WeightedFocalLoss(weights=torch.tensor(hyperparams['general']['pos_weight']).to(device), gamma=2, reduction='mean', eps=1e-8)
    
    # criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    

    # teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=hyperparams['teacher']['learning_rate'])
    # teacher_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     teacher_optimizer, mode='max', factor=hyperparams['teacher']['scheduler_factor'], 
    #     patience=hyperparams['teacher']['scheduler_patience'], verbose=True
    # )

    teacher_optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=hyperparams['teacher']['learning_rate'], weight_decay=hyperparams['teacher']['weight_decay'])
    # teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(teacher_model, T_max=10)

    teacher_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        teacher_optimizer, mode='min', factor=hyperparams['teacher']['scheduler_factor'], 
        patience=hyperparams['teacher']['scheduler_patience'], verbose=True
    )

    student_optimizer = optim.Adam(student_model.parameters(), lr=hyperparams['student']['learning_rate'], weight_decay=hyperparams['student']['weight_decay'])
    student_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        student_optimizer, mode='min', factor=hyperparams['student']['scheduler_factor'], 
        patience=hyperparams['student']['scheduler_patience'], verbose=True
    )

    # student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=hyperparams['student']['learning_rate'])
    # student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=10)


    # distilled_optimizer = optim.Adam(distilled_student_model.parameters(), lr=hyperparams['distilled_student']['learning_rate'])
    # distilled_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     distilled_optimizer, mode='max', factor=hyperparams['distilled_student']['scheduler_factor'], 
    #     patience=hyperparams['distilled_student']['scheduler_patience'], verbose=True
    # )

    distilled_optimizer = torch.optim.AdamW(distilled_student_model.parameters(), lr=hyperparams['distilled_student']['learning_rate'], weight_decay=hyperparams['distilled_student']['weight_decay'])
    # distilled_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(distilled_optimizer,  T_max=10)

    distilled_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        distilled_optimizer, mode='min', factor=hyperparams['distilled_student']['scheduler_factor'], 
        patience=hyperparams['distilled_student']['scheduler_patience'], verbose=True
    )


    # distillation_criterion = DistillationLoss_NN(
    #     alpha=hyperparams['distilled_student']['distillation_alpha'], distillation_mode='hard')
    
    distillation_criterion = MultiLabelDistillLoss(
        alpha=hyperparams['distilled_student']['distillation_alpha'],
        temperature=hyperparams['distilled_student']['distillation_temperature'], pos_weight=torch.tensor(hyperparams['general']['pos_weight']).to(device)
    )


    
    # Load best checkpoints for teacher and student
    teacher_start_epoch, teacher_history, teacher_thresholds = 0, None, np.array([0.5] * len(PATHOLOGIES))
    student_start_epoch, student_history, student_thresholds = 0, None, np.array([0.5] * len(PATHOLOGIES))
    distilled_start_epoch, distilled_history, distilled_thresholds = 0, None, np.array([0.5] * len(PATHOLOGIES))

    if args.resume and args.benchmark:
        # === Inference Efficiency Benchmarks (Validation) ===
        print("\n=== Inference Efficiency Benchmarks (Validation) ===")

        models_for_bench = {
            'Teacher': teacher_model,
            'Student': student_model,
            'Distilled_Student': distilled_student_model
        }

        bench_dir = os.path.join(results_dir, 'benchmarks_efficiency')
        bench_sizes = (1000, 5000, 10000, 20000, 30000, 40000)

        bench_results = run_inference_efficiency_benchmarks(
            models_dict=models_for_bench,
            dataset=val_dataset,         # usa tu valid_dataset creado arriba
            device=device,
            sizes=bench_sizes,
            batch_size=hyperparams['general']['batch_size'],
            num_workers=4,
            output_dir=bench_dir,
            assumed_power_watts=300.0,     # ajusta a tu hardware si no usas CodeCarbon
            emissions_factor_kg_per_kwh=0.5,
            make_comparative_plots=True
        )

        # === Energy Modeling Report (SOTA compute+memory model + calibration) ===
        
        energy_out_dir = os.path.join(results_dir, "energy_report")

        energy_summary_df = make_energy_report(
            models=models_for_bench,
            device=device,
            sample_input_shape=(1, 3, 224, 224),
            precision="fp16",              # si no usas AMP, cambia a "fp32"; si cuantizas, "int8"
            benchmark_results=bench_results,   # dict[str, DataFrame] devuelto arriba
            out_dir=energy_out_dir
        )

        print("\nEnergy report saved in:", energy_out_dir)
        print(energy_summary_df)
        exit()

    # Train or resume training
    if not args.resume or teacher_start_epoch < hyperparams['general']['num_epochs']:
        with open(progress_file, 'a') as f:
            f.write('\nTeacher Training\n')
        teacher_history, teacher_best_path, teacher_time, teacher_energy, teacher_emissions, teacher_thresholds = train_model(
            teacher_model, train_loader, val_loader, criterion, teacher_optimizer, teacher_scheduler, 
            hyperparams['general']['num_epochs'], device, 'teacher', checkpoint_save_dir, 'auc_roc', 
            hyperparams['teacher']['patience'], teacher_start_epoch, teacher_history, progress_file, plot_dir,
            threshold=hyperparams['general']['threshold'], compute_thresholds=hyperparams['general']['use_optimal_thresholds']
        )
    else:
        teacher_best_path = os.path.join(checkpoint_load_dir, 'best_teacher.pth')
        print(f"Skipping teacher training, using loaded model: {teacher_best_path}")

    if not args.resume or student_start_epoch < hyperparams['general']['num_epochs']:
        with open(progress_file, 'a') as f:
            f.write('\nStudent Training\n')
        student_history, student_best_path, student_time, student_energy, student_emissions, student_thresholds = train_model(
            student_model, train_loader, val_loader, criterion, student_optimizer, student_scheduler, 
            hyperparams['general']['num_epochs'], device, 'student', checkpoint_save_dir, 'auc_roc', 
            hyperparams['student']['patience'], student_start_epoch, student_history, progress_file, plot_dir,
            threshold=hyperparams['general']['threshold'], compute_thresholds=hyperparams['general']['use_optimal_thresholds']
        )
    else:
        student_best_path = os.path.join(checkpoint_load_dir, 'best_student.pth')
        print(f"Skipping student training, using loaded model: {student_best_path}")

    if not args.resume or teacher_start_epoch < hyperparams['general']['num_epochs']:
    
        with open(progress_file, 'a') as f:
            f.write('\nDistilled Student Training\n')
        distilled_history, distilled_best_path, distilled_time, distilled_energy, distilled_emissions, distilled_thresholds = train_distillation(
            teacher_model, distilled_student_model, train_loader, val_loader, 
            distillation_criterion, distilled_optimizer, distilled_scheduler, 
            hyperparams['general']['num_epochs'], device, checkpoint_save_dir, 
            hyperparams['distilled_student']['patience'], distilled_start_epoch, distilled_history, progress_file, plot_dir,
            threshold=hyperparams['general']['threshold'], compute_thresholds=hyperparams['general']['use_optimal_thresholds'],
            alpha=hyperparams['distilled_student']['distillation_alpha'], temperature=hyperparams['distilled_student']['distillation_temperature']
        )


    # Load best models
    teacher_model, _ , _ ,_ ,_, teacher_thresholds = load_checkpoint(teacher_model, teacher_optimizer, teacher_scheduler, teacher_best_path, device)
    student_model, _ , _ ,_ ,_, student_thresholds = load_checkpoint(student_model, student_optimizer, student_scheduler, student_best_path, device)
    distilled_student_model, _ , _ ,_ ,_, distilled_thresholds = load_checkpoint(distilled_student_model, distilled_optimizer, distilled_scheduler, distilled_best_path, device)



    print(f"Loaded best models from: {teacher_best_path}, {student_best_path}, {distilled_best_path}")



    with open(progress_file, 'a') as f:
        f.write(f"\nLoaded best models from: {teacher_best_path}, {student_best_path}, {distilled_best_path}\n")
        f.write(f"\nThresholds Teacher: {teacher_thresholds}\n")
        f.write(f"\nThresholds Student: {student_thresholds}\n")
        f.write(f"\nThresholds Distill Student: {distilled_thresholds}\n")


    # Plot final learning curves
    plot_learning_curves(teacher_history, 'Teacher', plot_dir)
    plot_learning_curves(student_history, 'Student', plot_dir)
    plot_learning_curves(distilled_history, 'Distilled Student', plot_dir)


    csv_valid = os.path.join(data_dir, "CheXpert-v1.0-small", "valid.csv")
    valid_dataset = CheXpertDataset(csv_valid, data_dir, val_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=hyperparams['general']['batch_size'], shuffle=False, num_workers=4)


    # Plot ROC and PR curves
    plot_roc_pr_curves(
        [teacher_model, student_model, distilled_student_model],
        ['Teacher', 'Student', 'Distilled Student'], val_loader, device,progress_file, f'{plot_dir}',
        thresholds=None, optimal = args.use_optimal_thresholds
    )

    # Plot ROC and PR curves
    plot_roc_pr_curves(
        [teacher_model, student_model, distilled_student_model],
        ['Teacher', 'Student', 'Distilled Student'], valid_loader, device, progress_file, f'{plot_dir}/Valid',
        thresholds=None,optimal =  args.use_optimal_thresholds
    )

    plot_combined_curves([teacher_model, student_model, distilled_student_model], ['Teacher', 'Student', 'Distilled Student'], val_loader, device, plot_dir)

    plot_combined_curves([teacher_model, student_model, distilled_student_model], ['Teacher', 'Student', 'Distilled Student'], valid_loader, device, f'{plot_dir}/Valid')
    
    

    # Evaluate and generate summary table
    with open(progress_file, 'a') as f:
        f.write('\nFinal Summary Metrics Table (Using Thresholds):\n')
        f.write(f'Teacher Thresholds: {teacher_thresholds}\n')
        f.write(f'Student Thresholds: {student_thresholds}\n')
        f.write(f'Distilled Student Thresholds: {distilled_thresholds}\n')
    summary_table, final_thresholds = evaluate_models(
        [teacher_model, student_model, distilled_student_model],
        ['Teacher', 'Student', 'Distilled Student'], val_loader, device, plot_dir, progress_file, 
        thresholds=[teacher_thresholds, student_thresholds, distilled_thresholds]
    )
    print("\nFinal Summary Metrics Table  Test Dataset:")
    print(summary_table)
    with open(progress_file, 'a') as f:
        f.write(summary_table.to_string() + '\n')

       # Evaluate and generate summary table
    with open(progress_file, 'a') as f:
        f.write('\nFinal Summary Metrics Table (Using Thresholds):\n')
        f.write(f'Teacher Thresholds: {teacher_thresholds}\n')
        f.write(f'Student Thresholds: {student_thresholds}\n')
        f.write(f'Distilled Student Thresholds: {distilled_thresholds}\n')
    summary_table, final_thresholds = evaluate_models(
        [teacher_model, student_model, distilled_student_model],
        ['Teacher', 'Student', 'Distilled Student'], valid_loader, device, f'{plot_dir}/Valid', progress_file, 
        thresholds=[teacher_thresholds, student_thresholds, distilled_thresholds]
    )

    print("\nFinal Summary Metrics Table  Valid Dataset:")
    print(summary_table)
    with open(progress_file, 'a') as f:
        f.write(summary_table.to_string() + '\n')

    # Print and save overall training summary
    overall_summary = (f"\nOverall Training Summary:\n"
                      f"Teacher - Time: {teacher_time:.2f}s, Energy: {teacher_energy:.6f} kWh, Emissions: {teacher_emissions:.6f} kgCO₂eq\n"
                      f"Student - Time: {student_time:.2f}s, Energy: {student_energy:.6f} kWh, Emissions: {student_emissions:.6f} kgCO₂eq\n"
                      f"Distilled Student - Time: {distilled_time:.2f}s, Energy: {distilled_energy:.6f} kWh, Emissions: {distilled_emissions:.6f} kgCO₂eq\n")
    print(overall_summary)
    with open(progress_file, 'a') as f:
        f.write(overall_summary)
    logging.info(overall_summary)

    sys.stdout.close()
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()