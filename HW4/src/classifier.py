# imports

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns

from .cnn_model import CustomCNN


class Classifier:
    def __init__(self, model_name, num_classes, learning_rate, optimizer_name, weight_decay, momentum, cnn_config=None, pretrained=False, freeze_backbone=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        self.model = self._build_model(model_name, num_classes, cnn_config, pretrained, freeze_backbone)
        self.model = self.model.to(self.device)

        # Use CNN's l2_reg if CustomCNN, otherwise use provided weight_decay
        if model_name.lower() == 'customcnn' and cnn_config and 'l2_reg' in cnn_config:
            weight_decay = cnn_config['l2_reg']

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer(optimizer_name, weight_decay, momentum)

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def _build_model(self, model_name, num_classes, cnn_config=None, pretrained=False, freeze_backbone=False):
        if model_name.lower() == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)

            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False

            model.fc = nn.Linear(model.fc.in_features, num_classes)

            status = "pretrained" if pretrained else "from scratch"
            frozen_status = " (backbone frozen)" if freeze_backbone else ""
            print(f"Built ResNet18 {status}{frozen_status} with {sum(p.numel() for p in model.parameters()):,} total parameters")
            print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        elif model_name.lower() == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)

            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False

            model.fc = nn.Linear(model.fc.in_features, num_classes)

            status = "pretrained" if pretrained else "from scratch"
            frozen_status = " (backbone frozen)" if freeze_backbone else ""
            print(f"Built ResNet50 {status}{frozen_status} with {sum(p.numel() for p in model.parameters()):,} total parameters")
            print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        elif model_name.lower() == 'customcnn':
            if cnn_config is None:
                raise ValueError("cnn_config must be provided for CustomCNN model")
            cnn_config['num_classes'] = num_classes
            model = CustomCNN(cnn_config)
            print(f"Built CustomCNN with {model.get_num_parameters():,} parameters")
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return model
    
    def _build_optimizer(self, optimizer_name, weight_decay, momentum):
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, 
                           momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f'Epoch [{epoch+1}/{num_epochs}] '
                  f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
    
    def test(self, test_loader):
        test_loss, test_acc = self.evaluate(test_loader)
        print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%')
        return test_loss, test_acc
    
    def save_model(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }, path)
        print(f'Model saved to {path}')
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']
        print(f'Model loaded from {path}')
    
    def plot_training_curves(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        ax1.plot(epochs, self.train_losses, label='Train Loss', marker='o')
        ax1.plot(epochs, self.val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, self.train_accs, label='Train Acc', marker='o')
        ax2.plot(epochs, self.val_accs, label='Val Acc', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Training curves saved to {save_path}')
        
        plt.show()
    
    def plot_confusion_matrix(self, test_loader, class_names=None, save_path=None):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Confusion matrix saved to {save_path}')
        
        plt.show()
        
        return all_preds, all_labels
    
    def compute_metrics(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        print(f'\nTest Set Metrics:')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        
        return precision, recall, f1
        
        