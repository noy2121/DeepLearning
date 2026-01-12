# imports

from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize

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

        pbar = tqdm(train_loader, desc='Training', leave=False)
        for inputs, labels in pbar:
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

            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': f'{100. * correct / total:.2f}%'})

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc
    
    def evaluate(self, val_loader, desc='Validation'):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(val_loader, desc=desc, leave=False)
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'acc': f'{100. * correct / total:.2f}%'})

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs, early_stopping_patience=None):
        best_val_loss = float('inf')
        patience_counter = 0

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

            # Early stopping logic
            if early_stopping_patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f'\nEarly stopping triggered after {epoch+1} epochs (patience={early_stopping_patience})')
                        print(f'Best validation loss: {best_val_loss:.4f}')
                        break
    
    def test(self, test_loader):
        test_loss, test_acc = self.evaluate(test_loader, desc='Testing')
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
        epochs = list(range(1, len(self.train_losses) + 1))

        # Create subplots with 1 row and 2 columns
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training and Validation Loss', 'Training and Validation Accuracy'),
            horizontal_spacing=0.12
        )

        # Plot 1: Loss curves
        fig.add_trace(
            go.Scatter(x=epochs, y=self.train_losses, mode='lines+markers',
                      name='Train Loss', line=dict(color='#636EFA'), marker=dict(symbol='circle')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=self.val_losses, mode='lines+markers',
                      name='Val Loss', line=dict(color='#EF553B'), marker=dict(symbol='square')),
            row=1, col=1
        )

        # Plot 2: Accuracy curves
        fig.add_trace(
            go.Scatter(x=epochs, y=self.train_accs, mode='lines+markers',
                      name='Train Acc', line=dict(color='#636EFA'), marker=dict(symbol='circle')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=self.val_accs, mode='lines+markers',
                      name='Val Acc', line=dict(color='#EF553B'), marker=dict(symbol='square')),
            row=1, col=2
        )

        # Update axes
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)

        # Update layout
        fig.update_layout(
            height=400,
            width=1000,
            showlegend=True,
            font=dict(size=11)
        )

        if save_path:
            output_path = f'{save_path}/training_curves.html'
            fig.write_html(output_path)
            print(f'Training curves saved to {output_path}')

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

        # Create Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names if class_names else list(range(len(cm))),
            y=class_names if class_names else list(range(len(cm))),
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Count")
        ))

        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='True',
            width=700,
            height=700,
            font=dict(size=12),
            yaxis=dict(autorange='reversed')  # Reverse y-axis to match standard CM orientation
        )

        if save_path:
            output_path = f'{save_path}/confusion_matrix.html'
            fig.write_html(output_path)
            print(f'Confusion matrix saved to {output_path}')

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

    def compute_roc_curve(self, test_loader):

        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_probs = np.vstack(all_probs)
        all_labels = np.array(all_labels)

        # Binarize labels for multi-class ROC
        all_labels_bin = label_binarize(all_labels, classes=range(self.num_classes))

        # Compute ROC curve and AUC for each class
        fpr_dict = {}
        tpr_dict = {}
        auc_dict = {}

        for i in range(self.num_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
            auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

        # Compute macro-average ROC curve
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(self.num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(self.num_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])

        mean_tpr /= self.num_classes

        # Compute macro-average AUC
        macro_auc = auc(all_fpr, mean_tpr)

        print(f'\nROC AUC (macro-average): {macro_auc:.4f}')

        return all_fpr.tolist(), mean_tpr.tolist(), macro_auc

