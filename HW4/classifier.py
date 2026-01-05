import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet50
import numpy as np
from pathlib import Path


class Classifier:
    def __init__(self, cfg):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_classes = cfg.model.num_classes
        self.learning_rate = cfg.train.learning_rate

        self.model = self._build_model(cfg.model.model_name, self.num_classes)
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer(cfg.train.optimizer, cfg.train.weight_decay, cfg.train.momentum)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def _build_model(self, model_name, num_classes):
        if model_name.lower() == 'resnet18':
            model = resnet18(weights=None)
        elif model_name.lower() == 'resnet50':
            model = resnet50(weights=None)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model.fc = nn.Linear(model.fc.in_features, num_classes)
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