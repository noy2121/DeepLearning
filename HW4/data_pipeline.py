import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


class CIFAR10Pipeline:
    def __init__(self, cfg, num_workers=4, download=True, val_split=0.1):
        self.batch_size = cfg.train.batch_size
        self.num_workers = num_workers
        self.download = download
        self.val_split = val_split
        self.mean = cfg.data.mean
        self.std = cfg.data.std 

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.num_classes = 10
        
    def setup(self):
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=self.download, transform=self.train_transform
        )
        
        val_size = int(len(train_dataset) * self.val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        val_dataset.dataset.transform = self.test_transform
        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=self.download, transform=self.test_transform
        )
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        
        return self.train_loader, self.val_loader, self.test_loader