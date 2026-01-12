# CIFAR-10 Image Classification

Deep learning project for image classification on CIFAR-10 using PyTorch with support for custom CNNs and pretrained models.

## Features

- **Multiple architectures**: ResNet18, ResNet50, CustomCNN
- **Transfer learning**: Pretrained models with optional backbone freezing
- **Interactive visualizations**: Plotly-based plots (training curves, confusion matrix, ROC curves)
- **Model persistence**: Save/load trained weights
- **Configurable training**: Early stopping, dropout, batch normalization

## Quick Start

```bash
# Install dependencies
pip install torch torchvision plotly scikit-learn pandas omegaconf tqdm

# Train a model
python main.py

# Or use Jupyter notebook
jupyter notebook main.ipynb
```

## Configuration

Edit `conf/config.yaml`:

```yaml
mode: pretrained              # pretrained or scratch
load_weights: false           # Set true to load existing weights

model:
  model_name: ResNet18        # ResNet18 | ResNet50 | CustomCNN

train:
  batch_size: 128
  learning_rate: 0.001
  num_epochs: 50
  optimizer: adam             # adam | sgd
  early_stopping_patience: 5
  freeze: false               # Freeze pretrained backbone (pretrained mode only)
```

## CustomCNN Configuration

Simplified list-based config with automatic padding and 2x2 max pooling:

```yaml
model:
  model_name: CustomCNN
  cnn:
    input_channels: 3
    use_batch_norm: true
    use_dropout: true
    dropout_rate: 0.3
    l2_reg: 0.0001
    num_layers: [32, 64, 128]      # Output channels per layer
    kernel_size: [3, 3, 3]         # Kernel size per layer
    stride: [1, 1, 1]              # Stride per layer
    fc_layers: [256, 128]          # Fully connected layer sizes
```

**Architecture examples:**

```yaml
# Shallow (2 layers)
num_layers: [32, 64]
kernel_size: [3, 3]
stride: [1, 1]
fc_layers: [128]

# Medium (3 layers)
num_layers: [64, 128, 256]
kernel_size: [5, 3, 3]
stride: [1, 1, 1]
fc_layers: [512, 256]

# Deep (4 layers)
num_layers: [32, 64, 128, 256]
kernel_size: [3, 3, 3, 3]
stride: [1, 1, 1, 1]
fc_layers: [512, 256, 128]
```

## Usage Examples

### Train from scratch
```python
from omegaconf import OmegaConf
from src.data_pipeline import CIFAR10Pipeline
from src.runner import run_experiment

cfg = OmegaConf.load("conf/config.yaml")
cfg.mode = 'scratch'
cfg.model.model_name = 'CustomCNN'

pipeline = CIFAR10Pipeline(cfg)
train_loader, val_loader, test_loader = pipeline.setup()

results = run_experiment(cfg, train_loader, val_loader, test_loader,
                        class_names=['airplane', 'automobile', 'bird', 'cat', 'deer',
                                   'dog', 'frog', 'horse', 'ship', 'truck'])
```

### Load and evaluate existing model
```python
from src.runner import build_model_path

cfg.load_weights = True
cfg.mode = 'pretrained'
cfg.model.model_name = 'ResNet18'

model_path = build_model_path(cfg)
print(f"Loading from: {model_path}")

results = run_experiment(cfg, train_loader, val_loader, test_loader,
                        class_names, load_weights=True)
```

### Compare multiple models
```python
from src.runner import compare_experiments

experiment_results = {
    'ResNet18_pretrained': results1,
    'ResNet18_frozen': results2,
    'CustomCNN_scratch': results3
}

compare_experiments(experiment_results, save_dir="results/comparison")
```

## Project Structure

```
├── conf/
│   └── config.yaml           # Configuration file
├── src/
│   ├── classifier.py         # Model wrapper with training logic
│   ├── cnn_model.py          # CustomCNN implementation
│   ├── data_pipeline.py      # CIFAR-10 data loading
│   └── runner.py             # Experiment orchestration
├── main.py                   # Training script
├── main.ipynb                # Jupyter notebook
├── checkpoints/              # Saved model weights
└── results/                  # Plots and metrics
```

## Notes

- All visualizations use Plotly (interactive HTML files)
- ROC curves computed using macro-averaging for multi-class
- Early stopping monitors validation loss
- Training curves only saved when training (not when loading weights)
