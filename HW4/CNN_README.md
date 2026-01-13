# CIFAR-10 Image Classification

Deep learning project for image classification on CIFAR-10 using PyTorch with support for custom CNNs and pretrained models.

## Features

- **Multiple architectures**: ResNet18, ResNet50, CustomCNN
- **Transfer learning**: Pretrained models with optional backbone freezing
- **Interactive visualizations**: Plotly-based plots (training curves, confusion matrix, ROC curves)
- **Model persistence**: Save/load trained weights
- **Configurable training**: Using Hydra config framework one can control all parameters easily

## Requirements

```bash
# Install dependencies
pip install torch torchvision plotly scikit-learn pandas numpy omegaconf tqdm
```

## Colab Installation
If you're running the code with google colab, please make sure to:
- Place `src`, `conf` folder in your root dir (e.g. HW4)
- Place `main.ipynb` file in your root dir
- Change your working directory to your root dir. You can do it using colab magic commands: `%cd path/to/root/dir`

## Usage Examples
Our app is simple, easy-to-use, and versatile. Decide the experiment type and parameters using the config file (`conf/config.yaml`), and follow along the main.ipynb notebook.


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


### Compare multiple models
You can also compare multiple experiment using the relevant cell inside `main.ipynb` or run the following:

```python
from src.runner import compare_experiments

experiment_results = {
    'ResNet18_pretrained': results1,
    'ResNet18_frozen': results2,
    'CustomCNN_scratch': results3
}

compare_experiments(experiment_results, save_dir="results/comparison")
```

If you want to comapre an existing models, make sure to set `load_weights=true`!


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
