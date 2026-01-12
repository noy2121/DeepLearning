# imports

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from .classifier import Classifier


def run_experiment(cfg, train_loader, val_loader, test_loader, class_names, pretrained=False, freeze_backbone=False):

    model_name = cfg.model.model_name

    # Pass CNN config if using CustomCNN model
    cnn_config = None
    if model_name.lower() == 'customcnn':
        cnn_config = cfg.model.cnn

    # Initialize classifier
    print(f"\nInitializing {model_name}...")
    classifier = Classifier(
        model_name=model_name,
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.train.learning_rate,
        optimizer_name=cfg.train.optimizer,
        weight_decay=cfg.train.weight_decay,
        momentum=cfg.train.momentum,
        cnn_config=cnn_config,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )

    # Train
    early_stopping_patience = cfg.train.get('early_stopping_patience', None)
    print(f"\nTraining {model_name} for {cfg.train.num_epochs} epochs...")
    if early_stopping_patience:
        print(f"Early stopping enabled with patience={early_stopping_patience}")
    classifier.train(train_loader, val_loader, cfg.train.num_epochs, early_stopping_patience=early_stopping_patience)

    # Evaluate on test set
    print(f"\nEvaluating {model_name} on test set...")
    test_loss, test_acc = classifier.test(test_loader)

    # Compute detailed metrics
    precision, recall, f1 = classifier.compute_metrics(test_loader)

    # Create experiment name with pretrained/frozen suffix
    exp_suffix = ""
    if pretrained:
        exp_suffix = "_pretrained"
        if freeze_backbone:
            exp_suffix += "_frozen"

    # Save model
    model_path = f"checkpoints/{model_name}{exp_suffix}_{cfg.train.optimizer}.pth"
    classifier.save_model(model_path)

    # Plot training curves
    curves_path = f"results/{model_name}{exp_suffix}_{cfg.train.optimizer}_curves.png"
    classifier.plot_training_curves(save_path=curves_path)

    # Plot confusion matrix
    cm_path = f"results/{model_name}{exp_suffix}_{cfg.train.optimizer}_confusion.png"
    classifier.plot_confusion_matrix(test_loader, class_names=class_names, save_path=cm_path)

    # Store results
    results = {
        'model_name': f"{model_name}{exp_suffix}",
        'test_loss': test_loss,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_losses': classifier.train_losses,
        'val_losses': classifier.val_losses,
        'train_accs': classifier.train_accs,
        'val_accs': classifier.val_accs,
        'num_params': classifier.model.get_num_parameters() if hasattr(classifier.model, 'get_num_parameters') else sum(p.numel() for p in classifier.model.parameters()),
        'trainable_params': sum(p.numel() for p in classifier.model.parameters() if p.requires_grad)
    }

    print(f"\n{model_name}{exp_suffix} Results:")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Total Parameters: {results['num_params']:,}")
    print(f"  Trainable Parameters: {results['trainable_params']:,}")

    return results


def compare_experiments(experiment_results, save_path="results/model_comparison.png"):
    if len(experiment_results) < 2:
        print("Need at least 2 experiments to compare.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Test Accuracy Comparison
    ax1 = axes[0, 0]
    models = list(experiment_results.keys())
    test_accs = [experiment_results[m]['test_acc'] for m in models]
    ax1.bar(models, test_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Test Accuracy Comparison')
    ax1.grid(True, alpha=0.3)
    for i, v in enumerate(test_accs):
        ax1.text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom')

    # Plot 2: F1-Score Comparison
    ax2 = axes[0, 1]
    f1_scores = [experiment_results[m]['f1'] for m in models]
    ax2.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score Comparison')
    ax2.grid(True, alpha=0.3)
    for i, v in enumerate(f1_scores):
        ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

    # Plot 3: Training Curves
    ax3 = axes[1, 0]
    for model in models:
        epochs = range(1, len(experiment_results[model]['train_losses']) + 1)
        ax3.plot(epochs, experiment_results[model]['val_accs'], label=f'{model} Val', marker='o', markersize=3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation Accuracy (%)')
    ax3.set_title('Validation Accuracy Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Model Parameters
    ax4 = axes[1, 1]
    num_params = [experiment_results[m]['num_params'] for m in models]
    ax4.bar(models, num_params, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
    ax4.set_ylabel('Number of Parameters')
    ax4.set_title('Model Size Comparison')
    ax4.grid(True, alpha=0.3)
    for i, v in enumerate(num_params):
        ax4.text(i, v + max(num_params)*0.02, f'{v:,}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    plt.show()

    # Print comparison table
    print("\n" + "="*80)
    print("Model Comparison Summary")
    print("="*80)
    print(f"{'Model':<20} {'Test Acc':<12} {'F1-Score':<12} {'Parameters':<15}")
    print("-"*80)
    for model in models:
        results = experiment_results[model]
        print(f"{model:<20} {results['test_acc']:>10.2f}% {results['f1']:>11.4f} {results['num_params']:>14,}")
    print("="*80)
