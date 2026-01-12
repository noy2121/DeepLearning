# imports

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from .classifier import Classifier


def build_model_path(cfg, mode=None):

    if mode is None:
        mode = cfg.mode

    model_name = cfg.model.model_name

    if mode == 'pretrained':
        freeze_backbone = cfg.train.get('freeze', False)
        exp_suffix = "_pretrained" + ("_frozen" if freeze_backbone else "")
    else:  # scratch
        exp_suffix = "_scratch"

    checkpoints_dir = cfg.get('checkpoints', 'checkpoints')
    weights_folder = f"{checkpoints_dir}/{model_name}{exp_suffix}_{cfg.train.optimizer}"
    model_path = f"{weights_folder}/weights.pth"

    return model_path


def train_model(classifier, cfg, train_loader, val_loader, results_output_folder):

    early_stopping_patience = cfg.train.early_stopping_patience
    print(f"\nTraining {classifier.model_name} for {cfg.train.num_epochs} epochs...")

    classifier.train(train_loader, val_loader, cfg.train.num_epochs, early_stopping_patience=early_stopping_patience)

    # Save model weights
    checkpoints_dir = cfg.checkpoints_dir
    weights_folder = f"{checkpoints_dir}/{classifier.model_name}_weights"
    Path(weights_folder).mkdir(parents=True, exist_ok=True)
    model_path = f"{weights_folder}/weights.pth"
    classifier.save_model(model_path)

    # Plot training curves
    classifier.plot_training_curves(save_path=results_output_folder)


def evaluate_model(classifier, test_loader, class_names, results_output_folder, model_name_suffix):

    print(f"\nEvaluating {classifier.model_name} on test set...")

    # Compute metrics
    test_loss, test_acc = classifier.test(test_loader)
    precision, recall, f1 = classifier.compute_metrics(test_loader)
    fpr, tpr, auc_score = classifier.compute_roc_curve(test_loader)

    # Plot confusion matrix
    classifier.plot_confusion_matrix(test_loader, class_names=class_names, save_path=results_output_folder)

    # Store results
    results = {
        'model_name': model_name_suffix,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc_score
    }

    # Save results to CSV
    df = pd.DataFrame.from_dict([results])
    results_csv_path = f"{results_output_folder}/results.csv"
    df.to_csv(results_csv_path, index=False)

    # Print results
    print(f"\n{model_name_suffix} Results:")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC: {auc_score:.4f}")

    return results


def run_experiment(cfg, train_loader, val_loader, test_loader, class_names, mode=None, load_weights=None):

    if mode is None:
        mode = cfg.mode

    if load_weights is None:
        load_weights = cfg.get('load_weights', False)

    model_name = cfg.model.model_name

    # Determine pretrained and freeze_backbone based on mode
    if mode == 'pretrained':
        pretrained = True
        freeze_backbone = cfg.train.get('freeze', False)
    elif mode == 'scratch':
        pretrained = False
        freeze_backbone = False
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'pretrained' or 'scratch'")

    cnn_config = None
    if model_name.lower() == 'customcnn':
        cnn_config = cfg.model.cnn

    # Initialize classifier
    print(f"\nInitializing {model_name}...")
    print(f"Mode: {mode}" + (f" (freeze={'on' if freeze_backbone else 'off'})" if mode == 'pretrained' else ""))

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

    # Build experiment suffix for naming
    if mode == 'pretrained':
        exp_suffix = "_pretrained" + ("_frozen" if freeze_backbone else "")
    else:  # scratch
        exp_suffix = "_scratch"


    results_dir = cfg.results_dir
    results_output_folder = f"{results_dir}/{model_name}{exp_suffix}_{cfg.train.optimizer}"
    Path(results_output_folder).mkdir(parents=True, exist_ok=True)

    # Load weights or train
    if load_weights:
        weights_path = build_model_path(cfg, mode=mode)
        if Path(weights_path).exists():
            print(f"\nLoading weights from {weights_path}...")
            classifier.load_model(weights_path)
        else:
            raise ValueError(f"Weights file not found at {weights_path}")
    else:
        original_model_name = classifier.model_name
        classifier.model_name = f"{model_name}{exp_suffix}_{cfg.train.optimizer}"

        # Train model
        train_model(classifier, cfg, train_loader, val_loader, results_output_folder)
        classifier.model_name = original_model_name

    # Evaluate model
    model_name_suffix = f"{model_name}{exp_suffix}"
    results = evaluate_model(
        classifier=classifier,
        test_loader=test_loader,
        class_names=class_names,
        results_output_folder=results_output_folder,
        model_name_suffix=model_name_suffix
    )

    return results


def compare_experiments(experiment_results, save_dir="results"):

    if len(experiment_results) < 2:
        print("Need at least 2 experiments to compare.")
        return

    models = list(experiment_results.keys())
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    metrics_data = {
        'Accuracy': [experiment_results[m]['test_acc'] for m in models],
        'F1-Score': [experiment_results[m]['f1'] * 100 for m in models], 
        'Precision': [experiment_results[m]['precision'] * 100 for m in models],
        'Recall': [experiment_results[m]['recall'] * 100 for m in models]
    }

    fig1 = go.Figure()

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        fig1.add_trace(go.Bar(
            name=metric_name,
            x=models,
            y=values,
            text=[f'{v:.2f}%' for v in values],
            textposition='outside',
            marker_color=colors[i]
        ))

    fig1.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score (%)',
        barmode='group',
        height=600,
        width=1000,
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(range=[0, 105])
    )

    metrics_path = f"{save_dir}/metrics_comparison.html"
    fig1.write_html(metrics_path)
    print(f"\nMetrics comparison saved to {metrics_path}")

    # Check if ROC data is available
    has_roc_data = all('fpr' in experiment_results[m] and 'tpr' in experiment_results[m]
                       for m in models)

    if has_roc_data:
        fig2 = go.Figure()

        # Add diagonal reference line
        fig2.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', dash='dash'),
            showlegend=True
        ))

        colors_roc = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
        for i, model in enumerate(models):
            fpr = experiment_results[model]['fpr']
            tpr = experiment_results[model]['tpr']
            auc = experiment_results[model].get('auc', None)

            label = f"{model}" + (f" (AUC={auc:.4f})" if auc else "")

            fig2.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=label,
                line=dict(color=colors_roc[i % len(colors_roc)], width=2)
            ))

        fig2.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=600,
            width=800,
            font=dict(size=12),
            legend=dict(
                orientation="v",
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            ),
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )

        roc_path = f"{save_dir}/roc_comparison.html"
        fig2.write_html(roc_path)
        print(f"ROC curves saved to {roc_path}")
    else:
        print("\nWarning: ROC data (fpr, tpr) not found in experiment results. Skipping ROC plot.")
        print("To generate ROC curves, add 'fpr', 'tpr', and optionally 'auc' to your results dict.")

    # Print comparison table
    print("\n" + "="*80)
    print("Model Comparison Summary")
    print("="*80)
    print(f"{'Model':<25} {'Test Acc':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    for model in models:
        results = experiment_results[model]
        print(f"{model:<25} {results['test_acc']:>10.2f}% "
              f"{results['precision']*100:>10.2f}% {results['recall']*100:>10.2f}% {results['f1']*100:>10.2f}%")
    print("="*80)
