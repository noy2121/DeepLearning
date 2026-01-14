# imports

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from .classifier import Classifier


class ExperimentRunner:

    def __init__(self, cfg, exp_name, train_loader, val_loader, test_loader, class_names, load_weights=None):

        self.cfg = cfg
        self.exp_name = exp_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.load_weights = load_weights if load_weights is not None else cfg.get('load_weights', False)

        # Determine mode settings
        self.mode = cfg.mode
        if self.mode == 'pretrained':
            self.pretrained = True
            self.freeze_backbone = cfg.train.get('freeze', False)
            self.exp_suffix = "_pretrained" + ("_frozen" if self.freeze_backbone else "")
        elif self.mode == 'scratch':
            self.pretrained = False
            self.freeze_backbone = False
            self.exp_suffix = "_scratch"
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'pretrained' or 'scratch'")

        # Setup paths
        self._setup_paths()

        # Initialize classifier
        self._init_classifier()

    def _setup_paths(self):
        results_dir = self.cfg.results_dir
        checkpoints_dir = self.cfg.checkpoints_dir
        optimizer = self.cfg.train.optimizer

        self.results_folder = f"{results_dir}/{self.exp_name}{self.exp_suffix}_{optimizer}"
        self.weights_folder = f"{checkpoints_dir}/{self.exp_name}{self.exp_suffix}_{optimizer}"
        self.weights_path = f"{self.weights_folder}/weights.pth"

        Path(self.results_folder).mkdir(parents=True, exist_ok=True)
        Path(self.weights_folder).mkdir(parents=True, exist_ok=True)

    def _init_classifier(self):
        model_name = self.cfg.model.model_name

        # Get CNN config if needed
        cnn_config = None
        if model_name.lower() == 'customcnn':
            cnn_config = self.cfg.model.cnn

        print(f"\nInitializing {model_name}...")
        print(f"Mode: {self.mode}" + (f" (freeze={'on' if self.freeze_backbone else 'off'})" if self.mode == 'pretrained' else ""))

        self.classifier = Classifier(
            model_name=model_name,
            num_classes=self.cfg.model.num_classes,
            learning_rate=self.cfg.train.learning_rate,
            optimizer_name=self.cfg.train.optimizer,
            weight_decay=self.cfg.train.weight_decay,
            momentum=self.cfg.train.momentum,
            cnn_config=cnn_config,
            pretrained=self.pretrained,
            freeze_backbone=self.freeze_backbone
        )

    def load_model_weights(self):
        if Path(self.weights_path).exists():
            print(f"\nLoading weights from {self.weights_path}...")
            self.classifier.load_model(self.weights_path)
        else:
            raise ValueError(f"Weights file not found at {self.weights_path}")

    def train(self):
        early_stopping_patience = self.cfg.train.early_stopping_patience
        print(f"\nTraining for {self.cfg.train.num_epochs} epochs...")

        self.classifier.train(
            self.train_loader,
            self.val_loader,
            self.cfg.train.num_epochs,
            early_stopping_patience=early_stopping_patience
        )

        # Save model weights
        self.classifier.save_model(self.weights_path)

        # Plot training curves
        self.classifier.plot_training_curves(save_path=self.results_folder)

    def evaluate(self):

        print(f"\nEvaluating on test set...")

        # Compute metrics
        test_loss, test_acc = self.classifier.test(self.test_loader)
        precision, recall, f1 = self.classifier.compute_metrics(self.test_loader)
        fpr, tpr, auc_score = self.classifier.compute_roc_curve(self.test_loader)

        # Plot confusion matrix
        self.classifier.plot_confusion_matrix(
            self.test_loader,
            class_names=self.class_names,
            save_path=self.results_folder
        )

        # Store results
        model_name_suffix = f"{self.cfg.model.model_name}{self.exp_suffix}"
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
        results_csv_path = f"{self.results_folder}/results.csv"
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

    def run(self):

        if self.load_weights:
            self.load_model_weights()
        else:
            self.train()

        return self.evaluate()

    @staticmethod
    def compare(experiment_results, save_dir="results"):

        if len(experiment_results) < 2:
            print("Need at least 2 experiments to compare.")
            return

        models = list(experiment_results.keys())
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Metrics Bar Chart 
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
            font=dict(size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(range=[0, 105])
        )

        metrics_path = f"{save_dir}/metrics_comparison.html"
        fig1.write_html(metrics_path)
        print(f"\nMetrics comparison saved to {metrics_path}")

        # ROC Curves
        has_roc_data = all('fpr' in experiment_results[m] and 'tpr' in experiment_results[m]
                          for m in models)

        if has_roc_data:
            fig2 = go.Figure()

            # Add diagonal reference line
            fig2.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
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
                    x=fpr, y=tpr,
                    mode='lines',
                    name=label,
                    line=dict(color=colors_roc[i % len(colors_roc)], width=2)
                ))

            fig2.update_layout(
                title='ROC Curves Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                font=dict(size=12),
                legend=dict(orientation="v", y=0.01, x=0.99),
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )

            roc_path = f"{save_dir}/roc_comparison.html"
            fig2.write_html(roc_path)
            print(f"ROC curves saved to {roc_path}")
        else:
            print("\nWarning: ROC data not found. Skipping ROC plot.")

        # Save results to csv
        data_list = []
        for model, results in experiment_results.items():
            data_list.append({
                "Model": model,
                "Test Accuracy": results['test_acc'],
                "Precision": results['precision'] * 100,
                "Recall": results['recall'] * 100,
                "F1": results['f1'] * 100
            })

        df = pd.DataFrame(data_list)
        df.to_csv(f"{save_dir}/comparison_results.csv", index=False)

        print(df)