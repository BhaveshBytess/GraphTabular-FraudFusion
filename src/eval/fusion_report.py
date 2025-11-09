"""Generate fusion comparison report with baseline metrics."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def create_comparison_report(
    fusion_metrics: dict,
    baseline_csv: str = "reports/metrics_summary.csv",
    output_dir: str = "reports"
):
    """
    Create side-by-side comparison of baseline vs fusion models.
    
    Args:
        fusion_metrics: Dictionary with fusion results
        baseline_csv: Path to baseline metrics CSV
        output_dir: Where to save plots and tables
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load baseline metrics
    baseline_df = pd.read_csv(baseline_csv)
    
    # Append fusion metrics
    fusion_row = {
        'timestamp': pd.Timestamp.now().timestamp(),
        'experiment': 'graph-tabular-fusion',
        'model': 'XGB+Embeddings',
        'split': 'test',
        'pr_auc': fusion_metrics['test']['pr_auc'],
        'roc_auc': fusion_metrics['test']['roc_auc'],
        'f1': fusion_metrics['test']['best_f1'],
        'recall@1%': fusion_metrics['test'].get('recall@1%', 0)
    }
    
    # Combine
    comparison_df = pd.concat([baseline_df, pd.DataFrame([fusion_row])], ignore_index=True)
    
    # Save updated CSV
    comparison_df.to_csv(output_path / "metrics_summary.csv", index=False)
    
    # Create comparison plots
    _plot_model_comparison(comparison_df, output_path)
    
    print(f"✅ Comparison report saved to {output_dir}")
    
    return comparison_df


def _plot_model_comparison(df: pd.DataFrame, output_path: Path):
    """Create comparison bar charts."""
    # Filter to test split only
    test_df = df[df['split'] == 'test'].copy()
    
    # Plot PR-AUC comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=test_df, x='model', y='pr_auc', palette='viridis')
    plt.title('PR-AUC Comparison (Test Set)')
    plt.ylabel('PR-AUC')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path / 'plots' / 'pr_auc_comparison.png', dpi=300)
    plt.close()
    
    # Plot F1 comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=test_df, x='model', y='f1', palette='viridis')
    plt.title('F1 Score Comparison (Test Set)')
    plt.ylabel('F1 Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path / 'plots' / 'f1_comparison.png', dpi=300)
    plt.close()
    
    print("✅ Comparison plots saved")


if __name__ == "__main__":
    print("Fusion report generator - use via notebooks or scripts")
