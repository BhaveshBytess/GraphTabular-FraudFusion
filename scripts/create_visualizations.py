"""
Create comprehensive visualizations for the fusion project README.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = Path("reports/plots")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Creating Comprehensive Visualizations")
print("=" * 60)

# Load metrics
metrics_df = pd.read_csv("reports/metrics_summary.csv")

# Filter to test split only
test_df = metrics_df[metrics_df['split'] == 'test'].copy()

# ============================================================
# 1. Model Performance Comparison (Bar Chart)
# ============================================================
print("\n1. Creating model performance comparison...")

# Get top models
models_to_plot = ['XGBoost', 'XGB+Embeddings', 'Random Forest', 'MLP', 'Logistic Regression']
plot_df = test_df[test_df['model'].isin(models_to_plot)].copy()

# Get best performance per model
plot_df = plot_df.groupby('model').agg({
    'pr_auc': 'max',
    'roc_auc': 'max',
    'f1': 'max'
}).reset_index()

# Sort by PR-AUC
plot_df = plot_df.sort_values('pr_auc', ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# PR-AUC
ax = axes[0]
colors = ['#2ecc71' if m == 'XGBoost' else '#3498db' if m == 'XGB+Embeddings' else '#95a5a6' 
          for m in plot_df['model']]
ax.barh(plot_df['model'], plot_df['pr_auc'], color=colors)
ax.set_xlabel('PR-AUC', fontweight='bold')
ax.set_title('Precision-Recall AUC (Test)', fontweight='bold', fontsize=12)
ax.set_xlim(0, 0.8)
for i, (model, value) in enumerate(zip(plot_df['model'], plot_df['pr_auc'])):
    ax.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=9)

# ROC-AUC
ax = axes[1]
ax.barh(plot_df['model'], plot_df['roc_auc'], color=colors)
ax.set_xlabel('ROC-AUC', fontweight='bold')
ax.set_title('ROC AUC (Test)', fontweight='bold', fontsize=12)
ax.set_xlim(0, 1.0)
for i, (model, value) in enumerate(zip(plot_df['model'], plot_df['roc_auc'])):
    ax.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=9)

# F1
ax = axes[2]
ax.barh(plot_df['model'], plot_df['f1'], color=colors)
ax.set_xlabel('F1 Score', fontweight='bold')
ax.set_title('F1 Score (Test)', fontweight='bold', fontsize=12)
ax.set_xlim(0, 0.8)
for i, (model, value) in enumerate(zip(plot_df['model'], plot_df['f1'])):
    ax.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=9)

plt.suptitle('Model Performance Comparison: Fusion vs Baseline', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "model_comparison.png", bbox_inches='tight', dpi=300)
plt.close()
print(f"   ✅ Saved: {output_dir / 'model_comparison.png'}")

# ============================================================
# 2. Fusion vs Baseline Direct Comparison
# ============================================================
print("\n2. Creating fusion vs baseline direct comparison...")

baseline_xgb = plot_df[plot_df['model'] == 'XGBoost'].iloc[0]
fusion = plot_df[plot_df['model'] == 'XGB+Embeddings'].iloc[0]

metrics = ['pr_auc', 'roc_auc', 'f1']
baseline_vals = [baseline_xgb[m] for m in metrics]
fusion_vals = [fusion[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (XGBoost)', color='#2ecc71')
bars2 = ax.bar(x + width/2, fusion_vals, width, label='Fusion (XGBoost + Node2Vec)', color='#3498db')

ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('Baseline vs Fusion: Direct Comparison', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(['PR-AUC', 'ROC-AUC', 'F1 Score'], fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Add difference annotations
for i, metric in enumerate(metrics):
    diff = fusion_vals[i] - baseline_vals[i]
    y_pos = max(baseline_vals[i], fusion_vals[i]) + 0.05
    color = 'red' if diff < 0 else 'green'
    ax.text(i, y_pos, f'{diff:+.3f}', ha='center', color=color, 
            fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "fusion_vs_baseline.png", bbox_inches='tight', dpi=300)
plt.close()
print(f"   ✅ Saved: {output_dir / 'fusion_vs_baseline.png'}")

# ============================================================
# 3. Pipeline Overview Diagram
# ============================================================
print("\n3. Creating pipeline overview diagram...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Title
ax.text(5, 4.5, 'Graph-Tabular Fusion Pipeline', 
        ha='center', fontsize=16, fontweight='bold')

# Stage 1: Data
box1_x, box1_y = 1, 2.5
ax.add_patch(plt.Rectangle((box1_x-0.4, box1_y-0.3), 0.8, 0.6, 
                           fill=True, facecolor='#ecf0f1', edgecolor='#34495e', linewidth=2))
ax.text(box1_x, box1_y, 'Elliptic++\nDataset', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(box1_x, box1_y-0.6, '203K nodes\n234K edges', ha='center', va='top', fontsize=7)

# Arrow 1
ax.annotate('', xy=(2.5, box1_y), xytext=(1.5, box1_y),
            arrowprops=dict(arrowstyle='->', lw=2, color='#34495e'))

# Stage 2: Features
box2_x, box2_y = 3.2, 3.2
ax.add_patch(plt.Rectangle((box2_x-0.4, box2_y-0.3), 0.8, 0.6,
                           fill=True, facecolor='#e8f5e9', edgecolor='#27ae60', linewidth=2))
ax.text(box2_x, box2_y, 'Tabular\nFeatures', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(box2_x, box2_y-0.6, 'Local AF1-93', ha='center', va='top', fontsize=7)

# Stage 3: Graph
box3_x, box3_y = 3.2, 1.8
ax.add_patch(plt.Rectangle((box3_x-0.4, box3_y-0.3), 0.8, 0.6,
                           fill=True, facecolor='#e3f2fd', edgecolor='#2196f3', linewidth=2))
ax.text(box3_x, box3_y, 'Graph\nStructure', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(box3_x, box3_y-0.6, 'Edge List', ha='center', va='top', fontsize=7)

# Arrows from data
ax.annotate('', xy=(2.7, box2_y), xytext=(1.5, box1_y+0.1),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#27ae60'))
ax.annotate('', xy=(2.7, box3_y), xytext=(1.5, box1_y-0.1),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#2196f3'))

# Stage 4: Embeddings
box4_x, box4_y = 4.7, 1.8
ax.add_patch(plt.Rectangle((box4_x-0.5, box4_y-0.3), 1.0, 0.6,
                           fill=True, facecolor='#fff3e0', edgecolor='#ff9800', linewidth=2))
ax.text(box4_x, box4_y, 'Node2Vec\nEmbeddings', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(box4_x, box4_y-0.6, '64-dim\nLeakage-free', ha='center', va='top', fontsize=7)

# Arrow to embeddings
ax.annotate('', xy=(4.1, box4_y), xytext=(3.7, box3_y),
            arrowprops=dict(arrowstyle='->', lw=2, color='#ff9800'))

# Stage 5: Fusion
box5_x, box5_y = 6.5, 2.5
ax.add_patch(plt.Rectangle((box5_x-0.5, box5_y-0.4), 1.0, 0.8,
                           fill=True, facecolor='#f3e5f5', edgecolor='#9c27b0', linewidth=2))
ax.text(box5_x, box5_y, 'Feature\nConcatenation', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(box5_x, box5_y-0.7, '93 + 64 = 157', ha='center', va='top', fontsize=7)

# Arrows to fusion
ax.annotate('', xy=(5.9, box5_y+0.1), xytext=(3.7, box2_y),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#27ae60'))
ax.annotate('', xy=(5.9, box5_y-0.1), xytext=(5.3, box4_y),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#ff9800'))

# Stage 6: Model
box6_x, box6_y = 8.3, 2.5
ax.add_patch(plt.Rectangle((box6_x-0.5, box6_y-0.3), 1.0, 0.6,
                           fill=True, facecolor='#ffebee', edgecolor='#e53935', linewidth=2))
ax.text(box6_x, box6_y, 'XGBoost\nClassifier', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(box6_x, box6_y-0.6, 'Early stopping', ha='center', va='top', fontsize=7)

# Arrow to model
ax.annotate('', xy=(7.7, box6_y), xytext=(7.1, box5_y),
            arrowprops=dict(arrowstyle='->', lw=2, color='#e53935'))

# Result
ax.text(8.3, 1.2, 'PR-AUC: 0.656', ha='center', fontsize=11, 
        bbox=dict(boxstyle='round', facecolor='#fff9c4', edgecolor='#f57f17', linewidth=2))

plt.tight_layout()
plt.savefig(output_dir / "pipeline_diagram.png", bbox_inches='tight', dpi=300)
plt.close()
print(f"   ✅ Saved: {output_dir / 'pipeline_diagram.png'}")

# ============================================================
# 4. Feature Contribution Summary
# ============================================================
print("\n4. Creating feature contribution summary...")

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Tabular\nFeatures\n(93)', 'Graph\nEmbeddings\n(64)', 'Combined\nFusion\n(157)']
performance = [0.6689, 0.0, 0.6555]  # XGBoost baseline, N/A for embeddings-only, Fusion

colors = ['#2ecc71', '#95a5a6', '#3498db']
bars = ax.bar(categories, performance, color=colors, edgecolor='black', linewidth=1.5)

ax.set_ylabel('PR-AUC Score', fontweight='bold', fontsize=12)
ax.set_title('Feature Contribution Analysis', fontweight='bold', fontsize=14)
ax.set_ylim(0, 0.8)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, performance)):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., 0.05,
                'N/A', ha='center', va='bottom', fontsize=10, style='italic')

# Add annotations
ax.text(0, 0.71, 'Best', ha='center', fontsize=10, color='green', fontweight='bold')
ax.text(2, 0.67, '-2%', ha='center', fontsize=10, color='red', fontweight='bold')

# Add note
ax.text(0.5, -0.15, 'Note: Embeddings-only not tested (not comparable without node features)',
        transform=ax.transAxes, ha='center', fontsize=8, style='italic')

plt.tight_layout()
plt.savefig(output_dir / "feature_contribution.png", bbox_inches='tight', dpi=300)
plt.close()
print(f"   ✅ Saved: {output_dir / 'feature_contribution.png'}")

# ============================================================
# 5. Summary Statistics Table
# ============================================================
print("\n5. Creating summary statistics visualization...")

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

# Create summary data
summary_data = [
    ['Metric', 'Baseline\n(XGBoost)', 'Fusion\n(XGBoost+Node2Vec)', 'Difference'],
    ['PR-AUC', '0.6689', '0.6555', '-0.0134 (-2%)'],
    ['ROC-AUC', '0.8881', '0.8608', '-0.0273 (-3%)'],
    ['F1 Score', '0.6988', '0.6877', '-0.0111 (-2%)'],
    ['Features', '93 (Local)', '157 (93+64)', '+64 embeddings'],
    ['Training Time', '~2 min', '~2 min', 'Similar'],
]

table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, 6):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('white')
        
        # Highlight differences in red
        if j == 3 and i in [1, 2, 3]:
            table[(i, j)].set_text_props(color='red', weight='bold')

plt.suptitle('Performance Summary: Fusion vs Baseline', 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig(output_dir / "summary_table.png", bbox_inches='tight', dpi=300)
plt.close()
print(f"   ✅ Saved: {output_dir / 'summary_table.png'}")

print("\n" + "=" * 60)
print("✅ All visualizations created successfully!")
print("=" * 60)
print(f"\nGenerated files in {output_dir}:")
print("  1. model_comparison.png - Multi-metric comparison")
print("  2. fusion_vs_baseline.png - Direct comparison")
print("  3. pipeline_diagram.png - Architecture overview")
print("  4. feature_contribution.png - Feature analysis")
print("  5. summary_table.png - Results summary")
