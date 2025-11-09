"""XGBoost fusion trainer."""
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score
import json
from pathlib import Path


def train_xgb_fusion(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    config: dict,
    output_dir: str = "reports"
):
    """
    Train XGBoost on fusion features.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        config: XGBoost configuration dict
        output_dir: Where to save results
        
    Returns:
        model: Trained XGBoost model
        metrics: Dictionary of evaluation metrics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Compute class weights if needed
    scale_pos_weight = config.get('scale_pos_weight')
    if scale_pos_weight is None:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        print(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Training parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': config.get('eval_metric', 'aucpr'),
        'max_depth': config.get('max_depth', 6),
        'learning_rate': config.get('learning_rate', 0.05),
        'subsample': config.get('subsample', 0.8),
        'colsample_bytree': config.get('colsample_bytree', 0.8),
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',
        'device': 'cuda' if config.get('device') == 'cuda' else 'cpu'
    }
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    # Train
    print("Training XGBoost...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=config.get('n_estimators', 800),
        evals=evals,
        early_stopping_rounds=config.get('early_stopping_rounds', 50),
        verbose_eval=50
    )
    
    # Predict
    y_pred_val = model.predict(dval)
    y_pred_test = model.predict(dtest)
    
    # Evaluate
    metrics = {
        'val': evaluate_predictions(y_val, y_pred_val),
        'test': evaluate_predictions(y_test, y_pred_test)
    }
    
    # Save
    model.save_model(str(output_path / "xgb_fusion.json"))
    with open(output_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Model and metrics saved to {output_dir}")
    
    return model, metrics


def evaluate_predictions(y_true, y_pred):
    """Calculate evaluation metrics."""
    roc_auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    
    # Find best F1 threshold
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = [f1_score(y_true, y_pred >= t) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = thresholds[best_idx]
    
    # Recall@K
    top_k_1pct = int(len(y_true) * 0.01)
    top_k_indices = np.argsort(y_pred)[-top_k_1pct:]
    recall_1pct = recall_score(y_true, np.isin(np.arange(len(y_true)), top_k_indices))
    
    return {
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'best_f1': float(best_f1),
        'threshold': float(best_thresh),
        'recall@1%': float(recall_1pct)
    }


if __name__ == "__main__":
    print("XGBoost fusion trainer - use via notebooks or scripts")
