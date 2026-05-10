# =============================================================================
# ai_training/train.py
#
# Full training pipeline:
#   1. Load data
#   2. Engineer features
#   3. Create labels
#   4. Time-series train/test split (no shuffle)
#   5. Scale features
#   6. Train RandomForestClassifier
#   7. Evaluate and print metrics
#   8. Save model + scaler
#
# Public API
# ----------
#   train_model() → (model, scaler, report_dict)
# =============================================================================

import os, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from ai_training.data_loader import load_data
from ai_training.features   import create_features, get_feature_cols, fit_scaler, scale_features
from ai_training.labels     import create_labels


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_model(data_source: str = "auto", df: pd.DataFrame = None):
    """
    Run the full training pipeline.

    Parameters
    ----------
    data_source : str
        Passed to load_data() only when df is None.
    df : pd.DataFrame, optional
        Pre-loaded OHLCV data. When provided, skips load_data() so that
        training and backtesting always use the identical dataset.

    Returns
    -------
    model   : fitted RandomForestClassifier
    scaler  : fitted RobustScaler
    report  : dict with evaluation metrics
    df_test : test-set DataFrame (for analysis)
    """

    # 1. Load data (or reuse pre-loaded frame)
    print("\n=== [1/7] Loading data ===")
    if df is None:
        df = load_data(source=data_source)
    else:
        print(f"[Train] Using pre-loaded data: {len(df):,} rows.")

    # 2. Feature engineering
    print("\n=== [2/7] Engineering features ===")
    df = create_features(df)

    # 3. Labels
    print("\n=== [3/7] Creating labels ===")
    df = create_labels(df)

    # 4. Identify feature columns
    feature_cols = get_feature_cols(df)
    print(f"[Train] Using {len(feature_cols)} features.")

    X = df[feature_cols]
    y = df["target"]

    # 5. Time-series split (no shuffle — preserves temporal order)
    print("\n=== [4/7] Splitting data ===")
    split_idx = int(len(df) * (1 - config.TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    df_test = df.iloc[split_idx:].copy()

    print(f"[Train] Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"[Train] Train label balance: {y_train.mean():.3f} (BUY ratio)")

    # 6. Scale features (fit ONLY on train, transform both)
    print("\n=== [5/7] Scaling features ===")
    scaler = fit_scaler(X_train, config.SCALER_PATH)
    X_train_sc = scale_features(X_train, scaler)
    X_test_sc  = scale_features(X_test,  scaler)

    # 7. Train model
    if config.USE_HGB:
        print("\n=== [6/7] Training HistGradientBoostingClassifier ===")
        model = HistGradientBoostingClassifier(
            max_iter         = config.MAX_ITER,
            max_depth        = config.MAX_DEPTH,
            min_samples_leaf = config.MIN_SAMPLES_LEAF,
            learning_rate    = config.LEARNING_RATE,
            class_weight     = config.CLASS_WEIGHT,
            random_state     = config.RANDOM_STATE,
        )
    else:
        print("\n=== [6/7] Training RandomForestClassifier ===")
        model = RandomForestClassifier(
            n_estimators     = config.N_ESTIMATORS,
            max_depth        = config.MAX_DEPTH,
            min_samples_leaf = config.MIN_SAMPLES_LEAF,
            class_weight     = config.CLASS_WEIGHT,
            random_state     = config.RANDOM_STATE,
            n_jobs           = -1,
        )
    model.fit(X_train_sc, y_train)

    # 8. Evaluate
    print("\n=== [7/7] Evaluating ===")
    y_pred       = model.predict(X_test_sc)
    y_prob       = model.predict_proba(X_test_sc)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred, zero_division=0)
    rec       = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_prob)
    cm        = confusion_matrix(y_test, y_pred)

    report = {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "auc_roc":   auc,
        "confusion_matrix": cm.tolist(),
        "n_train":   len(X_train),
        "n_test":    len(X_test),
        "features":  feature_cols,
    }

    print(f"\n{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"{'─'*50}")
    print("\nConfusion Matrix:")
    print(f"  [[TN={cm[0,0]:5d}  FP={cm[0,1]:5d}]")
    print(f"   [FN={cm[1,0]:5d}  TP={cm[1,1]:5d}]]")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['SELL','BUY'])}")

    # Feature importance (RandomForest only; HGB exposes no direct importances)
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        top10 = importances.nlargest(10)
        print("Top-10 Feature Importances:")
        for feat, imp in top10.items():
            print(f"  {feat:<25s} {imp:.4f}")

    # Store predictions on test set for analysis
    df_test = df_test.copy()
    df_test["pred"]       = y_pred
    df_test["confidence"] = y_prob

    return model, scaler, report, df_test


# ---------------------------------------------------------------------------
# Save / Load model
# ---------------------------------------------------------------------------

def save_model(model, model_path: str = config.MODEL_PATH):
    """Persist trained model to disk."""
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[Train] Model saved → {model_path}")


def load_model(model_path: str = config.MODEL_PATH):
    """Load a previously saved model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}. Run train_model() first.")
    model = joblib.load(model_path)
    print(f"[Train] Model loaded ← {model_path}")
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model, scaler, report, df_test = train_model(data_source="auto")
    save_model(model)
    print("\n[Train] Done. Model and scaler saved.")
