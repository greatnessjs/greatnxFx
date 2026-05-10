# =============================================================================
# ai_training/model.py
#
# Real-time prediction interface.
#
# Public API
# ----------
#   predict_signal(latest_df, model, scaler, feature_cols)
#       → {"signal": "BUY"|"SELL", "confidence": float, "raw_prob": float}
#
#   load_inference_bundle() → (model, scaler, feature_cols)
# =============================================================================

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from ai_training.train    import load_model
from ai_training.features import load_scaler, create_features, get_feature_cols


# ---------------------------------------------------------------------------
# Load everything needed for inference
# ---------------------------------------------------------------------------

def load_inference_bundle(
    model_path:  str = config.MODEL_PATH,
    scaler_path: str = config.SCALER_PATH,
):
    """
    Load the trained model and scaler from disk.

    Returns
    -------
    model        : fitted RandomForestClassifier
    scaler       : fitted RobustScaler
    feature_cols : list[str]  (preserved from training)
    """
    model  = load_model(model_path)
    scaler = load_scaler(scaler_path)

    # Feature column list is stored inside the model object for convenience
    feature_cols = getattr(model, "_feature_cols", None)
    if feature_cols is None:
        # Fallback: derive from a dummy featurised frame
        import warnings
        warnings.warn(
            "Model does not carry _feature_cols attribute. "
            "Re-train with the current pipeline to store feature names.",
            UserWarning,
        )
        feature_cols = []

    return model, scaler, feature_cols


# ---------------------------------------------------------------------------
# Predict on a single candle (latest bar)
# ---------------------------------------------------------------------------

def predict_signal(
    latest_df:    pd.DataFrame,
    model,
    scaler,
    feature_cols: list,
) -> dict:
    """
    Generate a trading signal for the most recent bar.

    Parameters
    ----------
    latest_df    : pd.DataFrame
        Recent OHLCV history (needs enough rows for indicators, e.g. 250+).
        Must NOT include future data.
    model        : fitted sklearn classifier
    scaler       : fitted sklearn scaler
    feature_cols : list[str]

    Returns
    -------
    dict
        {
            "signal":     "BUY" | "SELL",
            "confidence": float,   # probability of the predicted class
            "buy_prob":   float,   # raw P(BUY)
            "sell_prob":  float,   # raw P(SELL)
            "filtered":   bool,    # True if confidence > threshold
        }
    """
    # Re-compute features on the full window, then take the last row
    df_feat = create_features(latest_df.copy())

    if df_feat.empty or len(df_feat) == 0:
        return {
            "signal": "HOLD", "confidence": 0.0,
            "buy_prob": 0.5,  "sell_prob": 0.5, "filtered": False,
        }

    # If feature_cols list is empty (fallback), derive them
    if not feature_cols:
        feature_cols = get_feature_cols(df_feat)

    # Take only the last row for inference
    x = df_feat[feature_cols].iloc[[-1]]   # shape (1, n_features)
    x_sc = scaler.transform(x)

    probs     = model.predict_proba(x_sc)[0]   # [P(SELL), P(BUY)]
    buy_prob  = float(probs[1])
    sell_prob = float(probs[0])
    signal    = "BUY" if buy_prob >= 0.5 else "SELL"
    confidence = max(buy_prob, sell_prob)

    return {
        "signal":     signal,
        "confidence": confidence,
        "buy_prob":   buy_prob,
        "sell_prob":  sell_prob,
        "filtered":   confidence >= config.CONFIDENCE_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Batch prediction (for backtesting)
# ---------------------------------------------------------------------------

def predict_batch(
    df_feat:      pd.DataFrame,
    model,
    scaler,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Generate predictions for every row in a featurised DataFrame.

    Returns
    -------
    pd.DataFrame with columns: pred (int), buy_prob, sell_prob, confidence, filtered
    """
    if not feature_cols:
        feature_cols = get_feature_cols(df_feat)

    X     = df_feat[feature_cols]
    X_sc  = scaler.transform(X)
    probs = model.predict_proba(X_sc)          # (n, 2)

    results = pd.DataFrame({
        "pred":       probs[:, 1].round().astype(int),
        "buy_prob":   probs[:, 1],
        "sell_prob":  probs[:, 0],
        "confidence": np.max(probs, axis=1),
    }, index=df_feat.index)

    results["filtered"] = results["confidence"] >= config.CONFIDENCE_THRESHOLD
    return results


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from ai_training.data_loader import load_data
    from ai_training.train       import train_model, save_model

    print("Training a quick model for smoke-test …")
    model, scaler, report, _ = train_model(data_source="synthetic")

    # Attach feature_cols to model object
    model._feature_cols = report["features"]
    save_model(model)

    # Now test real-time prediction
    df = load_data(source="synthetic")
    window = df.tail(300).copy()          # last 300 bars

    result = predict_signal(window, model, scaler, report["features"])
    print(f"\nSignal: {result['signal']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Buy Prob  : {result['buy_prob']:.2%}")
    print(f"Filtered  : {result['filtered']}")
