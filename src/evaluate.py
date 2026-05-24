"""Evaluate trained models with stratified cross-validation and test metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import encode_product_features  # noqa: E402
from src.train_models import get_models  # noqa: E402
from src.utils import DATA_PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR, save_json  # noqa: E402

RANDOM_STATE = 42
CV_FOLDS = 10


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(DATA_PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_PROCESSED_DIR / "y_train.csv").squeeze()
    X_test = pd.read_csv(DATA_PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED_DIR / "y_test.csv").squeeze()
    X_train_raw = pd.read_csv(DATA_PROCESSED_DIR / "X_train_raw.csv")
    X_test_raw = pd.read_csv(DATA_PROCESSED_DIR / "X_test_raw.csv")
    return X_train, y_train, X_test, y_test, X_train_raw, X_test_raw


def load_test_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(DATA_PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_PROCESSED_DIR / "y_train.csv").squeeze()
    X_test = pd.read_csv(DATA_PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, y_train, X_test, y_test


def run_cross_validation(X_train_raw: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Run CV with product encodings recomputed inside each fold to avoid target leakage."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fold_scores = {name: [] for name in get_models()}

    for name, model_template in get_models().items():
        scores = {metric: [] for metric in metric_names}
        for train_index, val_index in cv.split(X_train_raw, y_train):
            fold_train_raw = X_train_raw.iloc[train_index]
            fold_val_raw = X_train_raw.iloc[val_index]
            fold_y_train = y_train.iloc[train_index]
            fold_y_val = y_train.iloc[val_index]

            X_fold_train, X_fold_val, _ = encode_product_features(
                fold_train_raw,
                fold_val_raw,
                fold_y_train,
            )

            model = clone(model_template)
            model.fit(X_fold_train, fold_y_train)
            y_pred = model.predict(X_fold_val)
            y_proba = model.predict_proba(X_fold_val)[:, 1]

            scores["Accuracy"].append(accuracy_score(fold_y_val, y_pred))
            scores["Precision"].append(precision_score(fold_y_val, y_pred, zero_division=0))
            scores["Recall"].append(recall_score(fold_y_val, y_pred, zero_division=0))
            scores["F1-Score"].append(f1_score(fold_y_val, y_pred, zero_division=0))
            scores["ROC-AUC"].append(roc_auc_score(fold_y_val, y_proba))

        fold_scores[name] = {metric: float(np.mean(values)) for metric, values in scores.items()}
        print(
            f"{name}: "
            f"Acc={fold_scores[name]['Accuracy']:.4f}, "
            f"F1={fold_scores[name]['F1-Score']:.4f}, "
            f"AUC={fold_scores[name]['ROC-AUC']:.4f}"
        )

    rows = [{"Model": name, **fold_scores[name]} for name in get_models()]
    comparison = pd.DataFrame(rows)
    return comparison.sort_values("F1-Score", ascending=False).reset_index(drop=True)


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Returned", "Returned"],
        yticklabels=["Not Returned", "Returned"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    output_path = FIGURES_DIR / "08_confusion_matrix_best_model.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_roc_curve(y_true, y_proba, model_name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.title("ROC Curve - Best Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    output_path = FIGURES_DIR / "09_roc_curve_best_model.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_feature_importance(model, feature_names: list[str]) -> Path | None:
    if not hasattr(model, "feature_importances_"):
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=[importances[i] for i in indices],
        y=[feature_names[i] for i in indices],
        orient="h",
    )
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    output_path = FIGURES_DIR / "10_feature_importance_random_forest.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def evaluate_models() -> dict:
    X_train, y_train, X_test, y_test, X_train_raw, _ = load_data()

    print("=" * 60)
    print(f"{CV_FOLDS}-FOLD STRATIFIED CROSS VALIDATION (TRAINING SET ONLY)")
    print("Product encodings recomputed inside each CV fold.")
    print(f"Training samples: {len(X_train)} | Held-out test samples: {len(X_test)}")
    print("=" * 60)
    comparison = run_cross_validation(X_train_raw, y_train)

    best_model_name = comparison.iloc[0]["Model"]
    best_model = get_models()[best_model_name]
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "model": best_model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    cm_path = plot_confusion_matrix(y_test, y_pred, best_model_name)
    roc_path = plot_roc_curve(y_test, y_proba, best_model_name)

    rf_model = get_models()["Random Forest"]
    rf_model.fit(X_train, y_train)
    fi_path = plot_feature_importance(rf_model, list(X_train.columns))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    results = {
        "cv_folds": CV_FOLDS,
        "cv_data": "training_set_only",
        "cv_encoding": "per_fold_product_features",
        "cv_samples": len(X_train),
        "test_samples": len(X_test),
        "model_selection_metric": "F1-Score",
        "feature_columns": list(X_train.columns),
        "comparison_table": comparison.to_dict(orient="records"),
        "best_model": best_model_name,
        "test_metrics": test_metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "figures": {
            "confusion_matrix": str(cm_path.relative_to(PROJECT_ROOT)),
            "roc_curve": str(roc_path.relative_to(PROJECT_ROOT)),
            "feature_importance": str(fi_path.relative_to(PROJECT_ROOT)) if fi_path else None,
        },
    }
    save_json(results, DATA_PROCESSED_DIR / "evaluation_results.json")
    models_dir = DATA_PROCESSED_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / "best_model.joblib")

    print("\nBest model:", best_model_name)
    print("Test metrics:", test_metrics)
    return results


def main() -> None:
    evaluate_models()


if __name__ == "__main__":
    main()
