"""Train classification models for product return prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import DATA_PROCESSED_DIR, save_json

RANDOM_STATE = 42


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(DATA_PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_PROCESSED_DIR / "y_train.csv").squeeze()
    return X_train, y_train


def get_models() -> dict:
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


def train_models() -> dict:
    X_train, y_train = load_training_data()
    models_dir = DATA_PROCESSED_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    trained: dict = {"models": [], "feature_names": list(X_train.columns)}

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        model_path = models_dir / f"{name.replace(' ', '_').lower()}.joblib"
        joblib.dump(model, model_path)
        trained["models"].append(
            {
                "name": name,
                "path": str(model_path.relative_to(PROJECT_ROOT)),
            }
        )
        print(f"Trained and saved: {name}")

    save_json(trained, DATA_PROCESSED_DIR / "trained_models.json")
    return trained


def main() -> None:
    train_models()


if __name__ == "__main__":
    main()
