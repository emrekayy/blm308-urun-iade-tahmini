"""Data inspection, preprocessing, and exploratory data analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (  # noqa: E402
    CATEGORY_COLUMNS,
    DATA_PROCESSED_DIR,
    DELIVERY_COLUMNS,
    DISCOUNT_COLUMNS,
    FIGURES_DIR,
    ID_COLUMNS,
    PRICE_COLUMNS,
    RATING_COLUMNS,
    detect_target_column,
    find_column,
    find_csv_file,
    inspect_dataset,
    save_json,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
PRODUCT_ID_COLUMN = "product_id"


def print_inspection_summary(summary: dict) -> None:
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"Shape: {summary['shape'][0]} rows x {summary['shape'][1]} columns")
    print(f"Columns: {summary['columns']}")
    print(f"Target column: {summary['target_column']}")
    print(f"Numerical columns: {summary['numeric_columns']}")
    print(f"Categorical columns: {summary['categorical_columns']}")
    print("\nMissing values:")
    for column, count in summary["missing_values"].items():
        print(f"  {column}: {count}")
    print("\nClass distribution:")
    for label, count in summary["class_distribution"].items():
        print(f"  {label}: {count}")


def encode_product_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Create product frequency and train-only return-rate features without target leakage."""
    if PRODUCT_ID_COLUMN not in X_train.columns:
        raise ValueError(f"Expected column '{PRODUCT_ID_COLUMN}' in feature dataframe.")

    frequency_map = X_train[PRODUCT_ID_COLUMN].value_counts().to_dict()
    train_with_target = X_train[[PRODUCT_ID_COLUMN]].copy()
    train_with_target["return_status"] = y_train.values
    return_rate_map = (
        train_with_target.groupby(PRODUCT_ID_COLUMN)["return_status"].mean().to_dict()
    )
    global_return_rate = float(y_train.mean())

    def transform_features(X: pd.DataFrame) -> pd.DataFrame:
        encoded = X.copy()
        encoded["product_id_frequency"] = (
            encoded[PRODUCT_ID_COLUMN].map(frequency_map).fillna(0).astype(float)
        )
        encoded["product_return_rate"] = (
            encoded[PRODUCT_ID_COLUMN].map(return_rate_map).fillna(global_return_rate).astype(float)
        )
        return encoded.drop(columns=[PRODUCT_ID_COLUMN])

    X_train_encoded = transform_features(X_train)
    X_test_encoded = transform_features(X_test)

    feature_columns = list(X_train_encoded.columns)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_encoded),
        columns=feature_columns,
        index=X_train_encoded.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_encoded),
        columns=feature_columns,
        index=X_test_encoded.index,
    )

    unseen_in_test = int((X_test[PRODUCT_ID_COLUMN].map(frequency_map).isna()).sum())

    metadata = {
        "feature_columns": feature_columns,
        "dropped_columns": ["order_id", PRODUCT_ID_COLUMN],
        "product_encoding": {
            "method": "frequency_and_train_return_rate",
            "product_id_frequency": "Count of each product_id in the training set",
            "product_return_rate": "Mean return_status per product_id computed on training set only",
            "unseen_product_return_rate_fallback": global_return_rate,
            "unseen_product_frequency_fallback": 0,
            "unseen_products_in_test": unseen_in_test,
            "unique_products_in_train": len(frequency_map),
        },
        "numeric_columns_scaled": feature_columns,
        "scaler_fit_on": "training_set_only",
    }
    return X_train_scaled, X_test_scaled, metadata


def run_eda(df: pd.DataFrame, target_column: str) -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="Set2")
    skipped_charts: list[str] = []

    plt.figure(figsize=(7, 5))
    counts = df[target_column].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.title("Return Status Class Distribution")
    plt.xlabel("Return Status")
    plt.ylabel("Count")
    for index, value in enumerate(counts.values):
        plt.text(index, value + 5, str(value), ha="center")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_class_distribution.png", dpi=150)
    plt.close()

    category_col = find_column(df, CATEGORY_COLUMNS)
    if category_col:
        return_rates = df.groupby(category_col)[target_column].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=return_rates.index.astype(str), y=return_rates.values)
        plt.title("Return Rate by Product Category")
        plt.xlabel("Category")
        plt.ylabel("Return Rate")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "02_return_rate_by_category.png", dpi=150)
        plt.close()
    else:
        skipped_charts.append("Return rate by category (no category column found)")

    price_col = find_column(df, PRICE_COLUMNS)
    if price_col:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=target_column, y=price_col)
        plt.title("Price vs Return Status")
        plt.xlabel("Return Status")
        plt.ylabel("Price")
        plt.xticks([0, 1], ["Not Returned", "Returned"])
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "03_price_vs_return.png", dpi=150)
        plt.close()
    else:
        skipped_charts.append("Price vs return (no price column found)")

    discount_col = find_column(df, DISCOUNT_COLUMNS)
    if discount_col:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=target_column, y=discount_col)
        plt.title("Discount vs Return Status")
        plt.xlabel("Return Status")
        plt.ylabel("Discount")
        plt.xticks([0, 1], ["Not Returned", "Returned"])
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "04_discount_vs_return.png", dpi=150)
        plt.close()
    else:
        skipped_charts.append("Discount vs return (no discount column found)")

    rating_col = find_column(df, RATING_COLUMNS)
    if rating_col:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[rating_col], bins=5, kde=True)
        plt.title("Customer Rating Distribution")
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "05_rating_distribution.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=target_column, y=rating_col)
        plt.title("Rating vs Return Status")
        plt.xlabel("Return Status")
        plt.ylabel("Rating")
        plt.xticks([0, 1], ["Not Returned", "Returned"])
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "05b_rating_vs_return.png", dpi=150)
        plt.close()
    else:
        skipped_charts.append("Rating distribution (no rating column found)")

    delivery_col = find_column(df, DELIVERY_COLUMNS)
    if delivery_col:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=target_column, y=delivery_col)
        plt.title("Delivery Time vs Return Status")
        plt.xlabel("Return Status")
        plt.ylabel("Delivery Time")
        plt.xticks([0, 1], ["Not Returned", "Returned"])
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "06_delivery_time_vs_return.png", dpi=150)
        plt.close()
    else:
        skipped_charts.append("Delivery time vs return (no delivery time column found)")

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap (Numerical Features)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "07_correlation_heatmap.png", dpi=150)
        plt.close()

    return {
        "skipped_charts": skipped_charts,
        "generated_charts": sorted(p.name for p in FIGURES_DIR.glob("*.png")),
    }


def preprocess_data() -> dict:
    csv_path = find_csv_file()
    print(f"Using dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    target_column = detect_target_column(df)
    summary = inspect_dataset(df, target_column)
    print_inspection_summary(summary)

    for column in df.columns:
        if df[column].isnull().any():
            if df[column].dtype in ["float64", "int64"]:
                df[column] = df[column].fillna(df[column].median())
            else:
                df[column] = df[column].fillna(df[column].mode().iloc[0])

    eda_info = run_eda(df, target_column)

    id_columns = [col for col in df.columns if col.lower() in ID_COLUMNS]
    model_df = df.drop(columns=id_columns)
    y = model_df[target_column]
    X_raw = model_df.drop(columns=[target_column])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_test, processing_metadata = encode_product_features(X_train_raw, X_test_raw, y_train)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = DATA_PROCESSED_DIR / "cleaned_dataset.csv"
    pd.concat([pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1)]).to_csv(
        processed_path,
        index=False,
    )

    X_train.to_csv(DATA_PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(DATA_PROCESSED_DIR / "X_test.csv", index=False)
    X_train_raw.to_csv(DATA_PROCESSED_DIR / "X_train_raw.csv", index=False)
    X_test_raw.to_csv(DATA_PROCESSED_DIR / "X_test_raw.csv", index=False)
    y_train.to_csv(DATA_PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(DATA_PROCESSED_DIR / "y_test.csv", index=False)

    output = {
        "dataset_file": str(csv_path.name),
        "inspection": summary,
        "eda": eda_info,
        "processing": processing_metadata,
        "split": {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "random_state": RANDOM_STATE,
            "split_before_target_encoding": True,
        },
    }
    save_json(output, DATA_PROCESSED_DIR / "preprocessing_metadata.json")
    print(f"\nFeature columns: {processing_metadata['feature_columns']}")
    print(f"Unseen products in test: {processing_metadata['product_encoding']['unseen_products_in_test']}")
    print(f"Cleaned dataset saved to: {processed_path}")
    return output


def main() -> None:
    preprocess_data()


if __name__ == "__main__":
    main()
