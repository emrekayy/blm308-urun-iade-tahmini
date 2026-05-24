"""Shared utilities for the BLM308 final project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RETURN_COLUMN_CANDIDATES = [
    "return_status",
    "returned",
    "is_returned",
    "return",
    "Return",
    "Returned",
    "Return_Status",
]

ID_COLUMNS = {"order_id", "orderid", "order", "transaction_id", "id"}
CATEGORY_COLUMNS = {"category", "product_category", "item_category", "category_name"}
PRICE_COLUMNS = {"price", "product_price", "item_price", "unit_price"}
DISCOUNT_COLUMNS = {"discount", "discount_rate", "discount_percent", "discount_amount"}
RATING_COLUMNS = {"rating", "customer_rating", "review.co_rating", "review_rating"}
DELIVERY_COLUMNS = {
    "delivery_time",
    "delivery_days",
    "shipping_time",
    "delivery_duration",
}


def find_csv_file() -> Path:
    """Locate the dataset CSV in the project directory."""
    search_dirs = [DATA_RAW_DIR, PROJECT_ROOT]
    for directory in search_dirs:
        if not directory.exists():
            continue
        csv_files = sorted(directory.glob("*.csv"))
        if csv_files:
            return csv_files[0]

    raise FileNotFoundError(
        "No CSV file found. Place the dataset under data/raw/ or the project root."
    )


def detect_target_column(df: pd.DataFrame) -> str:
    """Detect the most suitable binary return-related target column."""
    matches: list[tuple[str, float]] = []

    for column in df.columns:
        normalized = column.strip().lower()
        if normalized in {name.lower() for name in RETURN_COLUMN_CANDIDATES} or "return" in normalized:
            series = df[column]
            unique_count = series.nunique(dropna=True)
            if unique_count == 2:
                score = 10.0
                if normalized in {"return_status", "is_returned", "returned"}:
                    score += 5.0
                matches.append((column, score))
            elif unique_count <= 5:
                matches.append((column, 5.0))

    if not matches:
        raise ValueError(
            "Could not detect a return-related target column. "
            f"Checked columns: {list(df.columns)}"
        )

    matches.sort(key=lambda item: item[1], reverse=True)
    return matches[0][0]


def find_column(df: pd.DataFrame, candidates: set[str]) -> str | None:
    """Return the first matching column name using case-insensitive lookup."""
    lookup = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def inspect_dataset(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Collect dataset inspection metadata."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [
        col for col in df.columns if col not in numeric_cols and col != target_column
    ]

    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "target_column": target_column,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": df.isnull().sum().to_dict(),
        "class_distribution": df[target_column].value_counts(dropna=False).to_dict(),
    }


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)
