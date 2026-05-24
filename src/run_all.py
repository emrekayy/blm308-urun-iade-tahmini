"""Run the complete BLM308 project pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_models
from src.generate_presentation import build_presentation
from src.generate_report import build_report, export_pdf
from src.preprocess import preprocess_data
from src.train_models import train_models


def main() -> None:
    print("Step 1/5: Preprocessing and EDA...")
    preprocess_data()

    print("\nStep 2/5: Training models...")
    train_models()

    print("\nStep 3/5: Evaluating models...")
    evaluate_models()

    print("\nStep 4/5: Generating report...")
    docx_path = build_report()
    pdf_path = export_pdf(docx_path)
    print(f"Report DOCX: {docx_path}")
    if pdf_path:
        print(f"Report PDF: {pdf_path}")

    print("\nStep 5/5: Generating presentation...")
    pptx_path = build_presentation()
    print(f"Presentation: {pptx_path}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
