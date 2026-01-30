import argparse
import os
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

DATASET = "drtawfikrrahman/deep-learning-ev-battery-pack-diagnostics-sdg-7"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output directory for CSV")
    parser.add_argument("--file", default="", help="File path inside the Kaggle dataset")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not args.file:
        raise SystemExit(
            "Please provide --file. Use Kaggle UI to see the file name in the dataset."
        )

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET,
        args.file,
    )

    out_path = os.path.join(args.out, "dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()
