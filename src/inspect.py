import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print("Columns:")
    for c in df.columns:
        print(f"- {c} ({df[c].dtype})")
    print("\nHead:")
    print(df.head())


if __name__ == "__main__":
    main()
