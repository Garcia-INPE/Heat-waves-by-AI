"""Feature selection by correlation with Tx_Obs.

Rules:
- Load 1.1-DATA_ORIGINAL.csv (fallback to 1.1-DATA_ORIGINAL-FULL.csv)
- Exclude columns starting with data_*
- Exclude integer features
- Rank remaining features by absolute Pearson correlation to Tx_Obs
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_DIR = "output/1.0-analise_dos_dados/"
PRIMARY_DATASET = f"{IN_DIR}/1.1-DATA_ORIGINAL.csv"
FALLBACK_DATASET = f"{IN_DIR}/1.1-DATA_ORIGINAL-FULL.csv"
TARGET = "Tx_Obs"
OUT_DIR = "output/2.0-feature_selection_by_corr"
OUT_FILE_ALL = "2.1-correlation_ranking_all.csv"
OUT_FILE_FILTERED = "2.2-correlation_ranking_filtered.csv"
OUT_FILE_PLOT = "2.3-correlation_top_features.png"

# Optional filter for what is considered "more correlated"
MIN_ABS_CORR = 0.20
TOP_N_PLOT = 15


def resolve_dataset_path() -> str:
    if os.path.exists(PRIMARY_DATASET):
        return PRIMARY_DATASET
    if os.path.exists(FALLBACK_DATASET):
        return FALLBACK_DATASET
    raise FileNotFoundError(
        f"Neither '{PRIMARY_DATASET}' nor '{FALLBACK_DATASET}' was found in the current directory."
    )


def select_candidate_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    data_cols = [c for c in df.columns if c.startswith("data_")]
    drop_cols = set(data_cols + [TARGET])

    predictors = [c for c in df.columns if c not in drop_cols]

    # Keep only numeric columns that are NOT integer dtype
    numeric_non_int_predictors = []
    int_predictors = []
    for col in predictors:
        if pd.api.types.is_numeric_dtype(df[col]):
            if pd.api.types.is_integer_dtype(df[col]):
                int_predictors.append(col)
            else:
                numeric_non_int_predictors.append(col)

    x = df[numeric_non_int_predictors].copy()
    return x, data_cols, int_predictors


def compute_corr_table(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    records = []
    for col in x.columns:
        corr_pearson = y.corr(x[col], method="pearson")
        corr_spearman = y.corr(x[col], method="spearman")
        records.append(
            {
                "Feature": col,
                "Pearson_Corr_Tx_Obs": corr_pearson,
                "Abs_Pearson_Corr_Tx_Obs": np.abs(corr_pearson) if pd.notna(corr_pearson) else np.nan,
                "Spearman_Corr_Tx_Obs": corr_spearman,
                "Abs_Spearman_Corr_Tx_Obs": np.abs(corr_spearman) if pd.notna(corr_spearman) else np.nan,
            }
        )

    corr_df = pd.DataFrame(records)
    corr_df["Abs_Best_Corr"] = corr_df[
        ["Abs_Pearson_Corr_Tx_Obs", "Abs_Spearman_Corr_Tx_Obs"]
    ].max(axis=1)
    corr_df = corr_df.sort_values(
        "Abs_Best_Corr", ascending=False).reset_index(drop=True)
    return corr_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature selection by correlation with Tx_Obs")
    parser.add_argument(
        "--min-abs-corr",
        type=float,
        default=MIN_ABS_CORR,
        help="Minimum absolute correlation threshold for filtered output (default: 0.20)",
    )
    parser.add_argument(
        "--top-n-plot",
        type=int,
        default=TOP_N_PLOT,
        help="Number of top features to include in correlation plot (default: 15)",
    )
    return parser.parse_args()


def save_correlation_plot(corr_df: pd.DataFrame, out_path: str, top_n: int) -> None:
    top_n = max(1, min(top_n, len(corr_df)))
    top_df = corr_df.head(top_n).copy()

    # Reverse order to show highest at top in horizontal bars.
    top_df = top_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * top_n + 1)))
    y = np.arange(len(top_df))
    bar_h = 0.38

    ax.barh(
        y - bar_h / 2,
        top_df["Pearson_Corr_Tx_Obs"],
        height=bar_h,
        label="Pearson",
    )
    ax.barh(
        y + bar_h / 2,
        top_df["Spearman_Corr_Tx_Obs"],
        height=bar_h,
        label="Spearman",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(top_df["Feature"])
    ax.set_xlabel("Correlation with Tx_Obs")
    ax.set_title(f"Top {top_n} Features by Correlation to Tx_Obs")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.grid(axis="x", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    min_abs_corr = float(args.min_abs_corr)
    top_n_plot = int(args.top_n_plot)

    os.makedirs(OUT_DIR, exist_ok=True)

    dataset_path = resolve_dataset_path()
    df = pd.read_csv(dataset_path)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not found in dataset: {dataset_path}")

    x, data_cols, int_predictors = select_candidate_features(df)
    y = df[TARGET]

    if x.shape[1] == 0:
        raise ValueError(
            "No non-integer numeric features remain after excluding data_* and integer columns."
        )

    corr_df = compute_corr_table(x, y)
    filtered_df = corr_df[corr_df["Abs_Best_Corr"] >= min_abs_corr].copy()

    out_all = os.path.join(OUT_DIR, OUT_FILE_ALL)
    out_filtered = os.path.join(OUT_DIR, OUT_FILE_FILTERED)
    out_plot = os.path.join(OUT_DIR, OUT_FILE_PLOT)

    corr_df.to_csv(out_all, index=False)
    filtered_df.to_csv(out_filtered, index=False)
    save_correlation_plot(corr_df, out_plot, top_n=top_n_plot)

    print("=" * 88)
    print("FEATURE SELECTION BY CORRELATION TO Tx_Obs")
    print("=" * 88)
    print(f"Dataset used: {dataset_path}")
    if dataset_path != PRIMARY_DATASET:
        print(f"Note: '{PRIMARY_DATASET}' not found, fallback used.")
    print(f"Rows: {len(df)}")
    print(f"Dropped data_* columns: {data_cols}")
    print(f"Dropped integer predictors: {int_predictors}")
    print(f"Predictors analyzed (non-integer numeric): {x.shape[1]}")

    print("\nTop correlated features (best of absolute Pearson/Spearman):")
    print(corr_df.head(20).to_string(index=False))

    print(
        f"\nFiltered features with max(|Pearson|, |Spearman|) >= {min_abs_corr:.2f}: {len(filtered_df)}")
    print(filtered_df.to_string(index=False))

    print("\nSaved files:")
    print(f"  - {out_all}")
    print(f"  - {out_filtered}")
    print(f"  - {out_plot}")


if __name__ == "__main__":
    main()
