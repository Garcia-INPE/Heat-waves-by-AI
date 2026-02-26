"""Rank top individual versions from saved regression runs."""

from pathlib import Path
import re
import pandas as pd


DIR_INPUT = Path("output/3.0-regression/01-INFO")
DIR_RESULTS = Path("output/3.1-regression_top_versions")
TOP_N = 10


def parse_info_file(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")

    version_match = re.search(r"INFO-v(\d+)\.txt$", path.name)
    header_match = re.search(
        r"PRED=([^\n]+?) \+ SPL=([^\n]+?) \+ BAL=([^\n]+?) \+ SEM=([^\n]+)",
        text,
    )
    if not version_match or not header_match:
        return None

    pred, split, bal, sem = [x.strip() for x in header_match.groups()]

    model_rows = []
    for m in re.finditer(
        r"\b(Eta|RF|XGB|NN)\s*->\s*MAE:\s*([0-9.]+)\s*\|\s*RMSE:\s*([0-9.]+)\s*\|\s*CORR:\s*([-0-9.]+)",
        text,
    ):
        name, mae, rmse, corr = m.groups()
        model_rows.append((name, float(mae), float(rmse), float(corr)))

    if not model_rows:
        return None

    best_model = sorted(model_rows, key=lambda t: (t[2], t[1], -t[3]))[0]

    return {
        "version": int(version_match.group(1)),
        "split": split,
        "features": pred,
        "balance": bal,
        "sem_prev": sem,
        "best_model": best_model[0],
        "mae": best_model[1],
        "rmse": best_model[2],
        "corr": best_model[3],
    }


def main() -> None:
    rows = []
    for info_file in sorted(DIR_INPUT.glob("INFO-v*.txt")):
        row = parse_info_file(info_file)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No valid INFO-v*.txt files found in {DIR_INPUT}")

    df = pd.DataFrame(rows)

    # Rank by best RMSE
    by_rmse = df.sort_values(["rmse", "mae", "corr"],
                             ascending=[True, True, False])
    top_rmse = by_rmse.head(TOP_N)

    # Rank by best RMSE per split type
    by_rmse_70_30 = df[df['split'] == '70_30'].sort_values(
        ["rmse", "mae", "corr"], ascending=[True, True, False])
    top_rmse_70_30 = by_rmse_70_30.head(TOP_N)

    by_rmse_11_19 = df[df['split'] == '11_19'].sort_values(
        ["rmse", "mae", "corr"], ascending=[True, True, False])
    top_rmse_11_19 = by_rmse_11_19.head(TOP_N)

    # Rank by best CORR
    by_corr = df.sort_values(["corr", "rmse", "mae"],
                             ascending=[False, True, True])
    top_corr = by_corr.head(TOP_N)

    # Save rankings
    by_rmse.to_csv(DIR_RESULTS / "RANKING-Versions_By_RMSE.csv", index=False)
    by_rmse_70_30.to_csv(
        DIR_RESULTS / "RANKING-Versions_By_RMSE-Split_70_30.csv", index=False)
    by_rmse_11_19.to_csv(
        DIR_RESULTS / "RANKING-Versions_By_RMSE-Split_11_19.csv", index=False)
    by_corr.to_csv(DIR_RESULTS / "RANKING-Versions_By_CORR.csv", index=False)

    top_rmse.to_csv(DIR_RESULTS / "TOP-Versions_By_RMSE.csv", index=False)
    top_rmse_70_30.to_csv(
        DIR_RESULTS / "TOP-Versions_By_RMSE-Split_70_30.csv", index=False)
    top_rmse_11_19.to_csv(
        DIR_RESULTS / "TOP-Versions_By_RMSE-Split_11_19.csv", index=False)
    top_corr.to_csv(DIR_RESULTS / "TOP-Versions_By_CORR.csv", index=False)

    # Generate text reports
    with (DIR_RESULTS / "TOP-Versions_By_RMSE.txt").open("w", encoding="utf-8") as f:
        f.write(
            f"Top {TOP_N} Individual Versions (ranked by best RMSE, then MAE, then CORR)\n")
        f.write("=" * 150 + "\n")
        for i, row in top_rmse.reset_index(drop=True).iterrows():
            f.write(
                f"{i+1:>2}. v{int(row['version']):02d} | "
                f"MODEL={row['best_model']:<3} | "
                f"RMSE={row['rmse']:.4f} | MAE={row['mae']:.4f} | CORR={row['corr']:.4f} | "
                f"SPL={row['split']:<5} | PRED={row['features']:<12} | "
                f"BAL={row['balance']:<3} | SEM={row['sem_prev']}\n"
            )

    with (DIR_RESULTS / "TOP-Versions_By_RMSE-Split_70_30.txt").open("w", encoding="utf-8") as f:
        f.write(
            f"Top {TOP_N} Individual Versions (split=70_30, ranked by best RMSE, then MAE, then CORR)\n")
        f.write("=" * 150 + "\n")
        for i, row in top_rmse_70_30.reset_index(drop=True).iterrows():
            f.write(
                f"{i+1:>2}. v{int(row['version']):02d} | "
                f"MODEL={row['best_model']:<3} | "
                f"RMSE={row['rmse']:.4f} | MAE={row['mae']:.4f} | CORR={row['corr']:.4f} | "
                f"SPL={row['split']:<5} | PRED={row['features']:<12} | "
                f"BAL={row['balance']:<3} | SEM={row['sem_prev']}\n"
            )

    with (DIR_RESULTS / "TOP-Versions_By_RMSE-Split_11_19.txt").open("w", encoding="utf-8") as f:
        f.write(
            f"Top {TOP_N} Individual Versions (split=11_19, ranked by best RMSE, then MAE, then CORR)\n")
        f.write("=" * 150 + "\n")
        for i, row in top_rmse_11_19.reset_index(drop=True).iterrows():
            f.write(
                f"{i+1:>2}. v{int(row['version']):02d} | "
                f"MODEL={row['best_model']:<3} | "
                f"RMSE={row['rmse']:.4f} | MAE={row['mae']:.4f} | CORR={row['corr']:.4f} | "
                f"SPL={row['split']:<5} | PRED={row['features']:<12} | "
                f"BAL={row['balance']:<3} | SEM={row['sem_prev']}\n"
            )

    with (DIR_RESULTS / "TOP-Versions_By_CORR.txt").open("w", encoding="utf-8") as f:
        f.write(
            f"Top {TOP_N} Individual Versions (ranked by best CORR, then RMSE, then MAE)\n")
        f.write("=" * 150 + "\n")
        for i, row in top_corr.reset_index(drop=True).iterrows():
            f.write(
                f"{i+1:>2}. v{int(row['version']):02d} | "
                f"MODEL={row['best_model']:<3} | "
                f"CORR={row['corr']:.4f} | RMSE={row['rmse']:.4f} | MAE={row['mae']:.4f} | "
                f"SPL={row['split']:<5} | PRED={row['features']:<12} | "
                f"BAL={row['balance']:<3} | SEM={row['sem_prev']}\n"
            )

    print("=" * 100)
    print(f"TOP {TOP_N} VERSIONS BY RMSE (ALL SPLITS)")
    print("=" * 100)
    print(top_rmse[["version", "best_model", "rmse", "mae", "corr",
          "split", "features", "balance", "sem_prev"]].to_string(index=False))
    print("\n" + "=" * 100)
    print(f"TOP {TOP_N} VERSIONS BY RMSE (SPLIT=70_30)")
    print("=" * 100)
    print(top_rmse_70_30[["version", "best_model", "rmse", "mae", "corr",
          "split", "features", "balance", "sem_prev"]].to_string(index=False))
    print("\n" + "=" * 100)
    print(f"TOP {TOP_N} VERSIONS BY RMSE (SPLIT=11_19)")
    print("=" * 100)
    print(top_rmse_11_19[["version", "best_model", "rmse", "mae", "corr",
          "split", "features", "balance", "sem_prev"]].to_string(index=False))
    print("\n" + "=" * 100)
    print(f"TOP {TOP_N} VERSIONS BY CORR")
    print("=" * 100)
    print(top_corr[["version", "best_model", "corr", "rmse", "mae",
          "split", "features", "balance", "sem_prev"]].to_string(index=False))


if __name__ == "__main__":
    main()
