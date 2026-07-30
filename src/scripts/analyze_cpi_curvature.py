"""Measure CPI/curvature associations in generated amortizer datasets.

Correlations are computed after averaging trials within each simulated parameter
set. This avoids treating repeated trials from the same virtual user as
independent observations.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from ..utils.myutils import load_config
from ..utils.trajectory_features import CURVATURE_FEATURE_NAMES


def _dataset_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("tr_*.npz"))


def _finite_correlation(x: np.ndarray, y: np.ndarray, method: str):
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 3:
        return np.nan, np.nan, int(np.count_nonzero(keep))
    if np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return np.nan, np.nan, int(np.count_nonzero(keep))
    if method == "pearson":
        result = pearsonr(x[keep], y[keep])
    else:
        result = spearmanr(x[keep], y[keep])
    return float(result.statistic), float(result.pvalue), int(np.count_nonzero(keep))


def analyze(files: list[Path]) -> list[dict[str, float | int | str]]:
    cpi_parts: list[np.ndarray] = []
    feature_parts = {name: [] for name in CURVATURE_FEATURE_NAMES}

    for data_path in files:
        metadata_path = data_path.with_suffix(".yaml")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing feature metadata: {metadata_path}")
        columns = list(load_config(metadata_path).stat)
        required = ["cpi", *CURVATURE_FEATURE_NAMES]
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(
                f"{data_path.name} predates curvature support; missing {missing}"
            )

        with np.load(data_path, allow_pickle=True) as dataset:
            stats = np.asarray(dataset["stat_data"], dtype=np.float64)
        if stats.ndim != 3:
            raise ValueError(
                f"Expected stat_data (parameter, trial, feature), got {stats.shape}"
            )

        # One row per virtual user / parameter vector.
        means = np.nanmean(stats, axis=1)
        cpi_parts.append(means[:, columns.index("cpi")])
        for name in CURVATURE_FEATURE_NAMES:
            feature_parts[name].append(means[:, columns.index(name)])

    cpi = np.concatenate(cpi_parts)
    rows: list[dict[str, float | int | str]] = []
    for name in CURVATURE_FEATURE_NAMES:
        values = np.concatenate(feature_parts[name])
        pearson_r, pearson_p, n = _finite_correlation(cpi, values, "pearson")
        spearman_rho, spearman_p, _ = _finite_correlation(cpi, values, "spearman")
        rows.append(
            {
                "feature": name,
                "n_parameter_sets": n,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["spearman_rho"])), reverse=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank full-resolution cursor curvature features by CPI correlation."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Training .npz file or its dataset directory",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    files = _dataset_files(args.dataset)
    if not files:
        raise SystemExit(f"No tr_*.npz dataset found under {args.dataset}")
    rows = analyze(files)
    output = args.output or (
        args.dataset.parent if args.dataset.is_file() else args.dataset
    ) / "curvature_cpi_correlation.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Analyzed {len(files)} dataset file(s).")
    print(f"Correlation report: {output}")
    for row in rows[:5]:
        print(
            f"{row['feature']}: Spearman rho={row['spearman_rho']:.4f}, "
            f"Pearson r={row['pearson_r']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
