import os
import pandas as pd

OUT_DIR = "outputs/d_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

A_PATH = "outputs/a_baseline/a_baseline_summary.csv"
B_PATH = "outputs/b_motion/b_motion_summary.csv"
C_PATH = "outputs/c_temporal/c_temporal_summary.csv"

OUT_PATH = os.path.join(OUT_DIR, "abc_method_summary.csv")


def load_a():
    df = pd.read_csv(A_PATH)
    df["stage"] = "A_baseline"
    df["method_name"] = df["method"] + "_" + df["feature_type"]
    df["avg_raw_dbscan_num_clusters"] = None
    df["avg_raw_dbscan_noise_ratio"] = None
    df["fallback_frames"] = 0
    df["fallback_ratio"] = 0.0
    return df


def load_b():
    df = pd.read_csv(B_PATH)
    df["stage"] = "B_motion"
    df["method_name"] = df["feature_mode"] + "_eps" + df["eps"].astype(str) + "_mw" + df["motion_weight"].astype(str)
    return df


def load_c():
    df = pd.read_csv(C_PATH)
    df["stage"] = "C_temporal"
    df["method_name"] = df["method"]

    # 为 relation_graph 加上阈值标识，方便区分不同参数
    mask = df["method"] == "relation_graph"
    df.loc[mask, "method_name"] = (
        df.loc[mask, "method"] +
        "_dt" + df.loc[mask, "dist_threshold"].astype(str) +
        "_vt" + df.loc[mask, "vel_threshold"].astype(str)
    )

    df["avg_noise_ratio"] = 0.0
    return df


def main():
    all_dfs = []

    if os.path.exists(A_PATH):
        all_dfs.append(load_a())
    else:
        print(f"[WARN] Missing {A_PATH}")

    if os.path.exists(B_PATH):
        all_dfs.append(load_b())
    else:
        print(f"[WARN] Missing {B_PATH}")

    if os.path.exists(C_PATH):
        all_dfs.append(load_c())
    else:
        print(f"[WARN] Missing {C_PATH}")

    if len(all_dfs) == 0:
        raise RuntimeError("No summary files found.")

    merged = pd.concat(all_dfs, ignore_index=True, sort=False)

    cols = [
        "stage",
        "method_name",
        "method",
        "feature_mode",
        "num_frames",
        "avg_num_clusters",
        "min_num_clusters",
        "max_num_clusters",
        "std_num_clusters",
        "avg_noise_ratio",
        "avg_raw_dbscan_num_clusters",
        "avg_raw_dbscan_noise_ratio",
        "fallback_frames",
        "fallback_ratio",
        "avg_runtime_sec",
        "frames_with_less_than_2_clusters",
        "ratio_frames_at_least_2_clusters",
        "eps",
        "min_samples",
        "window_size",
        "momentum_alpha",
        "position_weight",
        "motion_weight",
        "beta",
        "dist_threshold",
        "vel_threshold",
        "sigma_d",
        "sigma_v",
        "result_path",
        "stats_path",
    ]

    existing_cols = [c for c in cols if c in merged.columns]
    merged = merged[existing_cols]

    merged.to_csv(OUT_PATH, index=False)

    print("=" * 80)
    print("A/B/C method summary")
    print("=" * 80)
    print(merged)
    print()
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
