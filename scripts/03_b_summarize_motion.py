import os
import glob
import pandas as pd

SUMMARY_DIR = "outputs/b_motion"
OUTPUT_PATH = os.path.join(SUMMARY_DIR, "b_motion_summary.csv")

def main():
    summary_files = glob.glob(os.path.join(SUMMARY_DIR, "summary_*.csv"))

    if len(summary_files) == 0:
        raise RuntimeError("No B-class summary files found.")

    dfs = []
    for f in summary_files:
        df = pd.read_csv(f)
        df["summary_file"] = f
        dfs.append(df)

    summary = pd.concat(dfs, ignore_index=True)

    cols = [
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
        "result_path",
        "stats_path",
    ]

    existing_cols = [c for c in cols if c in summary.columns]
    summary = summary[existing_cols]

    order = {
        "instant_velocity": 0,
        "window_velocity": 1,
        "momentum": 2,
        "direction": 3,
    }
    summary["order"] = summary["feature_mode"].map(order)
    summary = summary.sort_values("order").drop(columns=["order"])

    summary.to_csv(OUTPUT_PATH, index=False)

    print("=" * 80)
    print("B-class motion feature summary")
    print("=" * 80)
    print(summary)
    print()
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
