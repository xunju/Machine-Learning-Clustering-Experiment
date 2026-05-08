import os
import glob
import pandas as pd

SUMMARY_DIR = "outputs/c_temporal"
OUTPUT_PATH = os.path.join(SUMMARY_DIR, "c_temporal_summary.csv")

def main():
    summary_files = glob.glob(os.path.join(SUMMARY_DIR, "summary_*.csv"))

    if len(summary_files) == 0:
        raise RuntimeError("No C-class summary files found.")

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
        "avg_raw_num_clusters",
        "fallback_frames",
        "fallback_ratio",
        "avg_runtime_sec",
        "frames_with_less_than_2_clusters",
        "ratio_frames_at_least_2_clusters",
        "window_size",
        "beta",
        "n_clusters_param",
        "dist_threshold",
        "vel_threshold",
        "sigma_d",
        "sigma_v",
        "result_path",
        "stats_path",
    ]

    existing_cols = [c for c in cols if c in summary.columns]
    summary = summary[existing_cols]

    order = {
        "temporal_agglomerative": 0,
        "relation_graph": 1,
        "affinity_spectral": 2,
    }
    summary["order"] = summary["method"].map(order)
    summary = summary.sort_values("order").drop(columns=["order"])

    summary.to_csv(OUTPUT_PATH, index=False)

    print("=" * 80)
    print("C-class temporal relation summary")
    print("=" * 80)
    print(summary)
    print()
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
