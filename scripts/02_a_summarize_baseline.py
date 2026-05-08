import os
import glob
import pandas as pd

SUMMARY_DIR = "outputs/a_baseline"
OUTPUT_PATH = os.path.join(SUMMARY_DIR, "a_baseline_summary.csv")

def main():
    summary_files = glob.glob(os.path.join(SUMMARY_DIR, "summary_*.csv"))

    # 避免把最终汇总表自己又读进去
    summary_files = [
        f for f in summary_files
        if not os.path.basename(f).startswith("a_baseline_summary")
    ]

    if len(summary_files) == 0:
        raise RuntimeError(f"No summary files found in {SUMMARY_DIR}")

    dfs = []
    for file_path in summary_files:
        df = pd.read_csv(file_path)
        df["summary_file"] = file_path
        dfs.append(df)

    summary = pd.concat(dfs, ignore_index=True)

    cols = [
        "method",
        "feature_type",
        "num_frames",
        "avg_num_clusters",
        "min_num_clusters",
        "max_num_clusters",
        "std_num_clusters",
        "avg_noise_ratio",
        "avg_runtime_sec",
        "frames_with_less_than_2_clusters",
        "ratio_frames_at_least_2_clusters",
        "n_clusters_param",
        "eps",
        "min_samples",
        "result_path",
        "stats_path",
    ]

    existing_cols = [c for c in cols if c in summary.columns]
    summary = summary[existing_cols]

    method_order = {
        "kmeans": 0,
        "dbscan": 1,
        "agglomerative": 2,
        "gmm": 3,
    }

    summary["order"] = summary["method"].map(method_order)
    summary = summary.sort_values("order").drop(columns=["order"])

    summary.to_csv(OUTPUT_PATH, index=False)

    print("=" * 80)
    print("A-class baseline summary")
    print("=" * 80)
    print(summary)
    print()
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
