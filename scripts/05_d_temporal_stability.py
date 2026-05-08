import os
import pandas as pd

SUMMARY_PATH = "outputs/d_analysis/abc_method_summary.csv"
OUT_PATH = "outputs/d_analysis/temporal_stability_summary.csv"


def compute_stability(stats_path):
    df = pd.read_csv(stats_path)
    df = df.sort_values("time_step").reset_index(drop=True)

    cluster_counts = df["num_clusters"].values
    if len(cluster_counts) <= 1:
        return {
            "mean_cluster_count_change": 0.0,
            "max_cluster_count_change": 0.0,
        }

    diffs = abs(pd.Series(cluster_counts).diff().dropna())

    return {
        "mean_cluster_count_change": float(diffs.mean()),
        "max_cluster_count_change": float(diffs.max()),
    }


def main():
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(SUMMARY_PATH)

    summary = pd.read_csv(SUMMARY_PATH)

    records = []

    for _, row in summary.iterrows():
        stats_path = row.get("stats_path", None)

        if not isinstance(stats_path, str) or not os.path.exists(stats_path):
            print(f"[WARN] Missing stats file: {stats_path}")
            continue

        stability = compute_stability(stats_path)

        record = {
            "stage": row.get("stage", ""),
            "method_name": row.get("method_name", ""),
            "avg_num_clusters": row.get("avg_num_clusters", None),
            "std_num_clusters": row.get("std_num_clusters", None),
            "avg_noise_ratio": row.get("avg_noise_ratio", None),
            "avg_runtime_sec": row.get("avg_runtime_sec", None),
            "ratio_frames_at_least_2_clusters": row.get("ratio_frames_at_least_2_clusters", None),
            "stats_path": stats_path,
        }

        record.update(stability)
        records.append(record)

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_PATH, index=False)

    print("=" * 80)
    print("Temporal stability summary")
    print("=" * 80)
    print(out_df)
    print()
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
