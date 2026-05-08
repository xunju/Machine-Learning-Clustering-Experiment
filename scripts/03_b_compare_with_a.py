import os
import pandas as pd

A_PATH = "outputs/a_baseline/a_baseline_summary.csv"
B_PATH = "outputs/b_motion/b_motion_summary.csv"
OUT_PATH = "outputs/b_motion/ab_comparison_summary.csv"

def main():
    if not os.path.exists(A_PATH):
        raise FileNotFoundError(A_PATH)
    if not os.path.exists(B_PATH):
        raise FileNotFoundError(B_PATH)

    a = pd.read_csv(A_PATH)
    b = pd.read_csv(B_PATH)

    # 只保留关键指标，方便横向比较
    a_view = a.copy()
    a_view["stage"] = "A_baseline"
    a_view["name"] = a_view["method"] + "_" + a_view["feature_type"]

    b_view = b.copy()
    b_view["stage"] = "B_motion"
    b_view["name"] = b_view["feature_mode"]

    common_cols = [
        "stage",
        "name",
        "num_frames",
        "avg_num_clusters",
        "min_num_clusters",
        "max_num_clusters",
        "std_num_clusters",
        "avg_noise_ratio",
        "avg_runtime_sec",
        "frames_with_less_than_2_clusters",
        "ratio_frames_at_least_2_clusters",
    ]

    a_view = a_view[common_cols]
    b_view = b_view[common_cols]

    summary = pd.concat([a_view, b_view], ignore_index=True)
    summary.to_csv(OUT_PATH, index=False)

    print("=" * 80)
    print("A+B comparison summary")
    print("=" * 80)
    print(summary)
    print()
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
