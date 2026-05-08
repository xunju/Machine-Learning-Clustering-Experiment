import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUMMARY_PATH = "outputs/d_analysis/abc_method_summary.csv"
STABILITY_PATH = "outputs/d_analysis/temporal_stability_summary.csv"
FIG_DIR = "outputs/d_analysis/figures"

os.makedirs(FIG_DIR, exist_ok=True)


def shorten_name(name, max_len=35):
    name = str(name)
    if len(name) <= max_len:
        return name
    return name[:max_len] + "..."


def plot_bar(df, x_col, y_col, title, ylabel, output_path):
    plt.figure(figsize=(12, 6))

    names = [shorten_name(x) for x in df[x_col]]
    values = df[y_col].values

    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), names, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print("Saved:", output_path)


def main():
    summary = pd.read_csv(SUMMARY_PATH)

    # 只画关键方法，避免图太挤
    key_methods = [
        "kmeans_spatial_xy",
        "dbscan_spatial_xy",
        "agglomerative_spatial_xy",
        "gmm_spatial_xy",
        "window_velocity_eps1.0_mw0.5",
        "momentum_eps1.0_mw0.5",
        "temporal_agglomerative",
        "relation_graph_dt2.5_vt0.2",
        "affinity_spectral",
    ]

    key_df = summary[summary["method_name"].isin(key_methods)].copy()

    if len(key_df) == 0:
        key_df = summary.copy()

    plot_bar(
        key_df,
        "method_name",
        "avg_num_clusters",
        "Average number of clusters by method",
        "Average number of clusters",
        os.path.join(FIG_DIR, "avg_num_clusters.png")
    )

    plot_bar(
        key_df,
        "method_name",
        "std_num_clusters",
        "Cluster count fluctuation by method",
        "Std. of cluster count",
        os.path.join(FIG_DIR, "std_num_clusters.png")
    )

    plot_bar(
        key_df,
        "method_name",
        "avg_noise_ratio",
        "Average noise ratio by method",
        "Average noise ratio",
        os.path.join(FIG_DIR, "avg_noise_ratio.png")
    )

    plot_bar(
        key_df,
        "method_name",
        "avg_runtime_sec",
        "Average runtime per frame by method",
        "Runtime per frame (sec)",
        os.path.join(FIG_DIR, "avg_runtime_sec.png")
    )

    if os.path.exists(STABILITY_PATH):
        stability = pd.read_csv(STABILITY_PATH)
        key_stability = stability[stability["method_name"].isin(key_methods)].copy()
        if len(key_stability) == 0:
            key_stability = stability.copy()

        plot_bar(
            key_stability,
            "method_name",
            "mean_cluster_count_change",
            "Temporal stability: mean cluster count change",
            "Mean |C_t - C_{t-1}|",
            os.path.join(FIG_DIR, "mean_cluster_count_change.png")
        )

    print("All analysis figures generated.")


if __name__ == "__main__":
    main()
