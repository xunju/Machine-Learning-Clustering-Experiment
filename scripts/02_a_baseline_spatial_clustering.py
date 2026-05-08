import os
import time
import argparse
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


DATA_PATH = "outputs/inspect/students003_clean.csv"
OUT_DIR = "outputs/a_baseline"


def load_data(data_path):
    """
    读取第一阶段清洗后的 students003 数据。
    数据列包括：time_step, ID, X, Y。
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df["time_step"] = df["time_step"].astype(int)
    df["ID"] = df["ID"].astype(int)
    return df


def count_valid_clusters(labels):
    """
    统计有效簇数量。
    对于 DBSCAN，-1 表示噪声点，不计入有效簇。
    """
    unique_labels = set(labels.tolist())
    if -1 in unique_labels:
        unique_labels.remove(-1)
    return len(unique_labels)


def compute_noise_ratio(labels):
    """
    计算噪声点比例。
    只有 DBSCAN 可能产生 -1 标签。
    """
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    return float(np.sum(labels == -1) / len(labels))


def cluster_one_frame(points, method, n_clusters=3, eps=0.6, min_samples=2, random_state=42):
    """
    对单帧的二维坐标点进行聚类。
    points: shape = [num_pedestrians, 2]
    """
    n_samples = points.shape[0]

    if n_samples == 0:
        return np.array([])

    if n_samples == 1:
        return np.array([0])

    if method == "kmeans":
        k = min(n_clusters, n_samples)
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        return model.fit_predict(points)

    if method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        return model.fit_predict(points)

    if method == "agglomerative":
        k = min(n_clusters, n_samples)
        model = AgglomerativeClustering(n_clusters=k)
        return model.fit_predict(points)

    if method == "gmm":
        k = min(n_clusters, n_samples)
        model = GaussianMixture(n_components=k, random_state=random_state)
        return model.fit_predict(points)

    raise ValueError(f"Unknown method: {method}")


def run_spatial_clustering(df, method, n_clusters=3, eps=0.6, min_samples=2):
    """
    对所有帧逐帧进行基础空间聚类。
    本阶段只使用 X, Y 两个特征。
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = []
    frame_stats = []

    total_start = time.time()

    for time_step, frame_df in df.groupby("time_step"):
        frame_df = frame_df.copy()

        points = frame_df[["X", "Y"]].values

        # 单帧标准化，使每一帧中 X/Y 坐标在距离计算中尺度一致
        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(points)

        start = time.time()
        labels = cluster_one_frame(
            points=points_scaled,
            method=method,
            n_clusters=n_clusters,
            eps=eps,
            min_samples=min_samples
        )
        runtime = time.time() - start

        frame_df["cluster_id"] = labels
        frame_df["method"] = method
        frame_df["feature_type"] = "spatial_xy"

        num_clusters = count_valid_clusters(labels)
        noise_ratio = compute_noise_ratio(labels)

        all_results.append(frame_df)

        frame_stats.append({
            "time_step": time_step,
            "method": method,
            "feature_type": "spatial_xy",
            "num_pedestrians": len(frame_df),
            "num_clusters": num_clusters,
            "noise_ratio": noise_ratio,
            "runtime_sec": runtime,
            "n_clusters_param": n_clusters,
            "eps": eps,
            "min_samples": min_samples,
        })

    total_runtime = time.time() - total_start

    result_df = pd.concat(all_results, ignore_index=True)
    stats_df = pd.DataFrame(frame_stats)

    method_name = f"{method}_xy_k{n_clusters}_eps{eps}_min{min_samples}"
    method_name = method_name.replace(".", "p")

    result_path = os.path.join(OUT_DIR, f"clusters_{method_name}.csv")
    stats_path = os.path.join(OUT_DIR, f"stats_{method_name}.csv")
    summary_path = os.path.join(OUT_DIR, f"summary_{method_name}.csv")

    result_df.to_csv(result_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    summary = {
        "method": method,
        "feature_type": "spatial_xy",
        "num_frames": int(stats_df["time_step"].nunique()),
        "avg_num_clusters": float(stats_df["num_clusters"].mean()),
        "min_num_clusters": int(stats_df["num_clusters"].min()),
        "max_num_clusters": int(stats_df["num_clusters"].max()),
        "std_num_clusters": float(stats_df["num_clusters"].std()),
        "avg_noise_ratio": float(stats_df["noise_ratio"].mean()),
        "avg_runtime_sec": float(stats_df["runtime_sec"].mean()),
        "total_runtime_sec": float(total_runtime),
        "frames_with_less_than_2_clusters": int((stats_df["num_clusters"] < 2).sum()),
        "ratio_frames_at_least_2_clusters": float((stats_df["num_clusters"] >= 2).mean()),
        "n_clusters_param": n_clusters,
        "eps": eps,
        "min_samples": min_samples,
        "result_path": result_path,
        "stats_path": stats_path,
    }

    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    return summary, result_path, stats_path, summary_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["kmeans", "dbscan", "agglomerative", "gmm"]
    )

    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.6)
    parser.add_argument("--min-samples", type=int, default=2)

    args = parser.parse_args()

    print("=" * 80)
    print("Stage 2: A-class baseline spatial clustering")
    print("=" * 80)

    df = load_data(DATA_PATH)

    print("\n[1] Data loaded")
    print("Data shape:", df.shape)
    print("Number of frames:", df["time_step"].nunique())
    print("Columns:", df.columns.tolist())

    print("\n[2] Running baseline method")
    print("method:", args.method)
    print("n_clusters:", args.n_clusters)
    print("eps:", args.eps)
    print("min_samples:", args.min_samples)

    summary, result_path, stats_path, summary_path = run_spatial_clustering(
        df=df,
        method=args.method,
        n_clusters=args.n_clusters,
        eps=args.eps,
        min_samples=args.min_samples
    )

    print("\n[3] Summary")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n[4] Saved files")
    print("result_path:", result_path)
    print("stats_path:", stats_path)
    print("summary_path:", summary_path)


if __name__ == "__main__":
    main()
