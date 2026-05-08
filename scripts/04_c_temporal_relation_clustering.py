import os
import time
import argparse
import itertools
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.preprocessing import StandardScaler


DATA_PATH = "outputs/inspect/students003_clean.csv"
OUT_DIR = "outputs/c_temporal"


def load_data(data_path):
    df = pd.read_csv(data_path)
    df["time_step"] = df["time_step"].astype(int)
    df["ID"] = df["ID"].astype(int)
    df = df.sort_values(["ID", "time_step"]).reset_index(drop=True)
    return df


def add_velocity(df):
    df = df.copy()
    df = df.sort_values(["ID", "time_step"]).reset_index(drop=True)

    df["prev_X"] = df.groupby("ID")["X"].shift(1)
    df["prev_Y"] = df.groupby("ID")["Y"].shift(1)
    df["prev_time"] = df.groupby("ID")["time_step"].shift(1)

    dt = df["time_step"] - df["prev_time"]

    df["vx"] = (df["X"] - df["prev_X"]) / dt
    df["vy"] = (df["Y"] - df["prev_Y"]) / dt

    df["vx"] = df["vx"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["vy"] = df["vy"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df = df.drop(columns=["prev_X", "prev_Y", "prev_time"])
    return df


def get_recent_times(all_times, current_time, window_size):
    all_times = sorted(all_times)
    idx = all_times.index(current_time)
    start_idx = max(0, idx - window_size + 1)
    return all_times[start_idx: idx + 1]


def build_lookup(df):
    """
    构造 (time_step, ID) -> (X, Y, vx, vy) 查询表。
    """
    lookup = {}
    for _, row in df.iterrows():
        lookup[(int(row["time_step"]), int(row["ID"]))] = (
            float(row["X"]),
            float(row["Y"]),
            float(row["vx"]),
            float(row["vy"]),
        )
    return lookup


def compute_pairwise_relation_matrix(
    df,
    lookup,
    current_time,
    window_size=5,
    beta=0.7,
):
    """
    为当前帧计算 pairwise relation distance matrix。

    distance = beta * normalized_position_distance
             + (1 - beta) * normalized_velocity_difference
    """
    current_df = df[df["time_step"] == current_time].copy()
    current_ids = current_df["ID"].astype(int).tolist()

    n = len(current_ids)
    distance_matrix = np.zeros((n, n), dtype=float)

    all_times = sorted(df["time_step"].unique().tolist())
    recent_times = get_recent_times(all_times, current_time, window_size)

    # 保存原始统计用于分析
    cooccur_matrix = np.zeros((n, n), dtype=float)
    avg_pos_dist_matrix = np.zeros((n, n), dtype=float)
    avg_vel_dist_matrix = np.zeros((n, n), dtype=float)

    for a, b in itertools.combinations(range(n), 2):
        id_i = current_ids[a]
        id_j = current_ids[b]

        pos_dists = []
        vel_dists = []

        for t in recent_times:
            key_i = (t, id_i)
            key_j = (t, id_j)

            if key_i not in lookup or key_j not in lookup:
                continue

            xi, yi, vxi, vyi = lookup[key_i]
            xj, yj, vxj, vyj = lookup[key_j]

            pos_dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            vel_dist = np.sqrt((vxi - vxj) ** 2 + (vyi - vyj) ** 2)

            pos_dists.append(pos_dist)
            vel_dists.append(vel_dist)

        if len(pos_dists) == 0:
            # 没有共同出现历史，使用当前帧距离作为退化情况
            row_i = current_df[current_df["ID"] == id_i].iloc[0]
            row_j = current_df[current_df["ID"] == id_j].iloc[0]
            pos_avg = np.sqrt((row_i["X"] - row_j["X"]) ** 2 + (row_i["Y"] - row_j["Y"]) ** 2)
            vel_avg = 1.0
            cooccur = 0
        else:
            pos_avg = float(np.mean(pos_dists))
            vel_avg = float(np.mean(vel_dists))
            cooccur = len(pos_dists)

        cooccur_matrix[a, b] = cooccur
        cooccur_matrix[b, a] = cooccur

        avg_pos_dist_matrix[a, b] = pos_avg
        avg_pos_dist_matrix[b, a] = pos_avg

        avg_vel_dist_matrix[a, b] = vel_avg
        avg_vel_dist_matrix[b, a] = vel_avg

    # 归一化距离矩阵，避免位置距离和速度差尺度不一致
    pos_vals = avg_pos_dist_matrix[np.triu_indices(n, k=1)]
    vel_vals = avg_vel_dist_matrix[np.triu_indices(n, k=1)]

    pos_scale = np.median(pos_vals[pos_vals > 0]) if np.any(pos_vals > 0) else 1.0
    vel_scale = np.median(vel_vals[vel_vals > 0]) if np.any(vel_vals > 0) else 1.0

    pos_norm = avg_pos_dist_matrix / (pos_scale + 1e-8)
    vel_norm = avg_vel_dist_matrix / (vel_scale + 1e-8)

    distance_matrix = beta * pos_norm + (1.0 - beta) * vel_norm
    np.fill_diagonal(distance_matrix, 0.0)

    return current_df, current_ids, distance_matrix, avg_pos_dist_matrix, avg_vel_dist_matrix, cooccur_matrix


def count_valid_clusters(labels):
    labels = np.asarray(labels)
    unique_labels = set(labels.tolist())
    if -1 in unique_labels:
        unique_labels.remove(-1)
    return len(unique_labels)


def fallback_kmeans_if_needed(current_df, labels, fallback_k=2):
    if count_valid_clusters(labels) >= 2:
        return labels, False

    points = current_df[["X", "Y"]].values
    points = StandardScaler().fit_transform(points)
    k = min(fallback_k, len(points))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    new_labels = km.fit_predict(points)
    return new_labels, True


def cluster_temporal_agglomerative(distance_matrix, n_clusters=3):
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)

    k = min(n_clusters, n)

    try:
        model = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average"
        )
    except TypeError:
        # 兼容旧版 sklearn
        model = AgglomerativeClustering(
            n_clusters=k,
            affinity="precomputed",
            linkage="average"
        )

    return model.fit_predict(distance_matrix)


def cluster_relation_graph(avg_pos_dist_matrix, avg_vel_dist_matrix, dist_threshold=1.2, vel_threshold=0.08):
    """
    简单关系图连通分量：
    若两人最近窗口平均距离小于 dist_threshold 且速度差小于 vel_threshold，则连边。
    """
    n = avg_pos_dist_matrix.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if avg_pos_dist_matrix[i, j] <= dist_threshold and avg_vel_dist_matrix[i, j] <= vel_threshold:
                union(i, j)

    root_to_label = {}
    labels = np.zeros(n, dtype=int)
    next_label = 0

    for i in range(n):
        r = find(i)
        if r not in root_to_label:
            root_to_label[r] = next_label
            next_label += 1
        labels[i] = root_to_label[r]

    return labels


def cluster_affinity_spectral(avg_pos_dist_matrix, avg_vel_dist_matrix, n_clusters=3, sigma_d=2.0, sigma_v=0.1):
    n = avg_pos_dist_matrix.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)

    k = min(n_clusters, n)

    affinity = np.exp(-(avg_pos_dist_matrix ** 2) / (2 * sigma_d ** 2)) * \
               np.exp(-(avg_vel_dist_matrix ** 2) / (2 * sigma_v ** 2))

    np.fill_diagonal(affinity, 1.0)

    model = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        random_state=42,
        assign_labels="kmeans"
    )

    return model.fit_predict(affinity)


def run_temporal_method(
    df,
    method,
    window_size=5,
    beta=0.7,
    n_clusters=3,
    dist_threshold=1.2,
    vel_threshold=0.08,
    sigma_d=2.0,
    sigma_v=0.1,
):
    os.makedirs(OUT_DIR, exist_ok=True)

    lookup = build_lookup(df)
    time_steps = sorted(df["time_step"].unique().tolist())

    all_results = []
    frame_stats = []

    total_start = time.time()

    for time_step in time_steps:
        start = time.time()

        current_df, current_ids, distance_matrix, avg_pos_dist_matrix, avg_vel_dist_matrix, cooccur_matrix = compute_pairwise_relation_matrix(
            df=df,
            lookup=lookup,
            current_time=time_step,
            window_size=window_size,
            beta=beta
        )

        if method == "temporal_agglomerative":
            labels = cluster_temporal_agglomerative(
                distance_matrix,
                n_clusters=n_clusters
            )

        elif method == "relation_graph":
            labels = cluster_relation_graph(
                avg_pos_dist_matrix,
                avg_vel_dist_matrix,
                dist_threshold=dist_threshold,
                vel_threshold=vel_threshold
            )

        elif method == "affinity_spectral":
            labels = cluster_affinity_spectral(
                avg_pos_dist_matrix,
                avg_vel_dist_matrix,
                n_clusters=n_clusters,
                sigma_d=sigma_d,
                sigma_v=sigma_v
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        raw_num_clusters = count_valid_clusters(labels)
        labels, used_fallback = fallback_kmeans_if_needed(current_df, labels, fallback_k=2)

        runtime = time.time() - start

        current_df = current_df.copy()
        current_df["cluster_id"] = labels
        current_df["method"] = method
        current_df["feature_mode"] = "temporal_relation"
        current_df["used_fallback"] = used_fallback

        num_clusters = count_valid_clusters(labels)

        all_results.append(current_df)

        frame_stats.append({
            "time_step": time_step,
            "method": method,
            "feature_mode": "temporal_relation",
            "num_pedestrians": len(current_df),
            "num_clusters": num_clusters,
            "raw_num_clusters": raw_num_clusters,
            "used_fallback": int(used_fallback),
            "runtime_sec": runtime,
            "window_size": window_size,
            "beta": beta,
            "n_clusters_param": n_clusters,
            "dist_threshold": dist_threshold,
            "vel_threshold": vel_threshold,
            "sigma_d": sigma_d,
            "sigma_v": sigma_v,
        })

    total_runtime = time.time() - total_start

    result_df = pd.concat(all_results, ignore_index=True)
    stats_df = pd.DataFrame(frame_stats)

    method_name = (
        f"c_{method}"
        f"_w{window_size}_b{beta}_k{n_clusters}"
        f"_dt{dist_threshold}_vt{vel_threshold}"
        f"_sd{sigma_d}_sv{sigma_v}"
    ).replace(".", "p")

    result_path = os.path.join(OUT_DIR, f"clusters_{method_name}.csv")
    stats_path = os.path.join(OUT_DIR, f"stats_{method_name}.csv")
    summary_path = os.path.join(OUT_DIR, f"summary_{method_name}.csv")

    result_df.to_csv(result_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    summary = {
        "method": method,
        "feature_mode": "temporal_relation",
        "num_frames": int(stats_df["time_step"].nunique()),
        "avg_num_clusters": float(stats_df["num_clusters"].mean()),
        "min_num_clusters": int(stats_df["num_clusters"].min()),
        "max_num_clusters": int(stats_df["num_clusters"].max()),
        "std_num_clusters": float(stats_df["num_clusters"].std()),
        "avg_raw_num_clusters": float(stats_df["raw_num_clusters"].mean()),
        "fallback_frames": int(stats_df["used_fallback"].sum()),
        "fallback_ratio": float(stats_df["used_fallback"].mean()),
        "avg_runtime_sec": float(stats_df["runtime_sec"].mean()),
        "total_runtime_sec": float(total_runtime),
        "frames_with_less_than_2_clusters": int((stats_df["num_clusters"] < 2).sum()),
        "ratio_frames_at_least_2_clusters": float((stats_df["num_clusters"] >= 2).mean()),
        "window_size": window_size,
        "beta": beta,
        "n_clusters_param": n_clusters,
        "dist_threshold": dist_threshold,
        "vel_threshold": vel_threshold,
        "sigma_d": sigma_d,
        "sigma_v": sigma_v,
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
        choices=["temporal_agglomerative", "relation_graph", "affinity_spectral"]
    )

    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--n-clusters", type=int, default=3)

    parser.add_argument("--dist-threshold", type=float, default=1.2)
    parser.add_argument("--vel-threshold", type=float, default=0.08)

    parser.add_argument("--sigma-d", type=float, default=2.0)
    parser.add_argument("--sigma-v", type=float, default=0.1)

    args = parser.parse_args()

    print("=" * 80)
    print("Stage 4: C-class temporal relation clustering")
    print("=" * 80)

    df = load_data(DATA_PATH)
    df = add_velocity(df)

    print("[1] Data loaded")
    print("Data shape:", df.shape)
    print("Number of frames:", df["time_step"].nunique())

    print("[2] Running method")
    print("method:", args.method)

    summary, result_path, stats_path, summary_path = run_temporal_method(
        df=df,
        method=args.method,
        window_size=args.window_size,
        beta=args.beta,
        n_clusters=args.n_clusters,
        dist_threshold=args.dist_threshold,
        vel_threshold=args.vel_threshold,
        sigma_d=args.sigma_d,
        sigma_v=args.sigma_v,
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
