import os
import time
import argparse
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


DATA_PATH = "outputs/inspect/students003_clean.csv"
OUT_DIR = "outputs/b_motion"


def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df["time_step"] = df["time_step"].astype(int)
    df["ID"] = df["ID"].astype(int)
    df = df.sort_values(["ID", "time_step"]).reset_index(drop=True)
    return df


def add_motion_features(df, window_size=5, momentum_alpha=0.7):
    """
    根据轨迹计算 B 类运动特征：
    1. 瞬时速度 vx, vy
    2. 滑动窗口速度 window_vx, window_vy
    3. 指数衰减动量 momentum_vx, momentum_vy
    4. 运动方向 cos_theta, sin_theta
    """
    df = df.copy()
    df = df.sort_values(["ID", "time_step"]).reset_index(drop=True)

    # 相邻帧速度
    df["prev_X"] = df.groupby("ID")["X"].shift(1)
    df["prev_Y"] = df.groupby("ID")["Y"].shift(1)
    df["prev_time"] = df.groupby("ID")["time_step"].shift(1)

    dt = df["time_step"] - df["prev_time"]

    df["vx"] = (df["X"] - df["prev_X"]) / dt
    df["vy"] = (df["Y"] - df["prev_Y"]) / dt

    df["vx"] = df["vx"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["vy"] = df["vy"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 滑动窗口速度
    df["win_X"] = df.groupby("ID")["X"].shift(window_size)
    df["win_Y"] = df.groupby("ID")["Y"].shift(window_size)
    df["win_time"] = df.groupby("ID")["time_step"].shift(window_size)

    dt_win = df["time_step"] - df["win_time"]

    df["window_vx"] = (df["X"] - df["win_X"]) / dt_win
    df["window_vy"] = (df["Y"] - df["win_Y"]) / dt_win

    df["window_vx"] = df["window_vx"].replace([np.inf, -np.inf], np.nan)
    df["window_vy"] = df["window_vy"].replace([np.inf, -np.inf], np.nan)

    # 刚出现或历史不足的行人，用瞬时速度补充
    df["window_vx"] = df["window_vx"].fillna(df["vx"]).fillna(0.0)
    df["window_vy"] = df["window_vy"].fillna(df["vy"]).fillna(0.0)

    # 指数衰减动量
    momentum_vx = []
    momentum_vy = []

    for pid, group in df.groupby("ID", sort=False):
        mx_prev = 0.0
        my_prev = 0.0

        for _, row in group.iterrows():
            mx = momentum_alpha * mx_prev + (1.0 - momentum_alpha) * row["vx"]
            my = momentum_alpha * my_prev + (1.0 - momentum_alpha) * row["vy"]

            momentum_vx.append(mx)
            momentum_vy.append(my)

            mx_prev = mx
            my_prev = my

    df["momentum_vx"] = momentum_vx
    df["momentum_vy"] = momentum_vy

    # 运动方向近似朝向
    speed = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
    eps = 1e-8

    df["cos_theta"] = df["vx"] / (speed + eps)
    df["sin_theta"] = df["vy"] / (speed + eps)

    # 对静止或刚出现的行人，方向设为 0
    stationary_mask = speed < eps
    df.loc[stationary_mask, "cos_theta"] = 0.0
    df.loc[stationary_mask, "sin_theta"] = 0.0

    drop_cols = [
        "prev_X", "prev_Y", "prev_time",
        "win_X", "win_Y", "win_time"
    ]
    df = df.drop(columns=drop_cols)

    return df


def get_feature_columns(feature_mode):
    if feature_mode == "instant_velocity":
        return ["X", "Y", "vx", "vy"]

    if feature_mode == "window_velocity":
        return ["X", "Y", "window_vx", "window_vy"]

    if feature_mode == "momentum":
        return ["X", "Y", "momentum_vx", "momentum_vy"]

    if feature_mode == "direction":
        return ["X", "Y", "cos_theta", "sin_theta"]

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def count_valid_clusters(labels):
    labels = np.asarray(labels)
    unique_labels = set(labels.tolist())
    if -1 in unique_labels:
        unique_labels.remove(-1)
    return len(unique_labels)


def compute_noise_ratio(labels):
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    return float(np.sum(labels == -1) / len(labels))


def dbscan_with_kmeans_fallback(
    features,
    eps=0.8,
    min_samples=2,
    fallback_k=2,
    random_state=42
):
    """
    先 DBSCAN 自动发现组。
    如果有效组数少于 2，则使用 KMeans 兜底。
    返回：
    - labels: 最终用于可视化的标签
    - raw_labels: DBSCAN 原始标签
    - raw_num_clusters: DBSCAN 原始有效簇数
    - raw_noise_ratio: DBSCAN 原始噪声比例
    - used_fallback: 是否使用 KMeans 兜底
    """
    n_samples = features.shape[0]

    if n_samples == 0:
        return np.array([]), np.array([]), 0, 0.0, False

    if n_samples == 1:
        return np.array([0]), np.array([0]), 1, 0.0, False

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    raw_labels = dbscan.fit_predict(features)

    raw_num_clusters = count_valid_clusters(raw_labels)
    raw_noise_ratio = compute_noise_ratio(raw_labels)

    if raw_num_clusters >= 2:
        return raw_labels, raw_labels, raw_num_clusters, raw_noise_ratio, False

    k = min(fallback_k, n_samples)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)

    return labels, raw_labels, raw_num_clusters, raw_noise_ratio, True


def run_motion_clustering(
    df,
    feature_mode,
    eps=0.8,
    min_samples=2,
    window_size=5,
    momentum_alpha=0.7,
    position_weight=1.0,
    motion_weight=1.0,
):
    os.makedirs(OUT_DIR, exist_ok=True)

    feature_cols = get_feature_columns(feature_mode)

    # 全局标准化，保证不同帧的特征尺度一致
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].values)

    all_results = []
    frame_stats = []

    total_start = time.time()

    for time_step, frame_df in df.groupby("time_step"):
        frame_df = frame_df.copy()
        features = frame_df[feature_cols].values
        features_scaled = scaler.transform(features)

        # 位置与运动特征权重
        features_scaled[:, 0:2] *= position_weight
        features_scaled[:, 2:] *= motion_weight

        start = time.time()
        labels, raw_labels, raw_num_clusters, raw_noise_ratio, used_fallback = dbscan_with_kmeans_fallback(
            features_scaled,
            eps=eps,
            min_samples=min_samples,
            fallback_k=2
        )
        runtime = time.time() - start

        frame_df["cluster_id"] = labels
        frame_df["raw_dbscan_cluster_id"] = raw_labels
        frame_df["feature_mode"] = feature_mode
        frame_df["method"] = "dbscan_kmeans_fallback"
        frame_df["used_fallback"] = used_fallback

        num_clusters = count_valid_clusters(labels)
        noise_ratio = compute_noise_ratio(labels)

        all_results.append(frame_df)

        frame_stats.append({
            "time_step": time_step,
            "method": "dbscan_kmeans_fallback",
            "feature_mode": feature_mode,
            "num_pedestrians": len(frame_df),
            "num_clusters": num_clusters,
            "noise_ratio": noise_ratio,
            "raw_dbscan_num_clusters": raw_num_clusters,
            "raw_dbscan_noise_ratio": raw_noise_ratio,
            "used_fallback": int(used_fallback),
            "runtime_sec": runtime,
            "eps": eps,
            "min_samples": min_samples,
            "window_size": window_size,
            "momentum_alpha": momentum_alpha,
            "position_weight": position_weight,
            "motion_weight": motion_weight,
        })

    total_runtime = time.time() - total_start

    result_df = pd.concat(all_results, ignore_index=True)
    stats_df = pd.DataFrame(frame_stats)

    method_name = (
        f"b_{feature_mode}"
        f"_eps{eps}_min{min_samples}"
        f"_w{window_size}_a{momentum_alpha}"
        f"_pw{position_weight}_mw{motion_weight}"
    ).replace(".", "p")

    result_path = os.path.join(OUT_DIR, f"clusters_{method_name}.csv")
    stats_path = os.path.join(OUT_DIR, f"stats_{method_name}.csv")
    summary_path = os.path.join(OUT_DIR, f"summary_{method_name}.csv")

    result_df.to_csv(result_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    summary = {
        "method": "dbscan_kmeans_fallback",
        "feature_mode": feature_mode,
        "num_frames": int(stats_df["time_step"].nunique()),
        "avg_num_clusters": float(stats_df["num_clusters"].mean()),
        "min_num_clusters": int(stats_df["num_clusters"].min()),
        "max_num_clusters": int(stats_df["num_clusters"].max()),
        "std_num_clusters": float(stats_df["num_clusters"].std()),
        "avg_noise_ratio": float(stats_df["noise_ratio"].mean()),
        "avg_raw_dbscan_num_clusters": float(stats_df["raw_dbscan_num_clusters"].mean()),
        "avg_raw_dbscan_noise_ratio": float(stats_df["raw_dbscan_noise_ratio"].mean()),
        "fallback_frames": int(stats_df["used_fallback"].sum()),
        "fallback_ratio": float(stats_df["used_fallback"].mean()),
        "avg_runtime_sec": float(stats_df["runtime_sec"].mean()),
        "total_runtime_sec": float(total_runtime),
        "frames_with_less_than_2_clusters": int((stats_df["num_clusters"] < 2).sum()),
        "ratio_frames_at_least_2_clusters": float((stats_df["num_clusters"] >= 2).mean()),
        "eps": eps,
        "min_samples": min_samples,
        "window_size": window_size,
        "momentum_alpha": momentum_alpha,
        "position_weight": position_weight,
        "motion_weight": motion_weight,
        "result_path": result_path,
        "stats_path": stats_path,
    }

    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    return summary, result_path, stats_path, summary_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature-mode",
        type=str,
        required=True,
        choices=["instant_velocity", "window_velocity", "momentum", "direction"]
    )

    parser.add_argument("--eps", type=float, default=0.8)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--momentum-alpha", type=float, default=0.7)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--motion-weight", type=float, default=1.0)

    args = parser.parse_args()

    print("=" * 80)
    print("Stage 3: B-class motion feature enhanced clustering")
    print("=" * 80)

    df = load_data(DATA_PATH)
    print("\n[1] Data loaded")
    print("Data shape:", df.shape)
    print("Number of frames:", df["time_step"].nunique())

    print("\n[2] Adding motion features")
    df = add_motion_features(
        df,
        window_size=args.window_size,
        momentum_alpha=args.momentum_alpha
    )
    print("Columns:", df.columns.tolist())

    print("\n[3] Running B-class method")
    print("feature_mode:", args.feature_mode)
    print("eps:", args.eps)
    print("min_samples:", args.min_samples)
    print("window_size:", args.window_size)
    print("momentum_alpha:", args.momentum_alpha)
    print("position_weight:", args.position_weight)
    print("motion_weight:", args.motion_weight)

    summary, result_path, stats_path, summary_path = run_motion_clustering(
        df=df,
        feature_mode=args.feature_mode,
        eps=args.eps,
        min_samples=args.min_samples,
        window_size=args.window_size,
        momentum_alpha=args.momentum_alpha,
        position_weight=args.position_weight,
        motion_weight=args.motion_weight
    )

    print("\n[4] Summary")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n[5] Saved files")
    print("result_path:", result_path)
    print("stats_path:", stats_path)
    print("summary_path:", summary_path)


if __name__ == "__main__":
    main()
