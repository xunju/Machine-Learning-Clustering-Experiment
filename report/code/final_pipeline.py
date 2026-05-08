from sklearn.preprocessing import StandardScaler

def run_final_frame(frame_df, feature_cols, scaler,
                    eps=1.0,
                    min_samples=2,
                    position_weight=1.0,
                    motion_weight=0.5):
    """
    Final 方法的单帧处理流程：
    1. 提取 X, Y, window_vx, window_vy；
    2. 标准化并设置位置/运动权重；
    3. 使用 DBSCAN + KMeans fallback 得到最终分组。
    """
    features = frame_df[feature_cols].values
    features = scaler.transform(features)

    features[:, 0:2] *= position_weight
    features[:, 2:] *= motion_weight

    labels, raw_labels, used_fallback = dbscan_with_kmeans_fallback(
        features,
        eps=eps,
        min_samples=min_samples,
        fallback_k=2
    )

    frame_df = frame_df.copy()
    frame_df["cluster_id"] = labels
    frame_df["raw_dbscan_cluster_id"] = raw_labels
    frame_df["used_fallback"] = used_fallback

    return frame_df
