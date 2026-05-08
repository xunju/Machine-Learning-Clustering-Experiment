import numpy as np
from sklearn.cluster import DBSCAN, KMeans

def count_valid_clusters(labels):
    """
    统计有效簇数量。DBSCAN 中 -1 表示噪声点，不计入有效簇。
    """
    valid = set(labels.tolist())
    valid.discard(-1)
    return len(valid)


def dbscan_with_kmeans_fallback(features, eps=1.0, min_samples=2, fallback_k=2):
    """
    先使用 DBSCAN 自动发现行人组；
    若有效簇数少于 2，则使用 KMeans(n_clusters=2) 进行兜底。
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    raw_labels = dbscan.fit_predict(features)

    raw_num_clusters = count_valid_clusters(raw_labels)

    if raw_num_clusters >= 2:
        return raw_labels, raw_labels, False

    k = min(fallback_k, len(features))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    fallback_labels = kmeans.fit_predict(features)

    return fallback_labels, raw_labels, True
