import numpy as np

def relation_graph_clustering(avg_pos_dist, avg_vel_dist,
                              dist_threshold=2.5,
                              vel_threshold=0.20):
    """
    基于关系图的行人组发现方法。
    若两名行人在时间窗口内平均距离较近且速度差较小，则连边；
    最终将图中的连通分量作为行人组。
    """
    n = avg_pos_dist.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            close_in_space = avg_pos_dist[i, j] <= dist_threshold
            similar_motion = avg_vel_dist[i, j] <= vel_threshold
            if close_in_space and similar_motion:
                union(i, j)

    root_to_label = {}
    labels = np.zeros(n, dtype=int)
    next_label = 0

    for i in range(n):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[i] = root_to_label[root]

    return labels
