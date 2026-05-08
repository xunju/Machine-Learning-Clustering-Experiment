# Final 方法说明

## 1. 最终方法

本文最终采用 DBSCAN + KMeans fallback 策略生成 Group Discovery 视频。

最终特征为：

- X
- Y
- window_vx
- window_vy

其中 window_vx 和 window_vy 表示最近 5 个时间步内的滑动窗口平均速度。

## 2. 参数设置

| 参数 | 取值 |
|---|---:|
| eps | 1.0 |
| min_samples | 2 |
| window_size | 5 |
| position_weight | 1.0 |
| motion_weight | 0.5 |

## 3. 方法逻辑

每一帧首先使用 DBSCAN 自动发现行人组。若某一帧 DBSCAN 得到的有效组数少于 2，则使用 KMeans(n_clusters=2) 进行兜底划分，以满足作业要求中“每帧至少 2 组行人”的约束。

## 4. 可视化说明

最终视频中，不同颜色表示不同的行人组，点旁数字表示行人 ID。坐标轴范围固定为 X∈[-1,16]、Y∈[-1,15]。为了避免逐帧聚类标签编号变化导致颜色频繁跳变，视频渲染阶段使用 visual_group_id 对相邻帧组标签进行颜色对齐。该步骤仅用于可视化，不改变原始聚类结果。

## 5. 输出文件

| 文件 | 作用 |
|---|---|
| outputs/final/clusters_final.csv | 最终逐帧聚类结果 |
| outputs/final/stats_final.csv | 最终逐帧统计结果 |
| outputs/final/final_summary.csv | 最终方法总体指标 |
| outputs/final/frames/ | 最终逐帧可视化图片 |
| outputs/final/video/final_group_discovery.mp4 | 最终实验视频 |
| outputs/final/report_figures/ | 报告关键帧 |
