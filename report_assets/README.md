# Report Assets

本目录用于保存最终 LaTeX 实验报告所需的图片、表格和关键代码。

## figures

| 文件 | 用途 |
|---|---|
| a_dbscan_frame_0250.png | A 类 DBSCAN 关键帧 |
| b_window_velocity_frame_0250.png | B 类改进 window_velocity 关键帧 |
| c_relation_graph_frame_0250.png | C 类 relation_graph 关键帧 |
| final_frame_0250.png | Final 最终方法关键帧 |
| avg_num_clusters.png | 平均簇数对比图 |
| avg_noise_ratio.png | 噪声比例对比图 |
| mean_cluster_count_change.png | 连续帧稳定性对比图 |
| std_num_clusters.png | 簇数量波动对比图 |

## tables

| 文件 | 用途 |
|---|---|
| dataset_summary.csv | 数据集统计 |
| selected_method_summary.csv | 代表方法对比 |
| selected_stability_summary.csv | 稳定性对比 |
| final_summary.csv | 最终方法结果 |

## code

| 文件 | 用途 |
|---|---|
| data_loading.py | 数据读取与逐帧组织 |
| motion_features.py | 运动特征构造 |
| dbscan_fallback.py | DBSCAN + KMeans 兜底策略 |
| temporal_relation.py | 关系图方法 |
| visualization.py | 可视化与视频生成 |
| final_pipeline.py | Final 主流程 |
