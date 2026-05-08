# 第四阶段：C 类方法——时序关系建模

## 1. 阶段目标

本阶段在 A 类单帧空间聚类和 B 类运动特征增强的基础上，进一步引入行人之间的时序关系。与 A、B 类方法直接对单个行人的特征向量聚类不同，C 类方法重点建模“行人与行人之间是否在连续多帧中保持接近和共同运动”。

本阶段实现三类方法：

1. 持续邻近关系方法；
2. 基于关系图的图聚类方法；
3. 基于 Pairwise Affinity 的谱聚类方法。

## 2. 方法思想

持续同行的行人通常不仅在某一帧中空间距离较近，而且在连续多帧中保持相对接近，并具有相似的运动趋势。因此，本阶段为当前帧中的任意两个行人计算最近时间窗口内的平均空间距离和平均速度差，并基于这些 pairwise relation 进行分组。

## 3. 输出内容

每种方法输出：

| 文件类型 | 作用 |
|---|---|
| clusters_*.csv | 每帧每个行人的分组结果 |
| stats_*.csv | 每帧组数、运行时间等统计 |
| c_temporal_summary.csv | C 类方法汇总表 |
| frames/*.png | 每帧可视化 |
| video/*.mp4 | 方法视频 |
| report_figures/*.png | 报告关键帧 |

## 4. 评价指标

由于没有真实行人组标签，本阶段继续使用无监督指标，包括平均簇数量、簇数量波动、平均运行时间、至少 2 组帧比例，并结合可视化分析结果合理性。

## 5. 实际运行结果

本阶段实现了持续邻近关系层次聚类、关系图连通分量方法以及 Pairwise Affinity 谱聚类三类时序关系建模方法。实验结果如下：

| 方法 | 参数 | 平均簇数 | 最少簇数 | 最多簇数 | 簇数标准差 | 兜底帧数 | 至少 2 组比例 |
|---|---|---:|---:|---:|---:|---:|---:|
| temporal_agglomerative | n_clusters=3 | 3.000 | 3 | 3 | 0.000 | 0 | 1.000 |
| relation_graph | dt=1.2, vt=0.08 | 15.706 | 9 | 24 | 2.900 | 0 | 1.000 |
| relation_graph | dt=2.0, vt=0.15 | 7.810 | 3 | 15 | 1.974 | 0 | 1.000 |
| relation_graph | dt=2.5, vt=0.20 | 4.858 | 2 | 10 | 1.597 | 2 | 1.000 |
| affinity_spectral | n_clusters=3 | 3.000 | 3 | 3 | 0.000 | 0 | 1.000 |

## 6. 参数现象分析

关系图方法通过阈值判断两名行人之间是否连边，因此对距离阈值和速度阈值较为敏感。当 dist_threshold=1.2、vel_threshold=0.08 时，连边条件较严格，导致图中边较少，平均每帧形成约 15.71 个连通分量，存在明显过度切分现象。将阈值放宽到 dist_threshold=2.0、vel_threshold=0.15 后，平均簇数下降到约 7.81。进一步放宽到 dist_threshold=2.5、vel_threshold=0.20 后，平均簇数下降到约 4.86，簇数量波动也进一步减小，因此本文选择该参数作为 C2 关系图方法的代表结果。

## 7. 阶段性结论

C 类方法与 A、B 类方法的主要区别在于，它不再只为单个行人构造特征，而是直接建模行人与行人之间的时序关系。实验结果表明，固定簇数的 temporal_agglomerative 和 affinity_spectral 方法具有较强稳定性，但不能自动决定组数；关系图方法能够根据持续邻近和运动相似关系自动形成不同数量的组，更符合 Group Discovery 的任务目标，但其结果对连边阈值较为敏感。综合统计结果，本文将 relation_graph, dist_threshold=2.5, vel_threshold=0.20 作为 C 类代表性自动分组方法。

## 8. 全帧可视化与视频生成

本阶段对 temporal_agglomerative、relation_graph(dt=2.5, vt=0.20) 和 affinity_spectral 三种代表方法生成了完整逐帧可视化图片和 MP4 视频。不同颜色表示不同分组，点旁数字表示行人 ID，坐标轴范围保持固定。输出视频保存在：

- outputs/c_temporal/full_visualization/c_temporal_agglomerative/video/c_temporal_agglomerative.mp4
- outputs/c_temporal/full_visualization/c_relation_graph_dt2p5_vt0p20/video/c_relation_graph_dt2p5_vt0p20.mp4
- outputs/c_temporal/full_visualization/c_affinity_spectral/video/c_affinity_spectral.mp4

报告关键帧保存在：

- outputs/c_temporal/report_figures/

## 9. 可视化观察补充

从 relation_graph(dt=2.5, vt=0.20) 的关键帧可视化结果可以看出，关系图方法能够根据持续邻近关系自动形成不同数量的行人组，相比固定簇数方法更具有自适应性。然而，该方法也存在连通分量传播带来的过度合并问题：当多个局部小组之间通过中间行人形成连接时，图方法可能将较大区域内的行人合并为同一组。

因此，关系图方法的核心优势在于能够直接表达“持续同行关系”，但其结果对连边阈值较为敏感。阈值过严会导致图中边过少，产生过度切分；阈值过松则可能导致不同组被连接为较大的连通分量。本文最终选择 dist_threshold=2.5、vel_threshold=0.20 作为关系图方法的代表参数，是在簇数量、簇数量波动和可视化效果之间进行折中后的结果。
