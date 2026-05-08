# 第二阶段：A 类方法——基础单帧空间聚类

## 1. 阶段目标

本阶段只使用每一帧中行人的二维坐标 X、Y 进行聚类，作为 Group Discovery 实验的基础 baseline。

本阶段不使用速度、滑动窗口、动量、朝向或时序关系信息。这样做的目的是先建立仅依赖单帧空间位置的基础对照，为后续 B 类运动特征增强、C 类时序关系建模和 Final 最终视频生成策略提供比较基础。

## 2. 输入数据

输入文件：

outputs/inspect/students003_clean.csv

数据字段包括：

| 字段 | 含义 |
|---|---|
| time_step | 时间步 / 帧编号 |
| ID | 行人编号 |
| X | 行人二维横坐标 |
| Y | 行人二维纵坐标 |

本阶段按 time_step 分组，对每一帧独立进行聚类。

## 3. 方法设计

本阶段选取四种典型无监督聚类方法作为 baseline。

### 3.1 KMeans

KMeans 是基于簇中心的划分聚类方法，需要预先指定簇数量。本实验设置 n_clusters=3，用于观察固定簇数量下的基础空间分组效果。

### 3.2 DBSCAN

DBSCAN 是基于密度的聚类方法，不需要预先指定簇数量，能够自动发现密集区域并识别噪声点。本阶段初始参数设置为 eps=0.6，min_samples=2。

### 3.3 Agglomerative Clustering

Agglomerative Clustering 是层次聚类方法，通过逐步合并距离较近的样本或簇得到最终分组。本实验设置 n_clusters=3。

### 3.4 GMM

GMM 使用多个高斯分布对样本分布进行概率建模。本实验设置 n_components=3，用于与 KMeans 等固定簇数方法进行对比。

## 4. 实验命令

### 4.1 KMeans

python scripts/02_a_baseline_spatial_clustering.py \
  --method kmeans \
  --n-clusters 3 \
  | tee outputs/a_baseline/kmeans_xy.log

### 4.2 DBSCAN

python scripts/02_a_baseline_spatial_clustering.py \
  --method dbscan \
  --eps 0.6 \
  --min-samples 2 \
  | tee outputs/a_baseline/dbscan_xy_eps0p6_min2.log

### 4.3 Agglomerative Clustering

python scripts/02_a_baseline_spatial_clustering.py \
  --method agglomerative \
  --n-clusters 3 \
  | tee outputs/a_baseline/agglomerative_xy.log

### 4.4 GMM

python scripts/02_a_baseline_spatial_clustering.py \
  --method gmm \
  --n-clusters 3 \
  | tee outputs/a_baseline/gmm_xy.log

### 4.5 汇总结果

python scripts/02_a_summarize_baseline.py

### 4.6 一键运行脚本
```bash
bash scripts/02_a_run_all_baselines.sh
```

## 5. 输出文件

本阶段主要输出文件包括：

| 文件 | 作用 |
|---|---|
| outputs/a_baseline/clusters_*.csv | 保存每一帧每个行人的聚类结果 |
| outputs/a_baseline/stats_*.csv | 保存每一帧的簇数量、噪声比例、运行时间等统计 |
| outputs/a_baseline/summary_*.csv | 保存单个方法的总体统计 |
| outputs/a_baseline/a_baseline_summary.csv | 保存四个 baseline 方法的汇总对比 |
| outputs/a_baseline/*.log | 保存每个方法的运行日志 |

## 6. 评价指标

由于数据集中没有真实行人组标签，本阶段不使用准确率、召回率或 F1 值等监督指标。

本阶段关注以下无监督统计指标：

| 指标 | 含义 |
|---|---|
| avg_num_clusters | 平均每帧聚类组数 |
| min_num_clusters | 最少聚类组数 |
| max_num_clusters | 最多聚类组数 |
| std_num_clusters | 聚类组数波动 |
| avg_noise_ratio | 平均噪声点比例 |
| avg_runtime_sec | 平均每帧运行时间 |
| frames_with_less_than_2_clusters | 聚类组数少于 2 的帧数 |
| ratio_frames_at_least_2_clusters | 至少包含 2 组的帧比例 |

## 7. 初步分析思路

KMeans、Agglomerative Clustering 和 GMM 都需要预先指定簇数量，因此它们通常能够稳定输出固定数量的组，但不具备自动确定组数的能力。

DBSCAN 不需要预先指定簇数量，更符合自动发现行人组的任务目标，但其结果受 eps 和 min_samples 参数影响较大，可能出现某些帧有效簇数量不足 2 的情况。

因此，A 类方法的作用主要是建立基础对照。后续实验将进一步引入运动特征和时序关系，以提高分组结果的合理性和稳定性。最终视频生成阶段将采用 DBSCAN + KMeans 兜底策略，以保证每帧至少显示 2 个行人组。

## 12. 全帧可视化与视频生成

根据作业要求，实验不仅需要输出聚类统计结果，还需要对每一帧聚类结果进行可视化，并最终形成视频。因此，本阶段进一步对 KMeans、DBSCAN、Agglomerative Clustering 和 GMM 四种 baseline 方法分别进行了全帧可视化。

每种方法均对 541 个 time_step 生成一张 PNG 图片，并使用 ffmpeg 合成为 MP4 视频。可视化中，不同颜色表示不同的聚类组，点旁数字表示行人 ID，坐标轴范围固定为 X∈[-1,16]、Y∈[-1,15]。

输出视频文件包括：

- outputs/a_baseline/full_visualization/kmeans_xy/video/kmeans_xy.mp4
- outputs/a_baseline/full_visualization/dbscan_xy/video/dbscan_xy.mp4
- outputs/a_baseline/full_visualization/agglomerative_xy/video/agglomerative_xy.mp4
- outputs/a_baseline/full_visualization/gmm_xy/video/gmm_xy.mp4

为避免逐帧聚类标签编号在不同帧之间随机变化导致颜色跳变，本文在可视化阶段引入 visual_group_id。该编号通过相邻帧中聚类组的行人 ID 重叠程度和组中心距离进行匹配，使连续帧中成员重叠较多、位置相近的组尽可能继承相同颜色。需要说明的是，该步骤仅用于视频显示时的颜色一致性，不改变原始聚类结果。

## 13. 第二阶段结论

A 类基础单帧空间聚类实验已经完成。实验结果表明，仅使用 X、Y 空间坐标即可得到基本的逐帧聚类结果。其中 KMeans、Agglomerative Clustering 和 GMM 由于指定了固定簇数，输出结果较稳定，但不能自动适应不同帧中的真实组数变化。DBSCAN 能够根据空间密度自动发现不同数量的行人组，更符合 Group Discovery 的任务目标，但存在一定噪声点，并且对参数较敏感。

因此，A 类方法适合作为基础 baseline。后续实验将进一步引入速度、滑动窗口速度、指数动量和运动方向等特征，分析运动信息是否能够提升聚类结果的合理性和连续帧稳定性。
