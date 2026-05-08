# Group Discovery on students003

本项目完成了基于 students003 行人轨迹数据的 Group Discovery 实验。

## 实验内容

- 数据格式检查
- A 类：基础单帧空间聚类
- B 类：运动特征增强
- C 类：时序关系建模
- D 类：参数分析与稳定性评价
- Final：DBSCAN + KMeans fallback 生成最终视频

## Final 方法

最终采用：

- 特征：X, Y, window_vx, window_vy
- 聚类：DBSCAN + KMeans fallback
- eps = 1.0
- min_samples = 2
- window_size = 5
- position_weight = 1.0
- motion_weight = 0.5

## 主要输出

- `report/main.tex`：LaTeX 实验报告源码
- `outputs/final/final_summary.csv`：最终方法统计结果
- `outputs/final/final_method_description.md`：最终方法说明


