#!/bin/bash
set -e

cd /root/autodl-tmp

OUTPUT_ROOT="outputs/a_baseline/full_visualization"
mkdir -p ${OUTPUT_ROOT}

echo "Rendering KMeans full video..."
python scripts/02_a_render_full_video.py \
  --cluster-file outputs/a_baseline/clusters_kmeans_xy_k3_eps0p6_min2.csv \
  --method-name kmeans_xy \
  --output-root ${OUTPUT_ROOT} \
  --fps 10

echo "Rendering DBSCAN full video..."
python scripts/02_a_render_full_video.py \
  --cluster-file outputs/a_baseline/clusters_dbscan_xy_k3_eps0p6_min2.csv \
  --method-name dbscan_xy \
  --output-root ${OUTPUT_ROOT} \
  --fps 10

echo "Rendering Agglomerative full video..."
python scripts/02_a_render_full_video.py \
  --cluster-file outputs/a_baseline/clusters_agglomerative_xy_k3_eps0p6_min2.csv \
  --method-name agglomerative_xy \
  --output-root ${OUTPUT_ROOT} \
  --fps 10

echo "Rendering GMM full video..."
python scripts/02_a_render_full_video.py \
  --cluster-file outputs/a_baseline/clusters_gmm_xy_k3_eps0p6_min2.csv \
  --method-name gmm_xy \
  --output-root ${OUTPUT_ROOT} \
  --fps 10

echo "All A-class baseline full videos finished."
