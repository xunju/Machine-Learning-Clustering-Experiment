#!/bin/bash
set -e

cd /root/autodl-tmp

mkdir -p outputs/a_baseline

echo "Running KMeans baseline..."
python scripts/02_a_baseline_spatial_clustering.py \
  --method kmeans \
  --n-clusters 3 \
  | tee outputs/a_baseline/kmeans_xy.log

echo "Running DBSCAN baseline..."
python scripts/02_a_baseline_spatial_clustering.py \
  --method dbscan \
  --eps 0.6 \
  --min-samples 2 \
  | tee outputs/a_baseline/dbscan_xy_eps0p6_min2.log

echo "Running Agglomerative baseline..."
python scripts/02_a_baseline_spatial_clustering.py \
  --method agglomerative \
  --n-clusters 3 \
  | tee outputs/a_baseline/agglomerative_xy.log

echo "Running GMM baseline..."
python scripts/02_a_baseline_spatial_clustering.py \
  --method gmm \
  --n-clusters 3 \
  | tee outputs/a_baseline/gmm_xy.log

echo "Summarizing A-class baseline results..."
python scripts/02_a_summarize_baseline.py

echo "Stage 2 A-class baseline finished."
