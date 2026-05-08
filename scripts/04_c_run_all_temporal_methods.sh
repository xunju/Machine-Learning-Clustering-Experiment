#!/bin/bash
set -e

cd /root/autodl-tmp
mkdir -p outputs/c_temporal

WINDOW_SIZE=5
BETA=0.7
N_CLUSTERS=3

echo "Running C1: temporal agglomerative..."
python scripts/04_c_temporal_relation_clustering.py \
  --method temporal_agglomerative \
  --window-size ${WINDOW_SIZE} \
  --beta ${BETA} \
  --n-clusters ${N_CLUSTERS} \
  | tee outputs/c_temporal/c_temporal_agglomerative.log

echo "Running C2: relation graph..."
python scripts/04_c_temporal_relation_clustering.py \
  --method relation_graph \
  --window-size ${WINDOW_SIZE} \
  --beta ${BETA} \
  --dist-threshold 1.2 \
  --vel-threshold 0.08 \
  | tee outputs/c_temporal/c_relation_graph.log

echo "Running C3: affinity spectral..."
python scripts/04_c_temporal_relation_clustering.py \
  --method affinity_spectral \
  --window-size ${WINDOW_SIZE} \
  --beta ${BETA} \
  --n-clusters ${N_CLUSTERS} \
  --sigma-d 2.0 \
  --sigma-v 0.1 \
  | tee outputs/c_temporal/c_affinity_spectral.log

echo "Stage 4 C-class temporal relation methods finished."
