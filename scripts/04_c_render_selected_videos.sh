#!/bin/bash
set -e

cd /root/autodl-tmp

OUTPUT_ROOT="outputs/c_temporal/full_visualization"
mkdir -p ${OUTPUT_ROOT}

echo "Rendering C1 temporal agglomerative video..."
python scripts/02_a_render_full_video.py \
  --cluster-file outputs/c_temporal/clusters_c_temporal_agglomerative_w5_b0p7_k3_dt1p2_vt0p08_sd2p0_sv0p1.csv \
  --method-name c_temporal_agglomerative \
  --output-root ${OUTPUT_ROOT} \
  --fps 10 \
  --match-threshold 0.25

echo "Rendering C2 relation graph selected video..."
python scripts/02_a_render_full_video.py \
  --cluster-file outputs/c_temporal/clusters_c_relation_graph_w5_b0p7_k3_dt2p5_vt0p2_sd2p0_sv0p1.csv \
  --method-name c_relation_graph_dt2p5_vt0p20 \
  --output-root ${OUTPUT_ROOT} \
  --fps 10 \
  --match-threshold 0.25

echo "Rendering C3 affinity spectral video..."
python scripts/02_a_render_full_video.py \
  --cluster-file outputs/c_temporal/clusters_c_affinity_spectral_w5_b0p7_k3_dt1p2_vt0p08_sd2p0_sv0p1.csv \
  --method-name c_affinity_spectral \
  --output-root ${OUTPUT_ROOT} \
  --fps 10 \
  --match-threshold 0.25

echo "Selected C-class videos finished."
