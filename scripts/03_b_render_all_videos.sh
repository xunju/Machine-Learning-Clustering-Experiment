#!/bin/bash
set -e

cd /root/autodl-tmp

OUTPUT_ROOT="outputs/b_motion/full_visualization"
mkdir -p ${OUTPUT_ROOT}

echo "Rendering instant velocity video..."
python scripts/02_a_render_full_video.py \
  --cluster-file outputs/b_motion/clusters_b_instant_velocity_eps0p8_min2_w5_a0p7_pw1p0_mw1p0.csv \
  --method-name b_instant_velocity \
  --output-root ${OUTPUT_ROOT} \
  --fps 10 \
  --match-threshold 0.25

echo "Rendering window velocity video..."
python scripts/02_a_render_full_video.py \
  --cluster-file outputs/b_motion/clusters_b_window_velocity_eps0p8_min2_w5_a0p7_pw1p0_mw1p0.csv \
  --method-name b_window_velocity \
  --output-root ${OUTPUT_ROOT} \
  --fps 10 \
  --match-threshold 0.25

echo "Rendering momentum video..."
python scripts/02_a_render_full_video.py \
  --cluster-file outputs/b_motion/clusters_b_momentum_eps0p8_min2_w5_a0p7_pw1p0_mw1p0.csv \
  --method-name b_momentum \
  --output-root ${OUTPUT_ROOT} \
  --fps 10 \
  --match-threshold 0.25

echo "Rendering direction video..."
python scripts/02_a_render_full_video.py \
  --cluster-file outputs/b_motion/clusters_b_direction_eps0p8_min2_w5_a0p7_pw1p0_mw1p0.csv \
  --method-name b_direction \
  --output-root ${OUTPUT_ROOT} \
  --fps 10 \
  --match-threshold 0.25

echo "All B-class full videos finished."
