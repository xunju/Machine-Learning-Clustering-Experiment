#!/bin/bash
set -e

cd /root/autodl-tmp
mkdir -p outputs/b_motion

EPS=0.8
MIN_SAMPLES=2
WINDOW_SIZE=5
MOMENTUM_ALPHA=0.7
POSITION_WEIGHT=1.0
MOTION_WEIGHT=1.0

echo "Running B1: instant velocity..."
python scripts/03_b_motion_feature_clustering.py \
  --feature-mode instant_velocity \
  --eps ${EPS} \
  --min-samples ${MIN_SAMPLES} \
  --window-size ${WINDOW_SIZE} \
  --momentum-alpha ${MOMENTUM_ALPHA} \
  --position-weight ${POSITION_WEIGHT} \
  --motion-weight ${MOTION_WEIGHT} \
  | tee outputs/b_motion/b_instant_velocity.log

echo "Running B2: window velocity..."
python scripts/03_b_motion_feature_clustering.py \
  --feature-mode window_velocity \
  --eps ${EPS} \
  --min-samples ${MIN_SAMPLES} \
  --window-size ${WINDOW_SIZE} \
  --momentum-alpha ${MOMENTUM_ALPHA} \
  --position-weight ${POSITION_WEIGHT} \
  --motion-weight ${MOTION_WEIGHT} \
  | tee outputs/b_motion/b_window_velocity.log

echo "Running B3: momentum..."
python scripts/03_b_motion_feature_clustering.py \
  --feature-mode momentum \
  --eps ${EPS} \
  --min-samples ${MIN_SAMPLES} \
  --window-size ${WINDOW_SIZE} \
  --momentum-alpha ${MOMENTUM_ALPHA} \
  --position-weight ${POSITION_WEIGHT} \
  --motion-weight ${MOTION_WEIGHT} \
  | tee outputs/b_motion/b_momentum.log

echo "Running B4: direction..."
python scripts/03_b_motion_feature_clustering.py \
  --feature-mode direction \
  --eps ${EPS} \
  --min-samples ${MIN_SAMPLES} \
  --window-size ${WINDOW_SIZE} \
  --momentum-alpha ${MOMENTUM_ALPHA} \
  --position-weight ${POSITION_WEIGHT} \
  --motion-weight ${MOTION_WEIGHT} \
  | tee outputs/b_motion/b_direction.log

echo "Stage 3 B-class motion feature clustering finished."
