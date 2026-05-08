#!/bin/bash
set -e

cd /root/autodl-tmp

mkdir -p outputs/b_motion/report_figures

for method in b_instant_velocity b_window_velocity b_momentum b_direction; do
    mkdir -p outputs/b_motion/report_figures/${method}

    cp outputs/b_motion/full_visualization/${method}/frames/frame_0000.png \
       outputs/b_motion/report_figures/${method}/${method}_frame_0000.png

    cp outputs/b_motion/full_visualization/${method}/frames/frame_0100.png \
       outputs/b_motion/report_figures/${method}/${method}_frame_0100.png

    cp outputs/b_motion/full_visualization/${method}/frames/frame_0250.png \
       outputs/b_motion/report_figures/${method}/${method}_frame_0250.png

    cp outputs/b_motion/full_visualization/${method}/frames/frame_0400.png \
       outputs/b_motion/report_figures/${method}/${method}_frame_0400.png

    cp outputs/b_motion/full_visualization/${method}/frames/frame_0540.png \
       outputs/b_motion/report_figures/${method}/${method}_frame_0540.png
done

echo "B-class report figures extracted."
