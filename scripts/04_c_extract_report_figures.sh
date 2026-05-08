#!/bin/bash
set -e

cd /root/autodl-tmp

mkdir -p outputs/c_temporal/report_figures

for method in c_temporal_agglomerative c_relation_graph_dt2p5_vt0p20 c_affinity_spectral; do
    mkdir -p outputs/c_temporal/report_figures/${method}

    for idx in 0000 0100 0250 0400 0540; do
        cp outputs/c_temporal/full_visualization/${method}/frames/frame_${idx}.png \
           outputs/c_temporal/report_figures/${method}/${method}_frame_${idx}.png
    done
done

echo "C-class report figures extracted."
