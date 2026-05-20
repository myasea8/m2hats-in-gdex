#!/bin/bash

RAW="/lustre/desc1/scratch/myasears/M2HATS/FDA_datasets/surface_tiltcor_5min_raw"
OUT="/lustre/desc1/scratch/myasears/M2HATS/FDA_datasets/surface_tiltcor_5min"

mkdir -p "$OUT"

for f in "$RAW"/*.tar.gz; do
    echo "Extracting $(basename "$f")"
    tar -xzf "$f" -C "$OUT"
done