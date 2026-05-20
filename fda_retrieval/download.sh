#!/bin/bash

BASE_URL="https://data.eol.ucar.edu/pub/download/data/myase508366/"
OUTDIR="/lustre/desc1/scratch/myasears/M2HATS/FDA_datasets/ISS_radiosonde"

mkdir -p "$OUTDIR"

cd "$OUTDIR" || exit 1

# Download all linked .nc files from listing page
curl -s "$BASE_URL" \
| grep -oE 'href="[^"]+\.nc"' \
| sed 's/href="//;s/"//' \
| while read file; do
    echo "Downloading $file"
    curl -L -C - -O "$BASE_URL$file"
done