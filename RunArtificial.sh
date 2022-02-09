#!/bin/bash
DIR="$(dirname "$0")"
cd "$DIR" || exit

. venv/bin/activate
export PYTHONPATH=$(pwd)

mkdir -p "logs"
gpu=0
if [ -n "$1" ]; then
  gpu=$1
fi
module="maf.dim2"
klasses="NF1D_1Bumps NF1D_2Bumps NF2D_1Bumps NF2D_2Bumps NF2D_10Bumps NF2D_1Rect NF2D_3Rect NF2D_4Rect SS1DMafExperiment SS2DMafExperiment ShowCase1D1 NF2D_Diag4 NF2D_Row3 NF2D_Row4  NF2D_RandomA NF2D_RandomB"
#klasses="NF1D_1Bumps NF1D_2Bumps NF2D_1Bumps"

for klass in $klasses; do
#  echo "launching python bashlauncher.py \"$module\" \"$gpu\""
  python "bashlauncher.py" "$module.$klass" "$klass" "$gpu" &> "logs/$klass.log"
done
