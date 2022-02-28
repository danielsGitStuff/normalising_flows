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
module="maf.dim10"

gpu=0
nohup python "bashlauncher.py" "$module.Dim10bLargeGaps" "Dim10bLargeGaps" "$gpu" &>"logs/Dim10bLargeGaps.log" &
gpu=1
nohup python "bashlauncher.py" "$module.Dim10bSmallGaps" "Dim10bSmallGaps" "$gpu" &>"logs/Dim10bSmallGaps.log" &
gpu=2
nohup python "bashlauncher.py" "$module.Dim10bMediumGaps" "Dim10bMediumGaps" "$gpu" &>"logs/Dim10bMediumGaps.log" &