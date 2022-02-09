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
module="maf.dim30"
#klasses="EvalExample5 EvalExample6"

#for klass in $klasses; do
#  #  echo "launching python bashlauncher.py \"$module\" \"$gpu\""
#  python "bashlauncher.py" "$module.$klass" "$klass" "$gpu" &>"logs/$klass.log" &
#done


python "bashlauncher.py" "$module.Dim30aLargeGaps" "Dim30aLargeGaps" "$gpu" &>"logs/Dim30aLargeGaps.log" &
python "bashlauncher.py" "$module.Dim30bSmallGaps" "Dim30bSmallGaps" "$gpu" &>"logs/Dim30bSmallGaps.log" &
