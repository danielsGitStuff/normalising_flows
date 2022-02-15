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
#klasses="EvalExample5 EvalExample6"

#for klass in $klasses; do
#  #  echo "launching python bashlauncher.py \"$module\" \"$gpu\""
#  python "bashlauncher.py" "$module.$klass" "$klass" "$gpu" &>"logs/$klass.log" &
#done

gpu=0

python "bashlauncher.py" "$module.Dim10aCenteredMVG" "Dim10aCenteredMVG" "$gpu" &>"logs/Dim10aCenteredMVG.log" &
gpu=1
nohup python "bashlauncher.py" "$module.Dim10bLargeGaps" "Dim10bLargeGaps" "$gpu" &>"logs/Dim10bLargeGaps.log" &
gpu=2
nohup python "bashlauncher.py" "$module.Dim10cSmallGaps" "Dim10cSmallGaps" "$gpu" &>"logs/Dim10cSmallGaps.log" &
#python "bashlauncher.py" "$module.EvalExample6" "EvalExample6" "2" &>"logs/EvalExample6.log" &

#nohup python "bashlauncher.py" "$module.EvalExample7" "EvalExample7" "1" &>"logs/EvalExample7.log" &
#nohup python "bashlauncher.py" "$module.EvalExample8" "EvalExample8" "1" &>"logs/EvalExample8.log" &

#python visualise/CachePrinter.py ".cache"


gpu=1
nohup python "bashlauncher.py" "$module.zDim10bLargeGapsTest" "zDim10bLargeGapsTest" "$gpu" &>"logs/zDim10bLargeGapsTest.log" &