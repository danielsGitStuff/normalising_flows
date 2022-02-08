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


python "bashlauncher.py" "$module.EvalExample3" "EvalExample3" "2" &>"logs/EvalExample3.log" &
python "bashlauncher.py" "$module.EvalExample3A" "EvalExample3A" "2" &>"logs/EvalExample3A.log" &
python "bashlauncher.py" "$module.EvalExample5" "EvalExample5" "2" &>"logs/EvalExample5.log" &
python "bashlauncher.py" "$module.EvalExample6" "EvalExample6" "2" &>"logs/EvalExample6.log" &

python "bashlauncher.py" "$module.EvalExample7" "EvalExample7" "2" &>"logs/EvalExample7.log" &
python "bashlauncher.py" "$module.EvalExample8" "EvalExample8" "2" &>"logs/EvalExample8.log" &

#python visualise/CachePrinter.py ".cache"
