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
module="maf.dry_examples"
klasses="EvalExample1 EvalExample2 EvalExample3 EvalExample4"

for klass in $klasses; do
#  echo "launching python bashlauncher.py \"$module\" \"$gpu\""
  python "bashlauncher.py" "$module.$klass" "$klass" "$gpu" &> "logs/$klass.log"
done