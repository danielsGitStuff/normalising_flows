#!/bin/bash
DIR="$(dirname "$0")"
cd "$DIR" || exit

. venv/bin/activate
export PYTHONPATH=$(pwd)
module="maf.dim2"
module=$1
gpu=$2
shift
shift
klasses=("$@")
mkdir -p "logs"
echo ${#klasses[@]}
for i in "${!klasses[@]}"; do
  klass=${klasses[$i]}
  echo "launching python bashlauncher.py \"$module.$klass\" \"$klass\" \"$gpu\"> \"logs/$klass.log\""
  python "bashlauncher.py" "$module.$klass" "$klass" "$gpu" > "logs/$klass.log"
done


