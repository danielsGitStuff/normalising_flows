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

python "bashlauncher.py" "maf.mixlearn.MixLearnExperimentMiniBooneClfVarRunner" "MixLearnExperimentMiniBooneClfVarRunner" 1 > "logs/MixLearnExperimentMiniBooneClfVarRunner.log" &
python "bashlauncher.py" "maf.mixlearn.MixLearnExperimentMiniBooneClfVarRunnerBalanced" "MixLearnExperimentMiniBooneClfVarRunnerBalanced" 2 > "logs/MixLearnExperimentMiniBooneClfVarRunnerBalanced.log"
