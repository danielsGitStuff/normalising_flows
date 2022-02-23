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
module="maf.mixlearn"

gpu=0
nohup python "bashlauncher.py" "$module.MixLearnExperimentMiniBooneClfVarRunner" "MixLearnExperimentMiniBooneClfVarRunner" "$gpu" &>"logs/MixLearnExperimentMiniBooneClfVarRunner.log" &
gpu=1
nohup python "bashlauncher.py" "$module.MixLearnExperimentMiniBooneClfVarRunnerBalanced" "MixLearnExperimentMiniBooneClfVarRunnerBalanced" "$gpu" &>"logs/MixLearnExperimentMiniBooneClfVarRunnerBalanced.log" &
