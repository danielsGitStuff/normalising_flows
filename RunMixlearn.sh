#!/bin/bash
DIR="$(dirname "$0")"
cd "$DIR" || exit

. venv/bin/activate
export PYTHONPATH=$(pwd)

mkdir -p "logs"

python "bashlauncher.py" "maf.mixlearn.dl3.MinibooneDL3" "MinibooneDL3" 1 &> "logs/MinibooneDL3.log"
(python "bashlauncher.py" "maf.mixlearn.MixLearnExperimentMiniBooneClfVarRunner" "MixLearnExperimentMiniBooneClfVarRunner" 1 > "logs/MixLearnExperimentMiniBooneClfVarRunner.log" &
python "bashlauncher.py" "maf.mixlearn.MixLearnExperimentMiniBooneClfVarRunnerBalanced" "MixLearnExperimentMiniBooneClfVarRunnerBalanced" 2 > "logs/MixLearnExperimentMiniBooneClfVarRunnerBalanced.log")
