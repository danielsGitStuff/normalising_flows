#!/bin/bash
DIR="$(dirname "$0")"
cd "$DIR" || exit

. venv/bin/activate
export PYTHONPATH=$(pwd)
python maf/mixlearn/MixLearnExperimentMiniBooneRunDSizeVar.py