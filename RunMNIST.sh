#!/bin/bash
DIR="$(dirname "$0")"
cd "$DIR" || exit

. venv/bin/activate
PYTHONPATH=$(pwd) python maf/examples/MNISTRun.py