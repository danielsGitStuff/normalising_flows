#!/bin/bash
DIR="$(dirname "$0")"
cd "$DIR" || exit

. venv/bin/activate
export PYTHONPATH=$(pwd)
module="maf.dim1"

mkdir -p "logs"

klasses1=( "NF1D_1Bumps" "NF1D_2Bumps" )


./RunList.sh "$module" 1 "${klasses1[@]}" &