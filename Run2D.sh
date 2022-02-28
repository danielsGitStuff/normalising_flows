#!/bin/bash
DIR="$(dirname "$0")"
cd "$DIR" || exit

. venv/bin/activate
export PYTHONPATH=$(pwd)
module="maf.dim2"

mkdir -p "logs"

function run() {

  local gpu=$1
  shift
  local klasses=("$@")
  echo ${#klasses[@]}
  for i in "${!klasses[@]}"; do
    klass=${klasses[$i]}
    echo "launching python bashlauncher.py \"$module.$klass\" \"$klass\" \"$gpu\"&> \"logs/$klass.log\""
    python "bashlauncher.py" "$module.$klass" "$klass" "$gpu" &> "logs/$klass.log" &
  done
}

klasses1=( "NF2D_1Bumps" "NF2D_2Bumps" "NF2D_10Bumps" "NF2D_1Rect" "NF2D_3Rect"  )
klasses2=( "NF2D_4Rect" "SS1DMafExperiment" "SS2DMafExperiment" "ShowCase1D1" "NF2D_Diag4" "NF2D_Row3" "NF2D_Row4" "NF2D_4Connected1" )

# run klasses1 on gpu 1 and klasses2 on gpu 2
./RunList.sh "$module" 1 "${klasses1[@]}" &
./RunList.sh "$module" 2 "${klasses2[@]}"
