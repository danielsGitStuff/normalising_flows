#!/bin/bash
DIR="$(dirname "$0")"
cd "$DIR" || exit

. venv/bin/activate
export PYTHONPATH=$(pwd)

mkdir -p "logs"

module="maf.dim2"
klasses="NF1D_1Bumps NF1D_2Bumps NF2D_1Bumps NF2D_2Bumps NF2D_10Bumps NF2D_1Rect NF2D_3Rect NF2D_4Rect SS1DMafExperiment SS2DMafExperiment ShowCase1D1 NF2D_Diag4 NF2D_Row3 NF2D_Row4"
klasses=( "NF1D_1Bumps" "NF1D_2Bumps" "NF2D_1Bumps" "NF2D_2Bumps" "NF2D_10Bumps" "NF2D_1Rect" "NF2D_3Rect" "NF2D_4Rect" "SS1DMafExperiment" "SS2DMafExperiment" "ShowCase1D1" "NF2D_Diag4" "NF2D_Row3" "NF2D_Row4" )

#klasses=( "a" "bb" "ccc" "dddd" "eeeee")
echo ${#klasses[@]}
for i in "${!klasses[@]}"; do
  klass=${klasses[$i]}
  gpu=$(($i % 3))
  echo "launching python bashlauncher.py \"$module\" \"$klass\" \"$gpu\"&> \"logs/$klass.log\""
#  python "bashlauncher.py" "$module.$klass" "$klass" "$gpu" &> "logs/$klass.log"
done


#for klass in $klasses; do
#  #  echo "launching python bashlauncher.py \"$module\" \"$gpu\""
#  python "bashlauncher.py" "$module.$klass" "$klass" "$gpu" &>"logs/$klass.log" &
#done



#python "bashlauncher.py" "$module.EvalExample5" "EvalExample5" "2" &>"logs/EvalExample5.log" &
#python "bashlauncher.py" "$module.EvalExample6" "EvalExample6" "2" &>"logs/EvalExample6.log" &

#python "bashlauncher.py" "$module.EvalExample7" "EvalExample7" "1" &>"logs/EvalExample7.log" &
#python "bashlauncher.py" "$module.EvalExample8" "EvalExample8" "1" &>"logs/EvalExample8.log" &

#python visualise/CachePrinter.py ".cache"
