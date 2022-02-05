#!/bin/bash
DIR="$(dirname "$0")"
cd "$DIR" || exit

#(./RunArtificial.sh 0 && ./RunVisualRandom.sh 0) & ./RunMixlearn.sh 2 &

./RunMixlearn.sh&
./RunArtificial.sh 0
./RunVisualRandom.sh 0
#,/RunMixlearn.sh 2