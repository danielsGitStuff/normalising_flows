#!/bin/bash
DIR="$(dirname "$0")"
cd "$DIR" || exit

(./RunArtificial.sh 0 && ./RunDry.sh 0) & ./RunMixlearn.sh 2 &