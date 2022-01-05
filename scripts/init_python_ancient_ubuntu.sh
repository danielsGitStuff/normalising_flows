#!/bin/bash
DIR="$(dirname "$0")"
cd "$DIR" || exit
python_tgz='pyinstall.tar.xz'
target_root="$(pwd)/pyinstall"

# check if there is something to do
if [ -d "$target_root" ]; then
  echo "python binary dir '$target_root' already exists. exiting..."
  exit
fi

tar xvf "$python_tgz"



