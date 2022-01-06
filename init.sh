#!/bin/bash
DIR="$(dirname "$0")"
cd "$DIR" || exit

if [ -d "venv" ]; then
  echo "venv folder exists. nothing to do"
  exit
fi

# build a local python runtime
./init_python.sh

# create venv using local python runtime
if ! [ -d 'venv' ]; then
   pyinstall/bin/python3 -m venv venv
fi
source venv/bin/activate

# install dependencies
pip install -r requirements.txt