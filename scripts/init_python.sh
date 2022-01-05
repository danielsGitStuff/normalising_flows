#!/bin/bash
DIR="$(dirname "$0")"
cd "$DIR" || exit
python_tgz='python39.src.tgz'
python_src_dir='python39.src'
python_url='https://www.python.org/ftp/python/3.9.9/Python-3.9.9.tgz'
python_md5='a2da2a456c078db131734ff62de10ed5'
target_root="$(pwd)/pyinstall"

# check if there is something to do
if [ -d "$target_root" ]; then
  echo "python binary dir '$target_root' already exists. skipping build..."
  exit
fi

# download source
if ! [ -f "$python_tgz" ]; then
  curl "$python_url" --output "$python_tgz"
fi

# check if source has the expected hash
hash=$(md5sum "$python_tgz")
hash=${hash:0:32}
if [ "$hash" != "$python_md5" ]; then
  echo "expected md5 '$python_md5' for file '$python_tgz' but got '$hash'"
  exit 
fi

# create local python runtime dir
mkdir -p "$target_root"
echo "will install python to: '$target_root'"

# extract python source
if ! [ -d "$python_src_dir" ]; then
    mkdir -p "$python_src_dir"
    tar zxvf "$python_tgz" --strip-components=1 --directory "$python_src_dir"
fi

# build that thing
cd $python_src_dir || exit
./configure -prefix="$target_root" # --enable-optimizations
make
make install

# clean up
cd "$DIR" || exit
rm -rf "$python_src_dir"

