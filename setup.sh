#!/bin/zsh

cd ~/workspace/SPDRank/src

if [ $# -ne 1 ]; then
    echo "[Usage]: setup.sh target_name"
    exit 1
fi

target=$1
mkdir ../project/$target
data_dir=../project/$target/data
mkdir $data_dir
svmlight_file=../data/$target.svmlight
python data_serialize.py $svmlight_file -o $data_dir
activity_pkl=$data_dir/activity.pkl
python make_split_index.py $activity_pkl -o $data_dir/split.pkl
