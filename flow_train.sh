#!/bin/zsh

cd ~/workspace/SPDRank/src

if [ $# -ne 1 ]; then
    echo "[Usage]: flow_train.sh target_name"
    exit 1
fi

target=$1
sm=100000
mi=5000
cutoff_vals=(0 1 2 3 4 5 6 7 8 9 10)

function output_flow_file () {
    cutoff=$1
    ignore_negative=$2

    project_name=train_cutoff_$cutoff\_in_$ignore_negative
    flow_file=../temp/flow_$target\_$project_name.sh
    project_dir=../project/$target/$project_name
    pred_dir=$project_dir/pred
    mkdir $project_dir
    mkdir $pred_dir
    data_dir=../project/$target/data
    X_pkl=$data_dir/X.pkl
    ys_pkl=$data_dir/ys.pkl
    split_pkl=$data_dir/split.pkl
    activity_pkl=$data_dir/activity.pkl
    train_log=$project_dir/train_log.txt
    if [ $ignore_negative = true ]; then
        train_ignore_neg_pair="-ti"
    fi
    cat << EOS > $flow_file
#!/bin/zsh
cd ~/workspace/SPDRank/src
python train_cv.py $X_pkl $ys_pkl $split_pkl $activity_pkl $train_ignore_neg_pair -tr_c $cutoff -o $pred_dir -sm $sm -mi $mi > $train_log
EOS
    chmod +x $flow_file
    t2subV $flow_file
}


# ignore similar pair
for cutoff_val in $cutoff_vals; do
    output_flow_file $cutoff_val false &
done
# ignore similar pair and negative pair
for cutoff_val in $cutoff_vals; do
    output_flow_file $cutoff_val true &
done
wait
