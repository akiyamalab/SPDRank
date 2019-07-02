#!/bin/zsh

cd ~/workspace/SPDRank/src

if [ $# -ne 1 ]; then
    echo "[Usage]: flow_eval.sh target_name"
    exit 1
fi

target=$1
score_funcs=(ndcg ncg)
top=100
cutoff_vals=(0 1 2 3 4 5 6 7 8 9 10)

function output_flow_file () {
    cutoff=$1
    ignore_negative=$2
    score_func=$3

    project_name=train_cutoff_$cutoff\_in_$ignore_negative
    flow_file=../temp/flow_eval_$target\_$project_name\_$score_func\_$top.sh
    project_dir=../project/$target/$project_name
    pred_dir=$project_dir/pred
    data_dir=../project/$target/data
    ys_pkl=$data_dir/ys.pkl
    split_pkl=$data_dir/split.pkl
    activity_pkl=$data_dir/activity.pkl
    evaluation_log=$project_dir/eval_$score_func\_top\_$top.txt
    cat << EOS > $flow_file
#!/bin/zsh
cd ~/workspace/SPDRank/src
python eval_cv.py $pred_dir $ys_pkl $split_pkl $activity_pkl $score_func --top $top > $evaluation_log
EOS
    chmod +x $flow_file
    t2subV $flow_file
}

for score_func in $score_funcs; do
    for cutoff_val in $cutoff_vals; do
        output_flow_file $cutoff_val false $score_func &
        output_flow_file $cutoff_val true $score_func &
    done
done
wait
