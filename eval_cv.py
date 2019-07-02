# -*- coding: utf-8 -*-
"""
[input]
project_dir
[output]
optimal hyperparameters, score each fold, avg, sd
<eval_func>.csv
"""

import argparse
import glob
import os
import pickle
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from Evaluation import ndcg


def eval_by_hy_parm(hy_parm_dir):
    dir_name = os.path.basename(hy_parm_dir)
    lmd = dir_name.split("_")[1]
    step = dir_name.split("_")[-1]
    score_dict = {"lmd": lmd, "step": step, "scores": []}
    pred_files = sorted(glob.glob(hy_parm_dir + "/*"))
    for pred_file, (_, test_index) in zip(pred_files, skf):
        ys_pred = np.loadtxt(pred_file)
        ys_test = ys[test_index]
        activities_test = activities[test_index]
        score = score_func(ys_test, ys_pred, activities_test)
        score_dict["scores"].append(score)
    score_dict["mean"] = np.mean(score_dict["scores"])
    score_dict["std"] = np.std(score_dict["scores"])
    return score_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_dir")
    parser.add_argument("ys_pkl")
    parser.add_argument("split_pkl")
    parser.add_argument("activity_pkl")
    parser.add_argument("eval_func", choices=["ndcg", "ncg"])
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--ignore_negative", default=True)
    args = parser.parse_args()

    with open(args.ys_pkl, "rb") as ys_pkl_fp:
        ys = pickle.load(ys_pkl_fp)
    with open(args.split_pkl, "rb") as split_pkl_fp:
        skf = pickle.load(split_pkl_fp)
    with open(args.activity_pkl, "rb") as activity_pkl_fp:
        activities = pickle.load(activity_pkl_fp)

    """
    evaluation function
    """
    def score_func(y_true, y_pred, activities):
        if args.eval_func == "ndcg":
            return ndcg(y_true, y_pred,
                        top=args.top,
                        decay_func_name="log2",
                        ignore_negative=args.ignore_negative,
                        activity_labels=activities,
                        )
        elif args.eval_func == "ncg":
            return ndcg(y_true, y_pred,
                        top=args.top,
                        decay_func_name="cg",
                        ignore_negative=args.ignore_negative,
                        activity_labels=activities
                        )
        else:
            raise ValueError("{} not exist!".format(args.eval_func))

    """
    parallelization
    """
    hy_param_dirs = glob.glob(args.pred_dir + "/*")
    score_dicts = []
    score_dicts = Parallel(n_jobs=-1, verbose=5)(
        delayed(eval_by_hy_parm)(hy_param_dir)
        for hy_param_dir in hy_param_dirs
    )
    score_dicts.sort(key=lambda d: d["mean"], reverse=True)
    print(score_dicts)
    print(score_dicts[0])
