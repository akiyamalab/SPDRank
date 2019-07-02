# -*- coding: utf-8 -*-
"""
[input]
X.pkl
ys.pkl
split.pkl
[output]
prediction results of test data
"""

import subprocess
import pickle
import argparse
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from SPDRankSVM import IgnorePairSPDRankSVM


def train_by_split(train_index, test_index, i_fold, lmd):
    print("start i_fold: {} lmd: {}".format(i_fold, lmd))
    X_train = X[train_index]
    X_test = X[test_index]
    activities_train = activities[train_index]
    ys_train = ys[train_index]
    model = IgnorePairSPDRankSVM(X_train, ys_train,
                                 lmd=lmd,
                                 cutoff=args.train_cutoff,
                                 ignore_negative=args.train_ignore_neg_pair,
                                 activity_labels=activities_train)
    for i_step in range(1, args.step_max + 1):
        model.fit_one_step(X_train, ys_train)
        if i_step % args.monitor_interval == 0:
            ys_test_pred = model.predict(X_test)
            out_pred_name = args.output_pred_dir + \
                "/lmd_{}_step_{}/fold_{}.csv".format(lmd, i_step, i_fold)
            np.savetxt(out_pred_name, ys_test_pred, delimiter=",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("X_pkl")
    parser.add_argument("ys_pkl")
    parser.add_argument("split_pkl")
    parser.add_argument("activity_pkl")
    parser.add_argument("--train_ignore_neg_pair", "-ti", action="store_true")
    parser.add_argument("--train_cutoff", "-tr_c", type=float, default=0.0)
    parser.add_argument("--log_lmd_min", type=int, default=-5)
    parser.add_argument("--log_lmd_max", type=int, default=2)
    parser.add_argument("--step_max", "-sm", type=int, default=100000)
    parser.add_argument("--monitor_interval", "-mi", type=int, default=5000)
    parser.add_argument("--output_pred_dir", "-o", default=".")
    args = parser.parse_args()

    np.random.seed(0)

    with open(args.X_pkl, "rb") as X_pkl_fp:
        X = pickle.load(X_pkl_fp)
    with open(args.ys_pkl, "rb") as ys_pkl_fp:
        ys = pickle.load(ys_pkl_fp)
    with open(args.split_pkl, "rb") as split_pkl_fp:
        skf = pickle.load(split_pkl_fp)
    with open(args.activity_pkl, "rb") as activity_pkl_fp:
        activities = pickle.load(activity_pkl_fp)

    print("Done loading pickle files")

    lmds = [10 ** log_lmd
            for log_lmd in range(args.log_lmd_min, args.log_lmd_max + 1)]

    for lmd in lmds:
        for i_step in range(args.monitor_interval,
                            args.step_max + 1,
                            args.monitor_interval):
            dir_name = args.output_pred_dir + "/lmd_{}_step_{}".format(
                lmd, i_step)
            subprocess.call(["mkdir", dir_name])

    """
    parallelization
    """
    results_of_cv = Parallel(n_jobs=-1, verbose=5)(
        delayed(train_by_split)(train_index, test_index, i_fold, lmd)
        for i_fold, (train_index, test_index) in enumerate(skf)
        for lmd in lmds
    )
