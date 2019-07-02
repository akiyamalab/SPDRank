# -*- coding: utf-8 -*-
import pickle
import argparse
import numpy as np
from sklearn import cross_validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("acitivity_pkl")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output", "-o", default="split.pkl")
    args = parser.parse_args()

    np.random.seed(0)

    with open(args.acitivity_pkl, "rb") as activity_pkl_fp:
        activities = pickle.load(activity_pkl_fp)
    skf = cross_validation.StratifiedKFold(activities,
                                           n_folds=args.n_folds, shuffle=True)

    with open(args.output, "wb") as out_fp:
        pickle.dump(skf, out_fp)
