# -*- coding: utf-8 -*-
import csv
import pickle
import argparse
import numpy as np
from sklearn import datasets


def read_activity_from_svmlight(svmlight_name):
    """
    Active: 1
    Inactive: 0
    """
    activities = []
    with open(svmlight_name) as svmlight_fp:
        reader = csv.reader(svmlight_fp, delimiter=" ")
        activities = np.array([int(row[-1] == "Active") for row in reader])
    return activities

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("svmlight_name")
    parser.add_argument("--output_dir", "-o", default=".")
    args = parser.parse_args()

    np.random.seed(0)

    X, ys = datasets.load_svmlight_file(args.svmlight_name)
    activities = read_activity_from_svmlight(args.svmlight_name)
    print("Done loading svmlight file")

    X_pkl = args.output_dir + "/X.pkl"
    ys_pkl = args.output_dir + "/ys.pkl"
    activity_pkl = args.output_dir + "/activity.pkl"
    with open(X_pkl, "wb") as X_pkl_fp:
        pickle.dump(X, X_pkl_fp)
    with open(ys_pkl, "wb") as ys_pkl_fp:
        pickle.dump(ys, ys_pkl_fp)
    with open(activity_pkl, "wb") as activity_pkl_fp:
        pickle.dump(activities, activity_pkl_fp)
    print("Done pickle svmlight file")
