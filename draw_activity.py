# -*- coding: utf-8 -*-
"""
distribution from ys.pkl
"""
import math
import argparse
import pickle
import matplotlib.pyplot as plt
from seaborn import set_context

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ys_pkl")
    parser.add_argument("target_name")
    parser.add_argument("activity_name")
    parser.add_argument("--width", "-w", type=float, default=1)
    parser.add_argument("--output", "-o", default="./activity.png")
    args = parser.parse_args()

    with open(args.ys_pkl, "rb") as ys_pkl_fp:
        ys = pickle.load(ys_pkl_fp)

    ys_first = int(math.floor(min(ys)))
    ys_last = int(math.ceil(max(ys)))
    N = int((ys_last - ys_first) / args.width)
    bins = [ys_first + i * args.width for i in range(N)]
    set_context("poster")
    plt.figure(figsize=(6, 4))
    plt.hist(ys, bins=bins)
    plt.xlabel(args.activity_name)
    plt.ylabel("count")
    plt.xlim(-50, 100)
    plt.yscale("log")
    plt.title(args.target_name)
    plt.tight_layout()
    plt.savefig(args.output)
