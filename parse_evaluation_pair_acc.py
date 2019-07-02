# -*- coding: utf-8 -*-
"""
parse evaluation_pair_acc
* 1 input  -> output mean and sd
* 2 inputs -> output each mean and sd, and result of paired t-test
"""

import csv
import argparse
import numpy as np
from scipy.stats import ttest_rel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_csv_lst", nargs="+", help="base, proposed")
    args = parser.parse_args()

    if len(args.eval_csv_lst) != 1 and len(args.eval_csv_lst) != 2:
        exit("[Usage]: python parse_evaluation_pair.py eval_csv1 [eval_csv2]")

    if len(args.eval_csv_lst) == 1:
        eval_csv_name = args.eval_csv_lst[0]
        with open(eval_csv_name) as eval_csv_fp:
            reader = csv.reader(eval_csv_fp)
            next(reader)
            pair_acc_lst = [float(row[1]) for row in reader]
        print("mean {0:.3f}".format(np.mean(pair_acc_lst)))
        print("std {0:.3f}".format(np.std(pair_acc_lst)))
    else:
        with open(args.eval_csv_lst[0]) as eval_csv_fp_0:
            reader = csv.reader(eval_csv_fp_0)
            next(reader)
            pair_acc_1 = [float(row[1]) for row in reader]
        with open(args.eval_csv_lst[1]) as eval_csv_fp_1:
            reader = csv.reader(eval_csv_fp_1)
            next(reader)
            pair_acc_2 = [float(row[1]) for row in reader]
            t_stat, p_val = ttest_rel(pair_acc_1, pair_acc_2)
            # oneside test (half p-value)
            p_val /= 2
            print("1 mean {0:.3f} std {1:.3f}".format(
                np.mean(pair_acc_1), np.std(pair_acc_1)))
            print("2 mean {0:.3f} std {1:.3f}".format(
                np.mean(pair_acc_2), np.std(pair_acc_2)))
            print("p-value {0:.3f}".format(p_val))
