# -*- coding: utf-8 -*-
"""
[input]
* baseline_evaluation_file
* proposed_evaluation_file
[output]
* paired t test val
"""

import argparse
import ast
from scipy import stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_evaluation_file")
    parser.add_argument("proposed_evaluation_file")
    args = parser.parse_args()

    with open(args.baseline_evaluation_file) as base_fp:
        base_file_last_line = base_fp.readlines()[-1]
        base_file_last_line_d = ast.literal_eval(base_file_last_line)
        baseline_scores = base_file_last_line_d["scores"]

    with open(args.proposed_evaluation_file) as proposed_fp:
        proposed_file_last_line = proposed_fp.readlines()[-1]
        proposed_file_last_line_d = ast.literal_eval(proposed_file_last_line)
        proposed_scores = proposed_file_last_line_d["scores"]

    print("baseline")
    print(baseline_scores)
    print("proposed")
    print(proposed_scores)

    t_statistic, pvalue = stats.ttest_rel(
        baseline_scores, proposed_scores)
    print("p-value: {}".format(pvalue))
