# -*- coding: utf-8 -*-
import argparse
import ast

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir")
    parser.add_argument("score_func")
    args = parser.parse_args()

    cutoff_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ignore_negative = ["false", "true"]
    for cutoff_val in cutoff_vals:
        for ign in ignore_negative:
            dir_name = args.project_dir + "/train_cutoff_{}_in_{}".format(
                cutoff_val, ign)
            eval_file = dir_name + "/eval_{}.txt".format(args.score_func)
            with open(eval_file) as eval_fp:
                last_line = eval_fp.readlines()[-1]
                last_line_d = ast.literal_eval(last_line)
                if ign == "false":
                    false_val = last_line_d["mean"]
                elif ign == "true":
                    true_val = last_line_d["mean"]
        print("{0:.3f} {1:.3f}".format(false_val, true_val))
