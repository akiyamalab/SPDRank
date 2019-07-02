# -*- coding: utf-8 -*-
import numpy as np


def ndcg(y_true, y_pred, top=None, decay_func_name="log2",
         ignore_negative=False, activity_labels=None):
    """
    CAUTION: rel_1 + \sum_{i=2}^{k}rel_i/log2(i)
    """
    def decay_func(i):
        if decay_func_name == "log2":
            return np.log2(i)
        elif decay_func_name == "cg":  # cumulative gain
            return 1.
        else:
            raise ValueError("{} not exist!".format(decay_func_name))
    y_true_copied = np.copy(y_true)
    for i in range(len(y_true_copied)):
        # negative value -> 0
        if y_true_copied[i] < 0:
            y_true_copied[i] = 0
        # Ignoring negatives
        if ignore_negative and activity_labels[i] == 0:
            y_true_copied[i] = 0
    y_true_sorted = sorted(y_true_copied, reverse=True)
    if top is None:
        top = len(y_true_sorted)
    else:
        if top > len(y_true_sorted):
            raise ValueError("top: {} > N: {}".format(top, len(y_true_sorted)))

    ideal_dcg = y_true_sorted[0]
    for i in range(1, top):
        ideal_dcg += y_true_sorted[i] / decay_func(i + 1)
    # sort
    argsort_indices = np.argsort(y_pred)[::-1]
    dcg = y_true_copied[argsort_indices[0]]
    for i in range(1, top):
        dcg += y_true_copied[argsort_indices[i]] / decay_func(i + 1)
    ndcg = dcg / ideal_dcg
    return ndcg


def pairwise_accuracy(y_true, y_pred,
                      cutoff=0.0, ignore_negative=False, activity_labels=None,
                      top=None):
    if ignore_negative and activity_labels is None:
        raise ValueError(
            "If ignore_negative is True, activity_labels must be specified.")
    N = len(y_true)
    if top is None:
        top = N
    else:
        if top > N:
            raise ValueError(
                "top: {} > N: {}".format(top, N))
    ideal_score = 0  
    actual_score = 0 

    argsort_index = np.argsort(y_pred)[::-1]
    y_pred_sort = y_pred[argsort_index]
    y_true_pred_sort = y_true[argsort_index]

    for c1 in range(top):
        for c2 in range(c1 + 1, N):
            """
            Ignored pairs
            * (ignore_negative == True) && (neg, neg)
            * diff(y_true) <= cutoff
            """
            is_ignore = False
            # pos -> 1, neg -> 0
            if ignore_negative and \
                    (activity_labels[c1] == 0 and activity_labels[c2] == 0):
                is_ignore = True
            true_diff = y_true_pred_sort[c1] - y_true_pred_sort[c2]
            if abs(true_diff) <= cutoff:
                is_ignore = True
            if is_ignore:
                continue

            """
            (y1 > y2 && pred1 > pred2) || (y1 < y2 && pred1 < pred2) ; correct prediction
             --> (y1 - y2) * (pred1 - pred2) > 0  ; correct prediction
            """
            pred_diff = y_pred_sort[c1] - y_pred_sort[c2]
            if true_diff * pred_diff > 0:
                actual_score += 1
            ideal_score += 1
    return actual_score / ideal_score
