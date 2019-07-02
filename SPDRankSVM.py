# -*- coding: utf-8 -*-
"""
SPDRankSVM: Stochastic Pairwise Descent RankSVM
* update parameters when the difference two scores larger than \cutoff
* ignore the ranking order of inactive (negative) samples
"""

import numpy as np
import scipy.sparse as sp


class SPDRankSVM(object):

    def __init__(self, X, lmd=1e-5):
        N, D = X.shape
        self.N = N
        self.w = np.zeros(D)
        self.lmd = lmd
        self.i_step = 0

    def _update(self, pair_x, pair_y):
        """
        pair_x: (X_first, X_second)
        pair_y: (y_first, y_second)
        y * <w, x> < 1  -> (1 - eta * lmd) * w + eta * y * x
        y * <w, x> >= 1 -> (1 - eta * lmd) * w
        """
        self.i_step += 1

        eta = 1. / (self.lmd * self.i_step)
        if pair_y[0] > pair_y[1]:
            y_diff = 1
        elif pair_y[1] < pair_y[0]:
            y_diff = -1
        else:
            return
        x_diff = pair_x[0] - pair_x[1]
        if isinstance(x_diff, sp.csr_matrix):
            x_diff = np.array(x_diff.todense()).reshape(-1)

        if y_diff * x_diff.dot(self.w) < 1:
            self.w = (1 - eta * self.lmd) * self.w +\
                (eta * y_diff * x_diff)
        else:
            self.w = (1 - eta * self.lmd) * self.w

    def fit_one_step(self, X, ys):
        """
        sampling pairs
        """
        first_ind = np.random.random_integers(0, self.N - 1)
        second_ind = np.random.random_integers(0, self.N - 1)
        pair_x = (X[first_ind], X[second_ind])
        pair_y = (ys[first_ind], ys[second_ind])
        self._update(pair_x, pair_y)

    def predict(self, X):
        return X.dot(self.w)


class IgnorePairSPDRankSVM(SPDRankSVM):

    def __init__(self, X, ys, lmd=1e-5,
                 cutoff=0.0, ignore_negative=False, activity_labels=None):
        if ignore_negative and activity_labels is None:
            raise ValueError(
                "If ignore_negative is True, "
                "activity_labels must be specified.")

        """
        activity_labels: list `Active=1, Inactive=0'
        Index
        """
        self.ignore_negative = ignore_negative
        if ignore_negative:
            self.active_index_lst, = activity_labels.nonzero()
            self.active_index_set = set(self.active_index_lst)

        """
        cutoff = 0 ; learning from whole active pairs
        cutoff = \infty ; Ignoring whole active pairs
        """
        self.cutoff = cutoff

        super().__init__(X, lmd=lmd)

    def fit_one_step(self, X, ys):
        """
        sampling pairs
        """
        is_train = False
        while not is_train:
            if self.ignore_negative:
                first_ind = np.random.choice(self.active_index_lst)
                second_ind = np.random.random_integers(0, self.N - 1)
                if second_ind not in self.active_index_set:
                    is_train = True
                else:
                    y_diff = abs(ys[first_ind] - ys[second_ind])
                    if y_diff > self.cutoff:
                        is_train = True
            else:
                first_ind = np.random.random_integers(0, self.N - 1)
                second_ind = np.random.random_integers(0, self.N - 1)
                y_diff = abs(ys[first_ind] - ys[second_ind])
                if y_diff > self.cutoff:
                    is_train = True

        pair_x = (X[first_ind], X[second_ind])
        pair_y = (ys[first_ind], ys[second_ind])
        self._update(pair_x, pair_y)
