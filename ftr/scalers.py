# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:09:06 2019

@author: Ivan
"""

import numpy as np

VALID_SCALERS = ["z-score"]



class ZScoreScaler():


    def __init__(self):
        self.sum_value = 0
        self.sqsum_value = 0
        self.n = 0


    def get_mean(self):
        return self.sum_value / self.n


    def get_std(self):
        return ((self.sqsum_value / self.n - (self.sum_value / self.n) ** 2)) ** 0.5


    def partial_fit(self, col):
        self.sum_value += np.array(col).sum()
        self.sqsum_value += (np.array(col) * np.array(col)).sum()
        self.n += len(col)
        return self


    def transform(self, col):
        col_new = (col - self.get_mean()) / self.get_std()
        return col_new

    def transform_function(self):
        mean_value = self.get_mean()
        std_value = self.get_std()
        return lambda x: (x - mean_value) / std_value



def get_scaler(scaler_name="z-score"):
    if scaler_name == "z-score":
        scaler = ZScoreScaler()
    else:
        raise ValueError("Non-implemented scaler: {}".format(scaler_name))
    return scaler
