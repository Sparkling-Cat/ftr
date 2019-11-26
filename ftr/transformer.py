# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:40:20 2019

@author: Ivan
"""

import numpy as np
import pandas as pd
from .scalers import get_scaler



class FeatureTransformer():


    def __init__(self, scaler_names=None,
                 sep="\t", chunksize=10000, id_col_name="id_job"):
        if scaler_names is None:
            self.scaler_names = {"2": "z-score"}
        else:
            self.scaler_names = scaler_names
        self.sep = sep
        self.chunksize = chunksize
        self.id_col_name = id_col_name


    def _chunk_fit(self, df):
        df.set_index(self.id_col_name, inplace=True)
        for i, (col_name, col) in enumerate(df.iteritems()):
            #Check if mixed features happen
            col_expand = col.str.split(",", expand=True)
            feature_type = col_expand.iloc[:, 0].unique()
            try:
                same_type = np.array_equiv(feature_type, np.array(self.col_types[i]))
            except IndexError:
                if len(feature_type) == 1:
                    self.col_types.append(feature_type[0])
                    self.scaler_list.append(
                        [get_scaler(self.scaler_names[feature_type[0]]) \
                         for _ in range(1, len(col_expand.columns))]
                    )
                    same_type = True
                else:
                    same_type = False

            if not same_type:
                raise ValueError(
                    "Feature column {} has different feature type marks".format(i)
                    )
            for j in range(1, len(col_expand.columns)):
                self.scaler_list[i][j - 1].partial_fit(pd.to_numeric(col_expand.iloc[:, j]))
        return self


    def fit(self, file_name):
        self.col_types = []
        self.scaler_list = []
        with open(file_name) as f:
            for df in pd.read_csv(f, sep=self.sep, chunksize=self.chunksize):
                self._chunk_fit(df)
        return self


    def transform(self, file_in, file_out):
        with open(file_in) as f:
            col_features = []
            write_header = True
            write_mode = "w"
            for df in pd.read_csv(f, sep=self.sep, chunksize=self.chunksize):
                df.set_index(self.id_col_name, inplace=True)
                df_new = pd.DataFrame(None)
                for i, (col_name, col) in enumerate(df.iteritems()):
                    #Check if mixed features happen
                    col_expand = col.str.split(",", expand=True)
                    feature_type = col_expand.iloc[:, 0].unique()
                    try:
                        same_type = np.array_equiv(feature_type,
                                                   np.array(col_features[i]))
                    except IndexError:
                        if len(feature_type) == 1:
                            col_features.append(feature_type[0])
                            same_type = True
                        else:
                            same_type = False

                    if not same_type:
                        raise ValueError(
                            "Feature column {} has different feature type marks".format(i)
                            )
                    #Scaling works with the first appearence of certain feature type
                    #in the train file columns
                    feature_index = np.nonzero(
                        np.array(self.col_types) == col_features[i]
                        )[0][0]
                    col_expand_new = pd.DataFrame([
                        self.scaler_list[feature_index][j - 1].transform(
                            pd.to_numeric(col_expand.iloc[:, j])) \
                        for j in range(1, col_expand.shape[1])]
                        ).T
                    col_expand_new.columns = \
                        ["feature_{}_stand_{}".format(feature_type[0], j) \
                         for j in range(0, col_expand_new.shape[1])]
#                    col_expand_new.insert(0, self.id_col_name, df.index)
                    col_expand_new["max_feature_{}_index".format(feature_type[0])] = \
                        col_expand.iloc[:, 1:].astype(float).values.argmax(axis=1)
                    col_expand_new["max_feature_{}_abs_mean_diff".format(
                        feature_type[0])] = \
                        [abs(float(col_expand.iloc[
                            k,
                            col_expand_new["max_feature_{}_index".format(
                                feature_type[0])].iloc[k]\
                            + 1]) - \
                            self.scaler_list[feature_index]\
                            [col_expand_new["max_feature_{}_index".format(
                                feature_type[0])].iloc[k]].get_mean()
                            ) for k in range(col_expand_new.shape[0])]

                    df_new = pd.concat([df_new, col_expand_new], axis=1)

                df_new.insert(0, self.id_col_name, df.index)
                df_new.to_csv(file_out, sep="\t", header=write_header,
                              index=False, mode=write_mode)
                write_header = False
                write_mode = "a"
