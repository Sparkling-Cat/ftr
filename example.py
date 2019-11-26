# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:47:08 2019

@author: Ivan
"""

from ftr import FeatureTransformer

tt = FeatureTransformer({"2": "z-score"})
tt.fit("test.tsv")
tt.transform("test.tsv", "test_proc.tsv")
