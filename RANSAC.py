# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:01:37 2019

@author: xue
"""

import numpy as np
from sklearn import linear_model 

class line_ransac:
    def __init__ (self, X, y, ransac_params):
        self.X = X
        self.y = y
        self.max_iter = ransac_params["max_iter"]
        self.mean_iter = ransac_params["mean_iter"]
        self.l1_gap = ransac_params["line1_gap"]
        self.l2_gap = ransac_params["line2_gap"]
        self.l1_cut_range = ransac_params["line1_cut_range"]
        self.l2_cut_range = ransac_params["line2_cut_range"]
        self.l1_k_min = ransac_params["line1_k_min"]
        self.l1_k_max = ransac_params["line1_k_max"]
        self.l2_k_max = ransac_params["line2_k_max"]
    def valid_data_line1(self):
        def valid_data(X, y):
            if (np.max(X) - np.min(X)) < self.l1_gap :
                return True
            else:
                return False
        return valid_data

    def valid_data_line2(self):
        def valid_data(X, y):
            if (np.max(X) - np.min(X)) < self.l2_gap :
                return True
            else:
                return False
            return valid_data
        
    def valid_model_line1(self):
        def valid_model(model, X, y):
            k = model.coef_
            if self.l1_k_max > k > self.l1_k_min:
                return True
            else:
                return False
        return valid_model
    
    def valid_model_line2(self):
        def valid_model(model, X, y):
            k = model.coef_
            if k < self.l2_k_max:
                return True
            else:
                return False
        return valid_model
    
    def find_line_ransac(self, cut_range = 0, min_samp = None, res_thr = None, valid_data=None, valid_model=None):
        ransac = linear_model.RANSACRegressor(max_trials=self.max_iter,min_samples=min_samp, residual_threshold=res_thr, \
                                    is_data_valid=valid_data, is_model_valid=valid_model)
        thr = (1-cut_range)*np.min(self.X) + cut_range*np.max(self.X)
        index = (self.X).index(int(thr))
        X = (self.X).copy()
        y = (self.y).copy()
        del X[:index]
        del y[:index]
        X = np.array(X)[:,np.newaxis]
        y = np.array(y)[:,np.newaxis]
        ransac.fit(X,y)
        k = ransac.estimator_.coef_
        b = ransac.estimator_.intercept_
        return k, b
    
    def line_extraction_ransac(self):

        k1_list = []
        b1_list = []
        k2_list = []
        b2_list = []
        for j in range(self.mean_iter):
            k1_temp, b1_temp = self.find_line_ransac(cut_range = self.l1_cut_range, \
                                                     valid_data=self.valid_data_line1(), \
                                                     valid_model=self.valid_model_line1())
            k2_temp, b2_temp = self.find_line_ransac(cut_range = self.l2_cut_range, \
                                                     valid_data=self.valid_data_line2(), \
                                                     valid_model=self.valid_model_line2())
            k1_list.append(k1_temp)
            b1_list.append(b1_temp)
            k2_list.append(k2_temp)
            b2_list.append(b2_temp)
            
        k1 = np.mean(k1_list)
        b1 = np.median(b1_list)
        
        b2 = np.max(b2_list)
#         k2 = np.mean(k2_list)
        k2 = k2_list[b2_list.index(b2)]
        x_cross = - (b1 - b2) / (k1 - k2)
        y_cross = k1 * x_cross + b1
        res = {}
        res["k1_list"] = k1_list
        res["b1_list"] = b1_list
        res["k2_list"] = k2_list
        res["b2_list"] = b2_list
        res["k1"] = k1
        res["b1"] = b1
        res["k2"] = k2
        res["b2"] = b2
        res["x_cross"] = x_cross
        res["y_cross"] = y_cross
        return res