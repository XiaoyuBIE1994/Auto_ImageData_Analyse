# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:10:00 2019

@author: xue
"""

import Intergration
import numpy as np

# configuration
data_color_range = [60, 160, 160, 210, 130, 200]
start_ratio = 0.5
vert_ratio_num = 0.2
vert_ratio_gap=0.8

ransac_params = {"max_iter": 10000, "mean_iter":10, \
                 "line1_gap":50, "line1_cut_range":0.2, "line1_k_min":-1, \
                 "line2_gap":5, "line2_cut_range":0.90, "line2_k_max":-10}

find_corner_params = {"color_range":[0, 100, 0, 100, 0, 100], \
                      "X_search_range":4, \
                      "y_search_range":4}

ocr_params = {"scale_w":2, "scale_h":2, "sharpen_lower":160, "sharpen_upper":255, \
              "kernel": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) }

find_mark_params = {"SHOW_MARK":False, "thr": 170, "search_range":10, \
                    "bb_size_X":(12,1), "jug_cond_X":5, \
                    "bb_size_Y":(1,12), "jug_cond_Y":5}

find_coor_params = {"SHOW_COOR":False, "bias_w_x":-20, "bias_h_x":8, "img_w_x":30, "img_h_x":14, \
                    "bias_w_y":-32, "bias_h_y":-7, "img_w_y":30, "img_h_y":14, "ocr_params":ocr_params}

find_peak_params = {"SHOW_PEAK":False, "scale_w":2, "scale_h":2, \
                    "bias_w_l":5, "bias_w_r":-50, "bias_h":2, "img_w":30, "img_h":30, \
                    "x_thr":12, "y_thr":50, "color_range":[0,180,0,180,0,180],\
                    "ocr_params":ocr_params}

SHOW_RANSAC_LINE = True
CHECK_RESULT = True
MODE_CHECK = False

file = r"E:\Faurecia\Treatment_Data\Image_Data_small\SB_NAFI&Plate_LGF20_Ep2_7_T1_5_001.PNG"

pt1, pt2 = Intergration.model_run (file, data_color_range, start_ratio, vert_ratio_num, vert_ratio_gap, ransac_params,\
                      find_corner_params, find_mark_params, find_coor_params,find_peak_params,\
                      MODE_CHECK, SHOW_RANSAC_LINE, CHECK_RESULT)