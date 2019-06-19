# -*- coding: utf-8 -*-
"""
Created on Sun May 26 18:58:32 2019

@author: xue
"""

import cv2
import numpy as np
from collections import Counter


# load image by opencv, transfer bgr to rgb
def load_cv_image(img_path):
    cv_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    np_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return np_img

# rgb to gray; the relationn between rgb and grey parameters
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Extract data points by color
def color_filter(img, color_range):
    #exclude input errors
    assert len(img.shape) == 3
    assert len(color_range) == 6
    h, w, _ = img.shape
    X = []
    y = []

    for i in range(h):
        for j in range(w):
            r = img[i,j,0]
            g = img[i,j,1]
            b = img[i,j,2]
            if color_range[0] <= r <= color_range[1] \
            and color_range[2] <= g <= color_range[3] \
            and color_range[4] <= b <= color_range[5]:
                X.append(j)
                y.append(i)
            else:
                pass
    return X, y

# find 4 corners of frame in the figure
def find_corner(img, find_corner_params):
    color_range = find_corner_params["color_range"]
    X_num = find_corner_params["X_search_range"]
    y_num = find_corner_params["y_search_range"]
    dic_corners = {}
    X, y = color_filter(img, color_range)
    X_count = Counter(X).most_common(X_num)
    y_count = Counter(y).most_common(y_num)
    X_list = []
    y_list = []
    for i in range(X_num):
        X_list.append(X_count[i][0])
    for j in range(y_num):
        y_list.append(y_count[j][0])
        
    x_left = np.min([x for x in X_list if x < np.mean(X_list)])
    x_right = np.max([x for x in X_list if x > np.mean(X_list)])
    y_up = np.min([y for y in y_list if y < np.mean(y_list)])
    y_down = np.max([y for y in y_list if y > np.mean(y_list)])
    dic_corners["left_down"] = (x_left, y_down)
    dic_corners["left_up"] = (x_left, y_up)
    dic_corners["right_down"] = (x_right, y_down)
    dic_corners["right_up"] = (x_right, y_up)
    dic_corners["plt_w"] = x_right - x_left
    dic_corners["plt_h"] = y_down - y_up
    return dic_corners


#r=g=b, keep, otherwise turm into white
def color_erase(img):
    assert len(img.shape) == 3
    img_new = img.copy()
    h, w, _ = img.shape
    for i in range(h):
        for j in range(w):
            r = img[i,j,0]
            g = img[i,j,1]
            b = img[i,j,2]
            if r == g and g == b:
                pass
            else:
                img_new[i,j,:] = np.array([255,255,255], dtype=np.uint8)
    return img_new
                
# sort X and y in order of X, arranging data according to x axis
def sort_X(X, y):
    X_sort = sorted(X)
    y_sort = [y for _,y in sorted(zip(X,y))]
    return X_sort, y_sort

# delete start part
def del_start(X_orig, y_orig, x_thr=300):
    index = X_orig.index(x_thr+1)
    X_new = X_orig.copy()
    y_new = y_orig.copy()
    del X_new[:index]
    del y_new[:index]
    return X_new, y_new

# delete vertical part, delete when more than vert_thr points share the same x value
def del_vert(X_orig, y_orig, vert_thr_num=50, vert_thr_gap = 100):
    x_temp = []
    y_temp = []
    del_part = []
    x_del = []
    y_del = []
    
    for i in range(len(X_orig)):
        if len(x_temp) == 0:
            x_start = i
            x_temp.append(X_orig[i])
            y_temp.append(y_orig[i])
        else:
            if X_orig[i] != x_temp[0]:
                num_x = i - x_start
                if num_x > vert_thr_num:
                    del_part.append((x_start, i))
                elif len(y_temp) > 0 and np.max(y_temp) - np.min(y_temp) > vert_thr_gap:
                    m = y_temp.index(min(y_temp))
                    del_part.append((x_start, x_start + m))
                    del_part.append((x_start + m + 1, i))
                y_temp = []
                x_temp = []
                x_start = i
                x_temp.append(X_orig[i])
                y_temp.append(y_orig[i])
            else:
                y_temp.append(y_orig[i])
                
    X_new = X_orig.copy()
    y_new = y_orig.copy()
    for delete in del_part[::-1]:
        x_del = x_del + X_new[delete[0] : delete[1]]
        y_del = y_del + y_new[delete[0] : delete[1]] 
        del X_new[delete[0] : delete[1]]
        del y_new[delete[0] : delete[1]]  
    return X_new, y_new, x_del, y_del


# mask image in red
def image_mask(img, mask_X, mask_y, color = np.array([255,0,0], dtype=np.uint8)):
    assert len(mask_X) == len(mask_y)
    img_mask = img.copy()
    for i in range(len(mask_X)):
        col = mask_X[i]
        raw = mask_y[i]
        img_mask[raw,col,:] = color
    return img_mask

def draw_line_cv(img, k, b, color=(255,0,0), line_width=1):
    h, w, _ = img.shape
    pt1 = (0, int(b))
    pt2 = (w-1, int(k*(w-1)+b))
    cv2.line(img, pt1, pt2, color, line_width)
    return img