# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:03:11 2019

@author: xue
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from ImageProcess import rgb2gray, color_filter
from collections import Counter
import cv2
import pytesseract


# mark_X: first tick mark in X axis
# mark_y: second tick mark in y aixs (to avoid mixing with data)
# attention: the division of x axis and y axis by 5
def find_mark(img, dic_corners, find_mark_params):
    thr = find_mark_params["threshold"]
    search_range= find_mark_params["search_range"]
    bb_size_X= find_mark_params["bb_size_X"]
    jug_cond_X= find_mark_params["judge_condition_X"]
    bb_size_Y= find_mark_params["bb_size_Y"]
    jug_cond_Y= find_mark_params["judge_condition_Y"]
    SHOW_MARK= find_mark_params["SHOW_MARK"]
    pt_orig = dic_corners["left_down"]
    pt_xMax = dic_corners["right_down"]
    pt_yMax = dic_corners["left_up"]
    x_rough = int((pt_xMax[0] - pt_orig[0])/5 + pt_orig[0])
    y_rough = int(pt_orig[1] - 2*(pt_orig[1] - pt_yMax[1])/5)
    dic_markers = {}
    
    for i in range(x_rough-search_range, x_rough+search_range):
        Bounding_box = rgb2gray(img[pt_orig[1]-bb_size_X[0]:pt_orig[1],i-bb_size_X[1]:i,:])
        valid_num = np.sum(Bounding_box < thr)
        if SHOW_MARK:
            print("\n")
            print("X index: " + str(i))
            print(np.squeeze(Bounding_box))
            print("X valid number: {}".format(valid_num))
            plt.imshow(Bounding_box.T, cmap='gray')
            plt.show()
        if valid_num >= jug_cond_X:
            dic_markers["mark_X"] = i
            if SHOW_MARK:
                print("\n")
                print("mark X: {}".format(i))
            break

    for i in range(y_rough-search_range, y_rough+search_range):
        Bounding_box = rgb2gray(img[i:i+bb_size_Y[0],pt_orig[0]+1:pt_orig[0]+1+bb_size_Y[1],:])
        valid_num = np.sum(Bounding_box < thr)
        if SHOW_MARK:
            print("\n")
            print("y index: " + str(i))
            print(np.squeeze(Bounding_box))
            print("y valid number: {}".format(valid_num))
            plt.imshow(Bounding_box, cmap='gray')
            plt.show()
        if valid_num >= jug_cond_Y:
            dic_markers["mark_y"] = i
            if SHOW_MARK:
                print("\n")
                print("mark y: {}".format(i))
            break

    return dic_markers


# optical caracteristic recognition, with a given picture, identify the numbers inside;
# grey+ resize+ sharpen + binaryzation gray value >180 ->white, otherwise black
def ocr_detector(img, ocr_params, show_img=False):
    scale_w = ocr_params["scale_w"]
    scale_h = ocr_params["scale_h"]
    kernel = ocr_params["kernel"]
    sharpen_lower = ocr_params["sharpen_lower"]
    sharpen_upper = ocr_params["sharpen_upper"]
    img_gray = rgb2gray(img.copy())
    img_h, img_w = img_gray.shape
    img_resize = cv2.resize(img_gray, (int(scale_w * img_w),int(scale_h * img_h)),interpolation=cv2.INTER_LANCZOS4)
    img_sharpen = cv2.filter2D(img_resize, -1, kernel=kernel)
    _, img_thr = cv2.threshold(img_sharpen,sharpen_lower,sharpen_upper,cv2.THRESH_BINARY)
#    code = pytesseract.image_to_string(img_thr, config='digits')
    code_list = []
    
    rel = pytesseract.image_to_string(img_thr, config='digits') 
    if rel != "":
        try:
            code_list.append(float(rel))
        except:
            pass

    rel = pytesseract.image_to_string(img_thr[2:,2:], config='digits') 
    if rel != "":
        try:
            code_list.append(float(rel))
        except:
            pass
    
    rel = pytesseract.image_to_string(img_thr[2:,:-2], config='digits') 
    if rel != "":
        try:
            code_list.append(float(rel))
        except:
            pass
    
    rel = pytesseract.image_to_string(img_thr[:-2,2:], config='digits') 
    if rel != "":
        try:
            code_list.append(float(rel))
        except:
            pass
    
    rel = pytesseract.image_to_string(img_thr[-2:,:-2], config='digits') 
    if rel != "":
        try:
            code_list.append(float(rel))
        except:
            pass
    if code_list == []:
        code = ""
    else:
        code = str(stats.mode(code_list)[0][0])
    
    if show_img:
        plt.imshow(img_thr, cmap="gray")
        plt.show()
        print("code is : {}".format(code))
    return code

# found a area around the given point to identify the real axe value
def image_seg(img, pos_x=0, pos_y=0, bias_w = 0, bias_h = 0, img_w=40,img_h=14):
    assert len(img.shape) ==3
    h, w, _ = img.shape
    h_min = max(pos_y+bias_h, 0)
    h_max = min(pos_y+bias_h+img_h, h)
    w_min = max(pos_x+bias_w, 0)
    w_max = min(pos_x+bias_w+img_w, w)
    im = img[h_min:h_max, w_min:w_max,:].copy()
    return im
    
# find the coordiates in the image frame of all axes values
def find_coordinate(img, dic_corners, dic_markers, find_coor_params):
    pt_orig = dic_corners["left_down"]
    mark_x = dic_markers["mark_X"]
    mark_y = dic_markers["mark_y"]
    bias_w_x = find_coor_params["bias_w_x"]
    bias_h_x = find_coor_params["bias_h_x"]
    img_w_x = find_coor_params["img_w_x"]
    img_h_x = find_coor_params["img_h_x"]
    bias_w_y = find_coor_params["bias_w_y"]
    bias_h_y = find_coor_params["bias_h_y"]
    img_w_y = find_coor_params["img_w_y"]
    img_h_y = find_coor_params["img_h_y"]
    ocr_params = find_coor_params["ocr_params"]
    SHOW_COOR = find_coor_params["SHOW_COOR"]
    dist_x = mark_x - pt_orig[0]
    dist_y = (pt_orig[1] - mark_y) / 2
    # find x coordinate
    coor_x = []
    pos_y = int(pt_orig[1])
    for i in range(2):
        pos_x = int(pt_orig[0] + i * dist_x)
        img_seg = image_seg(img, pos_x, pos_y, bias_w=bias_w_x, bias_h=bias_h_x, img_w=img_w_x, img_h=img_h_x)
        if SHOW_COOR:
            print("\n")
            print("X axis, coordinate: {}".format(i))
            plt.imshow(img_seg, cmap="gray")
            plt.show()
        code = ocr_detector(img_seg,ocr_params,SHOW_COOR)
        if i == 0 and code == "":
            code = str(0)
        if SHOW_COOR:
            print("OCR result is {}".format(code))
        coor_x.append(code)
    # find y coordinate
    coor_y = []
    pos_x = int(pt_orig[0])
    for i in range(2):
        pos_y = int(pt_orig[1] - i * dist_y)
        img_seg = image_seg(img, pos_x, pos_y, bias_w=bias_w_y, bias_h=bias_h_y, img_w=img_w_y, img_h=img_h_y)
        if SHOW_COOR:
            print("\n")
            print("y axis, coordinate: {}".format(i))
            plt.imshow(img_seg, cmap="gray")
            plt.show()
        code = ocr_detector(img_seg,ocr_params,SHOW_COOR)
        if i == 0 and float(code) > 0:
            code = "-" + code
        if SHOW_COOR:
            print("OCR result is {}".format(code))
        coor_y.append(code)
    dic_coor = {}
    dic_coor["CoorX"] = coor_x
    dic_coor["CoorY"] = coor_y
    return dic_coor

# find the ratio of pixel and its values in real pictures
def find_scale(dic_line_params, dic_corners, dic_markers, dic_coor):
    dic_scale = {}
    pt_orig = dic_corners["left_down"]
    mark_X = dic_markers["mark_X"]
    mark_y = dic_markers["mark_y"]
    dist_X = mark_X - pt_orig[0]
    dist_y = (pt_orig[1] - mark_y) / 2
    coor_X = dic_coor["CoorX"]
    coor_y = dic_coor["CoorY"]
    value_X = float(coor_X[1]) - float(coor_X[0])
    value_y = float(coor_y[1]) - float(coor_y[0])
    scale_X = value_X / dist_X
    scale_y = value_y / dist_y
    dic_scale["scale_X"] = scale_X
    dic_scale["scale_y"] = scale_y
    return dic_scale

# by identify the max number in the fig
def find_peak(img, dic_corners, find_peak_params):
    bias_w_l = find_peak_params["bias_w_l"]
    bias_w_r = find_peak_params["bias_w_r"]
    bias_h = find_peak_params["bias_h"]
    img_w = find_peak_params["img_w"]
    img_h = find_peak_params["img_h"]
    x_thr = find_peak_params["x_thr"]
    y_thr = find_peak_params["y_thr"]
    color_mask = find_peak_params["color_mask"]
    orc_params = find_peak_params["ocr_params"]
    SHOW_PEAK = find_peak_params["SHOW_PEAK"]
    
    pt_ru = dic_corners["right_up"]
    pt_lu = dic_corners["left_up"]
    img_seg = img[pt_ru[1]+bias_h:pt_ru[1]+bias_h+img_h,\
                  pt_lu[0]+bias_w_l:pt_ru[0]+bias_w_r,:].copy()
#    plt.figure(figsize=(30,3))
#    plt.imshow(img_seg)
#    plt.show()
    X, y = color_filter(img_seg, color_mask)
    count_x = Counter(X).most_common(16)
    count_y = Counter(y).most_common(4)
#    print(count_x)
#    print(count_y)
    x_list = []
    y_list = []
    for i in range(16):
        if count_x[i][1] > x_thr:
            x_list.append(count_x[i][0])
    for i in range(4):     
        if count_y[i][1] > y_thr:
            y_list.append(count_y[i][0])
    x_bound_l = np.max([x for x in x_list if x < np.max(x_list) - img_w/2])
    x_bound_r = np.max([x for x in x_list if x > np.max(x_list) - img_w/2])
    y_bound_t = np.max([y for y in y_list if y < np.median(y_list)])
    y_bound_b = np.max([y for y in y_list if y > np.median(y_list)])
    img_bb = img_seg[y_bound_t+2:y_bound_b-1,x_bound_l+2:x_bound_r-2,:]
    img_bb = cv2.copyMakeBorder(img_bb, 2,2,2,2,cv2.BORDER_CONSTANT, value=[255,255,255])
    if SHOW_PEAK:
        plt.figure(figsize=(10,3))
        plt.imshow(img_bb)
        plt.show()
    
    code = ocr_detector(img_bb, orc_params, show_img=SHOW_PEAK)
    return code