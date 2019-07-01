# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:07:05 2019

@author: xue
"""

import os
import cv2
import ImageProcess
import RANSAC
import OCR
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output



def model_run (file, debug_params, mask_params, ransac_params, \
               find_corner_params, find_mark_params, find_coor_params, find_peak_params, \
               peak_value_table, archive_dir):
    # Debug parameters
    MODE_CHECK = debug_params['MODE_CHECK']
    SHOW_RANSAC_LINE = debug_params['SHOW_RANSAC_LINE']
    CHECK_RESULT = debug_params['CHECK_RESULT']
    SHOW_RESULT = debug_params['SHOW_RESULT']
    
    # Image pre-processing parameters 
    data_color_mask = mask_params['color_mask']
    start_ratio = mask_params['start_ratio']
    vert_ratio_num = mask_params['vert_ratio_num']
    vert_ratio_gap = mask_params['vert_ratio_gap']
    
    
    result = {}
    file_path, file_name = os.path.split(file)
    img = ImageProcess.load_cv_image(file)
    
    # extracted needed data
    X, y = ImageProcess.color_filter(img, data_color_mask)
    
    
    # find 4 plot corners to locate the scale information
    dic_corners = ImageProcess.find_corner(img, find_corner_params)
    img_w = dic_corners["plt_w"]
    img_h = dic_corners["plt_h"]
    pt_orig = dic_corners["left_down"]
    
    
    # sort data in the index of X
    X_sortX, y_sortX = ImageProcess.sort_X(X, y)
    start_point = int(start_ratio*img_w + dic_corners["left_down"][0])
    vert_thr_num = vert_ratio_num * img_h
    vert_thr_gap = vert_ratio_gap * img_h
    X_new, y_new = ImageProcess.del_start(X_sortX, y_sortX, start_point)
    X_new, y_new, x_del, y_del = ImageProcess.del_vert(X_new, y_new, vert_thr_num, vert_thr_gap)
    
    
    
    if MODE_CHECK:
        img_bord = img.copy()
        cv2.line(img_bord, dic_corners["left_up"], dic_corners["left_down"], (255,0,0), 2)
        cv2.line(img_bord, dic_corners["left_up"], dic_corners["right_up"], (255,0,0), 2)
        cv2.line(img_bord, dic_corners["left_down"], dic_corners["right_down"], (255,0,0), 2)
        cv2.line(img_bord, dic_corners["right_up"], dic_corners["right_down"], (255,0,0), 2)
        plt.figure(figsize=(20,6))
        plt.title("Bord Image")
        plt.imshow(img_bord)
        plt.show()
        
        img_mask = ImageProcess.image_mask(img, X_new, y_new, color=np.array([255,0,0], dtype=np.uint8))
        plt.figure(figsize=(20,6))
        plt.title("Mask Image")
        plt.imshow(img_mask)
        plt.show()
            
    # RANSAC line extraction
    loop_ransac = True
    while loop_ransac:
        try:
            model_ransac = RANSAC.line_ransac(X_new, y_new, ransac_params)
            dic_line_params = model_ransac.line_extraction_ransac()
            
            k1_list = dic_line_params["k1_list"]
            b1_list = dic_line_params["b1_list"]
            k2_list = dic_line_params["k2_list"]
            b2_list = dic_line_params["b2_list"]
            x_cross = dic_line_params["x_cross"]
            y_cross = dic_line_params["y_cross"]
            if SHOW_RANSAC_LINE:
                img_ransac = img.copy()
                for i in range(len(k1_list)):
                    k1 = k1_list[i]
                    b1 = b1_list[i]
                    k2 = k2_list[i]
                    b2 = b2_list[i]
                    img_ransac = ImageProcess.draw_line_cv(img_ransac, k1, b1, line_width=1)
                    img_ransac = ImageProcess.draw_line_cv(img_ransac, k2, b2, line_width=1)
#                cv2.line(img_ransac, (int(x_cross), 0), (int(x_cross), img_ransac.shape[0]), (0,0,0), 1)
#                cv2.line(img_ransac, (0, int(y_cross)), (img_ransac.shape[1], int(y_cross)), (0,0,0), 1)
                fig = plt.figure(figsize=(20,6))
                fig.suptitle("RANSAC Image: {}".format(file_name))
                #fig1 = plt.gcf()
                plt.imshow(img_ransac)
                plt.show()
            data_in = input("R: rerun, others: continue")
            if data_in == "R":
                clear_output()
                continue
            else:
                loop_ransac = False
        except ValueError:
            print( "Error: Fail to RANSAC")
            data_in = input("R: rerun, others: return")
            if data_in == "R":
                clear_output()
                continue
            elif data_in == "A":
                return result
            else:
                return result
        
    cv2.imwrite(os.path.join(archive_dir, file_name[:-4] + "T.PNG"), cv2.cvtColor(img_ransac, cv2.COLOR_RGB2BGR))
    
    # find coordinate markers
    try:
        dic_markers = OCR.find_mark(img, dic_corners, find_mark_params)
        mark_x = dic_markers["mark_X"]
        mark_y = dic_markers["mark_y"]
        dist_x = mark_x - pt_orig[0]
        dist_y = (pt_orig[1] - mark_y) / 2
    except ValueError:
        print( "Error: Fail to find markers")
        return result 
    
    # find coordinate
    try:
        dic_coor = OCR.find_coordinate(img, dic_corners, dic_markers, find_coor_params)
    except ValueError:
        print( "Error: Fail to find coordinate, need to enter manually later")
        dic_coor = {"CoorX":["",""], "CoorY":["",""],}
        
    # find peak value
    try:
#         peak_value = OCR.find_peak(img, dic_corners, find_peak_params)
        peak_value = peak_value_table[file_name[:-4]]
    except ValueError:
        print( "Error: Fail to find peak value, need to enter manually later")
        
    
            

    if CHECK_RESULT:
        while True:
            print ("\nX : {}, Y: {}, Peak Value: {}".format(dic_coor["CoorX"], dic_coor["CoorY"], peak_value))
            print("OCR Check, if right, press Enter, if not, please enter its index and true value")
            print("Enter A to abandon direcly")
            data_in = input("OCR correction: ")
            if data_in == "":
                break
            elif data_in == "A":
                return result
            else:
                data_in = data_in.split()
                if len(data_in) % 2 != 0:
                    print ("Wrong input type, please enter with format: index1, value1, ...")
                else:
                    for i in range(len(data_in)//2):
                        index = float(data_in[2*i])
                        value = float(data_in[2*i+1])
                        if index == 0 or index == 1:
                            dic_coor["CoorX"][int(index)] = str(value)
                        elif index == 2 or index == 3:
                            dic_coor["CoorY"][int(index-2)] = str(value)
                        elif index == 4:
                            peak_value = value
                        else:
                            print("Index should be in 0 to 4")
                            continue
    # find scale
    dic_scale = OCR.find_scale(dic_line_params, dic_corners, dic_markers, dic_coor)
        
    scale_x = dic_scale["scale_X"]
    scale_y = dic_scale["scale_y"]
    pt1_x = (x_cross - pt_orig[0]) * scale_x
    pt1_y = (pt_orig[1] - y_cross) * scale_y + float(dic_coor["CoorY"][0])
    pt2_y = float(peak_value)
    y_pix = int(pt_orig[1] - (pt2_y - float(dic_coor["CoorY"][0])) / scale_y)
    pt2_x = ((y_pix - b2)/k2 - pt_orig[0]) * scale_x
    pt1 = (pt1_x, pt1_y)
    pt2= (pt2_x, pt2_y)
    
    if SHOW_RESULT:
        print("=============Resualt=============")
        print("\n")
        print("The origin of coordinate is: {}".format(pt_orig))
        print("The firs marker in X axis is: {}".format((mark_x, pt_orig[1])))
        print("The second marker in y axis is: {}".format((pt_orig[0], mark_y)))
        print("The distance of every tick mark in X axis is {} pixel".format(dist_x))
        print("The distance of every tick mark in y axis is {} pixel".format(dist_y))
        print("The scale in x axis is: {}".format(dic_coor["CoorX"]))
        print("The scale in y axis is: {}".format(dic_coor["CoorY"]))
        print("The x scale is {}/pixel".format(float('%0.3f'%scale_x)))
        print("The y scale is {}/pixel".format(float('%0.5f'%scale_y)))
        print("Line 1 is: y = {}x + {}".format(('%0.3f'%k1), ('%0.3f'%b1)))
        print("Line 2 is: y = {}x + {}".format(('%0.3f'%k2), ('%0.3f'%b2)))
        print("The first point is: ({}, {})".format(float('%0.3f'%pt1_x), float('%0.3f'%pt1_y)))
        print("The second point is: ({}, {})".format(float('%0.3f'%pt2_x), float('%0.3f'%pt2_y)))
        print("\n")
        print("=============Finish=============")
    
    result["x0"] = dic_coor["CoorX"][0]
    result["x1"] = dic_coor["CoorX"][1]
    result["y0"] = dic_coor["CoorY"][0]
    result["y1"] = dic_coor["CoorY"][1]
    result["k1_list"] = k1_list
    result["b1_list"] = b1_list
    result["k2_list"] = k2_list
    result["b2_list"] = b2_list
    result["k1"] = '%0.3f'%k1
    result["b1"] = '%0.3f'%b1
    result["k2"] = '%0.3f'%k2
    result["b2"] = '%0.3f'%b2
    result["pt1_x"] = pt1_x
    result["pt1_y"] = pt1_y
    result["pt2_x"] = pt2_x
    result["pt2_y"] = pt2_y
    
    return result
