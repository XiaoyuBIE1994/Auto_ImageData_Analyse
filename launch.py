# -*- coding: utf-8 -*-

import os
import shutil
import numpy as np
import Exchange
import Intergration
from IPython.display import clear_output
from configparser import ConfigParser

# Re-write configure class, enable to distinguish betwwen upper and lower case
class myconf(ConfigParser):
    def __init__(self,defaults=None):
        ConfigParser.__init__(self,defaults=None)
    def optionxform(self, optionstr):
        return optionstr

def launch(config_path = os.getcwd()):
    
    # read config file
    cfg_path = os.path.join(config_path, "cfg.ini")
    cfg = myconf()
    cfg.read(cfg_path)
    
    work_dir = cfg.get('path', 'work_dir')
    local_dir = cfg.get('path', 'local_dir')
    abandon_dir = cfg.get('path', 'abandon_dir')
    processed_dir = cfg.get('path', 'processed_dir')
    archive_dir = cfg.get('path', 'archive_dir')
    
    debug_params = {}
    mask_params = {}
    ransac_params = {}
    ocr_params = {}
    find_corner_params = {}
    find_mark_params = {}
    find_coor_params = {}
    find_peak_params = {}
    
    debug_params['SHOW_RANSAC_LINE'] = cfg.getboolean('debug parameters', 'SHOW_RANSAC_LINE')
    debug_params['CHECK_RESULT'] = cfg.getboolean('debug parameters', 'CHECK_RESULT')
    debug_params['MODE_CHECK'] = cfg.getboolean('debug parameters', 'MODE_CHECK')
    debug_params['SHOW_RESULT'] = cfg.getboolean('debug parameters', 'SHOW_RESULT')
    
    mask_params['color_mask'] = list(map(int, cfg['mask parameters']['color_mask'].split(' ')))
    mask_params['start_ratio'] = cfg.getfloat('mask parameters', 'start_ratio')
    mask_params['vert_ratio_num'] = cfg.getfloat('mask parameters', 'vert_ratio_num')
    mask_params['vert_ratio_gap'] = cfg.getfloat('mask parameters', 'vert_ratio_gap')
    
    ransac_params['max_iter'] = cfg.getint('ransac parameters', 'max_iter')
    ransac_params['mean_iter'] = cfg.getint('ransac parameters', 'mean_iter')
    ransac_params['line1_gap'] = cfg.getint('ransac parameters', 'line1_gap')
    ransac_params['line1_cut_range'] = cfg.getfloat('ransac parameters', 'line1_cut_range')
    ransac_params['line1_k_min'] = cfg.getfloat('ransac parameters', 'line1_k_min')
    ransac_params['line1_k_max'] = cfg.getfloat('ransac parameters', 'line1_k_max')
    ransac_params['line2_gap'] = cfg.getfloat('ransac parameters', 'line2_gap')
    ransac_params['line2_cut_range'] = cfg.getfloat('ransac parameters', 'line2_cut_range')
    ransac_params['line2_k_max'] = cfg.getfloat('ransac parameters', 'line2_k_max')
    
    ocr_params['scale_w'] = cfg.getint('OCR parameters', 'scale_w')
    ocr_params['scale_h'] = cfg.getint('OCR parameters', 'scale_h')
    ocr_params['sharpen_lower'] = cfg.getint('OCR parameters', 'sharpen_lower')
    ocr_params['sharpen_upper'] = cfg.getint('OCR parameters', 'sharpen_upper')
    ocr_params['kernel'] = np.asarray(list(map(int, cfg['OCR parameters']['kernel'].split(' ')))).reshape(3,3).astype(np.float32) 
    
    
    find_corner_params['color_mask'] = list(map(int, cfg['find corner parameters']['color_mask'].split(' ')))
    find_corner_params['X_search_range'] = cfg.getint('find corner parameters', 'X_search_range')
    find_corner_params['y_search_range'] = cfg.getint('find corner parameters', 'y_search_range')
    
    find_mark_params['SHOW_MARK'] = cfg.getboolean('find mark parameters', 'SHOW_MARK')
    find_mark_params['threshold'] = cfg.getint('find mark parameters', 'threshold')
    find_mark_params['search_range'] = cfg.getint('find mark parameters', 'search_range')
    find_mark_params['bb_size_X'] = (cfg.getint('find mark parameters', 'BoundingBox_X_x'), \
                                     cfg.getint('find mark parameters', 'BoundingBox_X_y'))
    find_mark_params['judge_condition_X'] = cfg.getint('find mark parameters', 'judge_condition_X')
    find_mark_params['bb_size_Y'] = (cfg.getint('find mark parameters', 'BoundingBox_Y_x'), \
                                     cfg.getint('find mark parameters', 'BoundingBox_Y_y'))
    find_mark_params['judge_condition_Y'] = cfg.getint('find mark parameters', 'judge_condition_Y')
    
    find_coor_params['SHOW_COOR'] = cfg.getboolean('find coordinate parameters', 'SHOW_COOR')
    find_coor_params['bias_w_x'] = cfg.getint('find coordinate parameters', 'bias_w_x')
    find_coor_params['bias_h_x'] = cfg.getint('find coordinate parameters', 'bias_h_x')
    find_coor_params['img_w_x'] = cfg.getint('find coordinate parameters', 'img_w_x')
    find_coor_params['img_h_x'] = cfg.getint('find coordinate parameters', 'img_h_x')
    find_coor_params['bias_w_y'] = cfg.getint('find coordinate parameters', 'bias_w_y')
    find_coor_params['bias_h_y'] = cfg.getint('find coordinate parameters', 'bias_h_y')
    find_coor_params['img_w_y'] = cfg.getint('find coordinate parameters', 'img_w_y')
    find_coor_params['img_h_y'] = cfg.getint('find coordinate parameters', 'img_h_y')
    find_coor_params['ocr_params'] = ocr_params
    
    find_peak_params['SHOW_PEAK'] = cfg.getboolean('find peak parameters', 'SHOW_PEAK')
    find_peak_params['scale_w'] = cfg.getint('find peak parameters', 'scale_w')
    find_peak_params['bias_w_l'] = cfg.getint('find peak parameters', 'bias_w_l')
    find_peak_params['bias_w_r'] = cfg.getint('find peak parameters', 'bias_w_r')
    find_peak_params['bias_h'] = cfg.getint('find peak parameters', 'bias_h')
    find_peak_params['img_w'] = cfg.getint('find peak parameters', 'img_w')
    find_peak_params['img_h'] = cfg.getint('find peak parameters', 'img_h')
    find_peak_params['x_thr'] = cfg.getint('find peak parameters', 'x_thr')
    find_peak_params['y_thr'] = cfg.getint('find peak parameters', 'y_thr')
    find_peak_params['color_mask'] = list(map(int, cfg['find peak parameters']['color_mask'].split(' ')))
    find_peak_params['ocr_params'] = ocr_params
    
    
    # read peak value
    peak_value_table = Exchange.import_peak(work_dir)
    
    # loop for image recognition
    result_dir = {}
    if not os.path.isdir(abandon_dir):
        os.mkdir(abandon_dir)
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)
    if not os.path.isdir(archive_dir):
        os.mkdir(archive_dir)
    for subdir, dirs, files in os.walk(local_dir):
        for file in files:
            if ".PNG" in file:
                filename = file
#                 print (peakvalue[filename[:-4]])
                imagepath = os.path.join(subdir, file)
#                 print (imagepath)
                result = Intergration.model_run (imagepath, debug_params, mask_params, ransac_params, \
                                                 find_corner_params, find_mark_params, find_coor_params, find_peak_params, \
                                                 peak_value_table, archive_dir)
                if result == {}:
                    copy_path = os.path.join(abandon_dir, file)
                    shutil.move(imagepath, copy_path)
                    data_in = input("Error: Auto detection fail, press E to exit, others to continue")
                    if data_in == "E":
                        clear_output()
                        return result_dir
                    else:
                        clear_output()
                        continue

                data_in = input("E: exit, S: save and exit, A: abandon result, others: continue\n")
                if data_in == "E":
                    return result_dir
                elif data_in == "S":
                    result_dir[filename[:-4]] = result
                    Exchange.export_result(filename, work_dir, result)
                    copy_path = os.path.join(processed_dir, file)
                    shutil.move(imagepath, copy_path)
                    return result_dir
                elif data_in == "A":
                    result_dir = {}
                    copy_path = os.path.join(abandon_dir, file)
#                     shutil.copy(imagepath, copy_path)
                    shutil.move(imagepath, copy_path)
                else:
                    Exchange.export_result(filename, work_dir, result)
                    result_dir[filename[:-4]] = result
                    copy_path = os.path.join(processed_dir, file)
                    shutil.move(imagepath, copy_path)
                clear_output()
    return result_dir


if __name__ == '__main__':
    result_dir = launch()