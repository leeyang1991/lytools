# coding=utf-8
import sys

__python_version = sys.version_info.major
assert __python_version == 3, 'Python Version Error'

import numpy as np
import numpy.ma as ma

from tqdm import tqdm

from scipy import stats
from scipy.stats import gaussian_kde as kde
from scipy import interpolate

import pandas as pd
import geopandas as gpd

import seaborn as sns

from netCDF4 import Dataset

import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
from threading import Thread, Lock
from pathos.multiprocessing import ProcessPool


from functools import partialmethod
import os
from os.path import *
import copyreg
import types
import math
import copy
import random
import pickle
import time
import datetime
import itertools
import subprocess
import shutil

import urllib3
import requests
from requests.auth import HTTPBasicAuth
import dns.resolver

from operator import itemgetter
from itertools import groupby

from osgeo import osr
from osgeo import ogr
from osgeo import gdal

from pyproj import Transformer
from shapely.geometry import box
from shapely.geometry import mapping

import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio import windows

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches

import hashlib
from calendar import monthrange

import psutil
import zipfile
from functools import wraps
from pathlib import Path

import warnings

class Tools:

    def __init__(self):
        pass

    def mk_dir(self, dir, force=False):
        if not os.path.isdir(dir):
            if force == True:
                try:
                    os.makedirs(dir)
                except Exception as e:
                    print(e)

            else:
                try:
                    os.mkdir(dir)
                except Exception as e:
                    print(e)

    def mkdir(self, dir, force=False):
        self.mk_dir(dir, force)

    def mk_class_dir(self, class_name, result_root_this_script, mode=1):
        if mode == 1:
            this_class_arr = join(result_root_this_script, f'arr/{class_name}/')
            this_class_tif = join(result_root_this_script, f'tif/{class_name}/')
            this_class_png = join(result_root_this_script, f'png/{class_name}/')
            self.mkdir(this_class_arr, force=True)
            self.mkdir(this_class_tif, force=True)
            self.mkdir(this_class_png, force=True)
        elif mode == 2:
            this_class_arr = join(result_root_this_script, f'{class_name}/arr')
            this_class_tif = join(result_root_this_script, f'{class_name}/tif')
            this_class_png = join(result_root_this_script, f'{class_name}/png')
            self.mkdir(this_class_arr, force=True)
            self.mkdir(this_class_tif, force=True)
            self.mkdir(this_class_png, force=True)
        else:
            raise Exception('mode error')

        return this_class_arr, this_class_tif, this_class_png

    def path_join(self, *args):
        path = os.path.join(*args)
        return path

    def load_npy_dir(self, fdir, condition=''):
        dic = {}
        for f in tqdm(Tools().listdir(fdir), desc='loading ' + fdir):
            if not condition in f:
                continue
            dic_i = self.load_npy(os.path.join(fdir, f))
            dic.update(dic_i)
        return dic

    def load_dict_txt(self, f):
        nan = np.nan
        dic = eval(open(f, 'r').read())
        return dic

    def save_dict_to_txt(self, results_dic, outf):
        fw = outf + '.txt'
        fw = open(fw, 'w')
        fw.write(str(results_dic))
        pass

    def save_dict_to_binary(self, dic, outf):
        if outf.endswith('pkl'):
            fw = open(outf, 'wb')
            pickle.dump(dic, fw)
            fw.close()
        else:
            fw = open(outf + '.pkl', 'wb')
            pickle.dump(dic, fw)
            fw.close()

    def save_npy(self, dic, outf):
        np.save(outf, dic)

    def load_dict_from_binary(self, f):
        fr = open(f, 'rb')
        try:
            dic = pickle.load(fr)
        except:
            dic = pickle.load(fr, encoding="latin1")

        return dic

    def load_npy(self, f):
        try:
            return dict(np.load(f, allow_pickle=True).item())
        except Exception as e:
            return dict(np.load(f, allow_pickle=True, encoding='latin1').item())
        except:
            return dict(np.load(f).item())

    def save_distributed_perpix_dic(self, dic, outdir, n=10000,prefix='spatial_dict',istqdm=True):
        '''
        :param dic:
        :param outdir:
        :param n: save to each file every n sample
        :return:
        '''
        flag = 0
        temp_dic = {}
        if istqdm:
            iterator = tqdm(dic, desc='saving...')
        else:
            iterator = dic
        for key in iterator:
            flag += 1
            arr = dic[key]
            arr = np.array(arr)
            temp_dic[key] = arr
            if flag % n == 0:
                outf = join(outdir, f'{prefix}_{int(flag / n):03d}.npy')
                np.save(outf, temp_dic)
                temp_dic = {}
        outf = join(outdir, f'{prefix}_{int(flag / n) + 1:03d}.npy')
        np.save(outf, temp_dic)

        pass

    def load_df(self, f):
        df = pd.read_pickle(f)
        df = pd.DataFrame(df)
        return df

    def save_df(self, df, outf):
        df.to_pickle(outf)

    def df_to_excel(self, df, dff, n=1000, random=False):
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

    def mask_999999_arr(self, arr, warning=True):
        arr = np.array(arr, dtype=float)
        arr[arr < -9999] = np.nan
        if warning == True:
            raise UserWarning('\33[7m' + "Fatal Bug !!!  \t  Need to change !!!  \t  Value return added" + '\33[0m')
        return arr

    def lonlat_to_address(self, lon, lat):
        temporary_foler = 'temporary_foler/lonlat_to_address'
        self.mkdir(temporary_foler, force=True)
        outf = join(temporary_foler, 'address.xlsx')
        if not isfile(outf):
            address = self.__lonlat_to_address(lon, lat)
            add_dic = {
                'lon_lat': str((lon, lat)),
                'address': address
            }
            print(add_dic)
            df = pd.DataFrame(data=add_dic, index=[0])
            df.to_excel(outf, index=False)
            return address
        else:
            df = pd.read_excel(outf, index_col=False)
            lon_lat = str((lon, lat))
            address_dic = self.df_to_dic(df, 'lon_lat')
            # print(address_dic)
            # exit()
            if lon_lat in address_dic:
                return address_dic[lon_lat]['address']
            else:
                add = self.__lonlat_to_address(lon, lat)
                df.loc[-1] = {'lon_lat': lon_lat, 'address': add}
                df.index = df.index + 1
                df.to_excel(outf, index=False)
                return add

    def __lonlat_to_address(self, lon, lat):
        return None

    def spatial_arr_filter_n_sigma(self, spatial_arr, n=3):
        arr_std = np.nanstd(spatial_arr)
        arr_mean = np.nanmean(spatial_arr)
        top = arr_mean + n * arr_std
        bottom = arr_mean - n * arr_std
        spatial_arr[spatial_arr > top] = np.nan
        spatial_arr[spatial_arr < bottom] = np.nan

    def pix_to_address(self, pix, outf, pix_to_lon_lat_dic_f):
        # 只适用于单个像素查看，不可大量for循环pix，存在磁盘重复读写现象
        # outf = self.this_class_arr + 'pix_to_address_history.npy'
        if not os.path.isfile(outf):
            np.save(outf, {0: 0})
        # pix_to_lon_lat_dic_f = DIC_and_TIF().this_class_arr+'pix_to_lon_lat_dic.npy'
        if not os.path.isfile(pix_to_lon_lat_dic_f):
            DIC_and_TIF().spatial_tif_to_lon_lat_dic(pix_to_lon_lat_dic_f)
        lon_lat_dic = self.load_npy(pix_to_lon_lat_dic_f)
        # print(pix)
        lon, lat = lon_lat_dic[pix]
        print((lon, lat))

        history_dic = self.load_npy(outf)

        if pix in history_dic:
            # print(history_dic[pix])
            return lon, lat, history_dic[pix]
        else:
            address = self.lonlat_to_address(lon, lat)
            key = pix
            val = address
            history_dic[key] = val
            np.save(outf, history_dic)
            return lon, lat, address

    def interp_1d(self, val, threashold):
        if len(val) == 0 or np.std(val) == 0:
            return [None]

        # 1、插缺失值
        x = []
        val_new = []
        flag = 0
        for i in range(len(val)):
            if val[i] >= threashold:
                flag += 1.
                index = i
                x = np.append(x, index)
                val_new = np.append(val_new, val[i])
        if flag / len(val) < 0.3:
            return [None]
        # if flag == 0:
        #     return
        interp = interpolate.interp1d(x, val_new, kind='nearest', fill_value="extrapolate")

        xi = list(range(len(val)))
        yi = interp(xi)

        # 2、利用三倍sigma，去除离群值
        # print(len(yi))
        val_mean = np.mean(yi)
        sigma = np.std(yi)
        n = 3
        yi[(val_mean - n * sigma) > yi] = -999999
        yi[(val_mean + n * sigma) < yi] = 999999
        bottom = val_mean - n * sigma
        top = val_mean + n * sigma

        # 3、插离群值
        xii = []
        val_new_ii = []

        for i in range(len(yi)):
            if -999999 < yi[i] < 999999:
                index = i
                xii = np.append(xii, index)
                val_new_ii = np.append(val_new_ii, yi[i])

        interp_1 = interpolate.interp1d(xii, val_new_ii, kind='nearest', fill_value="extrapolate")

        xiii = list(range(len(val)))
        yiii = interp_1(xiii)

        return yiii

    def interp_1d_1(self, val, threshold):
        # 不插离群值 只插缺失值
        if len(val) == 0 or np.std(val) == 0:
            return [None]

        # 1、插缺失值
        x = []
        val_new = []
        flag = 0
        for i in range(len(val)):
            if val[i] >= threshold:
                flag += 1.
                index = i
                x = np.append(x, index)
                val_new = np.append(val_new, val[i])
        if flag / len(val) < 0.3:
            return [None]
        interp = interpolate.interp1d(x, val_new, kind='nearest', fill_value="extrapolate")

        xi = list(range(len(val)))
        yi = interp(xi)

        return yi

    def interp_nan(self, val, kind='nearest', valid_percent=0.1):
        if len(val) == 0 or np.std(val) == 0:
            return [None]

        # 1、插缺失值
        x = []
        val_new = []
        flag = 0
        for i in range(len(val)):
            if not np.isnan(val[i]):
                flag += 1.
                index = i
                x.append(index)
                # val_new = np.append(val_new, val[i])
                val_new.append(val[i])
        if flag / len(val) < valid_percent:
            return [None]
        interp = interpolate.interp1d(x, val_new, kind=kind, fill_value="extrapolate")

        xi = list(range(len(val)))
        yi = interp(xi)

        return yi


    def interp_nan_climatology(self, vals):
        vals = np.array(vals)
        vals_reshape = np.reshape(vals, (-1, 12))
        vals_reshape_T = vals_reshape.T
        month_mean = []
        for m in vals_reshape_T:
            mean = np.nanmean(m)
            month_mean.append(mean)
        nan_index = np.isnan(vals)
        val_new = []
        for i in range(len(nan_index)):
            isnan = nan_index[i]
            month = i % 12
            interp_val = month_mean[month]
            if isnan:
                val_new.append(interp_val)
            else:
                val_new.append(vals[i])
        val_new = np.array(val_new)
        return val_new

    def detrend_vals(self, vals):
        vals = np.array(vals)
        if self.is_all_nan(vals):
            return vals
        x = np.arange(len(vals))
        not_nan_ind = ~np.isnan(vals)
        m, b, r_val, p_val, std_err = stats.linregress(x[not_nan_ind], vals[not_nan_ind])
        detrend_vals = vals - (m * x + b)
        detrend_vals = detrend_vals + np.nanmean(vals)
        return detrend_vals

    def detrend_dic(self, dic):
        dic_new = {}
        for key in dic:
            vals = dic[key]
            if len(vals) == 0:
                dic_new[key] = []
                continue
            try:
                vals_new = self.detrend_vals(vals)
            except:
                vals_new = np.nan
                # print(vals)
                # exit()
            dic_new[key] = vals_new

        return dic_new

    def arr_mean(self, arr, threshold):
        grid = arr > threshold
        arr_mean = np.mean(arr[np.logical_not(grid)])
        return arr_mean

    def arr_mean_nan(self, arr):

        flag = 0.
        sum_ = 0.
        x = []
        for i in arr:
            if np.isnan(i):
                continue
            sum_ += i
            flag += 1
            x.append(i)
        if flag == 0:
            return np.nan, np.nan
        else:
            mean = sum_ / flag
            # xerr = mean/np.std(x,ddof=1)
            xerr = np.std(x)
            # print mean,xerr
            # if xerr > 10:
            #     print x
            #     print xerr
            #     print '........'
            #     plt.hist(x,bins=10)
            #     plt.show()
            #     exit()
            return mean, xerr

    def arrs_nan_trend(self, arrs):
        arrs = np.array(arrs)
        trend_matrix = []
        P_matrix = []
        for r in range(arrs.shape[1]):
            trend_i = []
            p_i = []
            for c in range(arrs.shape[2]):
                val_list = []
                for arr in arrs:
                    val = arr[r, c]
                    val_list.append(val)
                val_list = np.array(val_list)
                if self.is_all_nan(val_list):
                    trend_i.append(np.nan)
                    p_i.append(np.nan)
                    continue
                a, b, R, p = self.nan_line_fit(np.arange(len(val_list)), val_list)
                trend_i.append(a)
                p_i.append(p)
            trend_matrix.append(trend_i)
            P_matrix.append(p_i)
        trend_matrix = np.array(trend_matrix)
        P_matrix = np.array(P_matrix)
        return trend_matrix, P_matrix

    def pick_vals_from_2darray(self, array, index, pick_nan=False):
        # 2d
        ################# check zone #################
        # plt.imshow(array)
        # for r,c in index:
        #     # print(r,c)
        #     array[r,c] = 100
        # #     # exit()
        # plt.figure()
        # plt.imshow(array)
        # plt.show()
        ################# check zone #################
        if pick_nan == False:
            picked_val = []
            for r, c in index:
                val = array[r, c]
                if np.isnan(val):
                    continue
                picked_val.append(val)
            picked_val = np.array(picked_val)
            return picked_val
        else:
            picked_val = []
            for r, c in index:
                val = array[r, c]
                picked_val.append(val)
            picked_val = np.array(picked_val)
            return picked_val

    def pick_vals_from_1darray(self, arr, index):
        # 1d
        picked_vals = []
        for i in index:
            picked_vals.append(arr[i])
        picked_vals = np.array(picked_vals)
        return picked_vals

    def pick_min_indx_from_1darray(self, arr, indexs):
        min_index = 99999
        min_val = 99999
        # plt.plot(arr)
        # plt.show()
        for i in indexs:
            val = arr[i]
            # print val
            if val < min_val:
                min_val = val
                min_index = i
        return min_index

    def pick_max_indx_from_1darray(self, arr, indexs):
        max_index = 99999
        max_val = -99999
        # plt.plot(arr)
        # plt.show()
        for i in indexs:
            val = arr[i]
            # print val
            if val > max_val:
                max_val = val
                max_index = i
        return max_index

    def pick_max_key_val_from_dict(self, dic):
        key_list = []
        val_list = []
        for key in dic:
            val = dic[key]
            key_list.append(key)
            val_list.append(val)

        max_val_index = np.argmax(val_list)
        max_key = key_list[max_val_index]
        return max_key

    def pick_max_n_index(self, vals, n):
        vals = np.array(vals)
        vals[np.isnan(vals)] = -np.inf
        argsort = np.argsort(vals)
        max_n_index = argsort[::-1][:n]
        max_n_val = self.pick_vals_from_1darray(vals, max_n_index)
        return max_n_index, max_n_val

    def point_to_shp(self, inputlist, outSHPfn,wkt=None):
        '''

        :param inputlist:

        # input list format
        # [
        # [lon,lat,{'col1':value1,'col2':value2,...}],
        #      ...,
        # [lon,lat,{'col1':value1,'col2':value2,...}]
        # ]

        :param outSHPfn:
        :return:
        '''

        fieldType_dict = {
            'float': ogr.OFTReal,
            'int': ogr.OFTInteger,
            'str': ogr.OFTString
        }

        if len(inputlist) > 0:
            if outSHPfn.endswith('.shp'):
                outSHPfn = outSHPfn
            else:
                outSHPfn = outSHPfn + '.shp'
            # Create the output shapefile
            shpDriver = ogr.GetDriverByName("ESRI Shapefile")
            if os.path.exists(outSHPfn):
                shpDriver.DeleteDataSource(outSHPfn)
            outDataSource = shpDriver.CreateDataSource(outSHPfn)
            outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPoint)
            # Add the fields we're interested in
            col_list = inputlist[0][2].keys()
            col_list = list(col_list)
            col_list.sort()
            value_type_list = []
            for col in col_list:
                if len(col) > 10:
                    raise UserWarning(
                        f'The length of column name "{col}" is too long, length must be less than 10\nplease rename the column')
                value = inputlist[0][2][col]
                value_type = type(value)
                value_type_list.append(value_type)
            for i in range(len(value_type_list)):
                ogr_type = fieldType_dict[value_type_list[i].__name__]
                col_name = col_list[i]
                fieldDefn = ogr.FieldDefn(col_name, ogr_type)
                outLayer.CreateField(fieldDefn)
            for i in range(len(inputlist)):
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(inputlist[i][0], inputlist[i][1])
                featureDefn = outLayer.GetLayerDefn()
                outFeature = ogr.Feature(featureDefn)
                outFeature.SetGeometry(point)
                for j in range(len(col_list)):
                    col_name = col_list[j]
                    value = inputlist[i][2][col_name]
                    outFeature.SetField(col_name, value)
                # outFeature.SetField('val', inputlist[i][2])
                # 加坐标系
                spatialRef = osr.SpatialReference()
                if wkt is None:
                    wkt=DIC_and_TIF().wkt_84()
                spatialRef.ImportFromWkt(wkt)
                spatialRef.MorphToESRI()
                file = open(outSHPfn[:-4] + '.prj', 'w')
                file.write(spatialRef.ExportToWkt())
                file.close()

                outLayer.CreateFeature(outFeature)
                outFeature.Destroy()
            outFeature = None

    def line_to_shp(self, inputlist, outSHPfn):
        ############重要#################
        gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
        ############重要#################
        # start,end,outSHPfn,val1,val2,val3,val4,val5
        # _,_,_,_=start[1],start[0],end[0],end[1]

        shpDriver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(outSHPfn):
            shpDriver.DeleteDataSource(outSHPfn)
        outDataSource = shpDriver.CreateDataSource(outSHPfn)
        outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbLineString)

        # create line geometry
        line = ogr.Geometry(ogr.wkbLineString)

        for i in range(len(inputlist)):
            start = inputlist[i][0]
            end = inputlist[i][1]

            line.AddPoint(start[0], start[1])
            line.AddPoint(end[0], end[1])

            featureDefn = outLayer.GetLayerDefn()
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(line)
            outLayer.CreateFeature(outFeature)
            outFeature.Destroy()
            line = ogr.Geometry(ogr.wkbLineString)
            outFeature = None

        # define the spatial reference, WGS84
        spatialRef = osr.SpatialReference()
        wkt_84 = DIC_and_TIF().wkt_84()
        spatialRef.ImportFromWkt(wkt_84)
        spatialRef.MorphToESRI()
        file = open(outSHPfn[:-4] + '.prj', 'w')
        file.write(spatialRef.ExportToWkt())
        file.close()

    def read_point_shp(self, shpfn):
        shpDriver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = shpDriver.Open(shpfn, 0)
        layer = dataSource.GetLayer()
        featureCount = layer.GetFeatureCount()
        shp_layer_dict = {}
        flag = 0
        for feature in layer:
            geom = feature.GetGeometryRef()
            x = geom.GetX()
            y = geom.GetY()
            shp_layer_dict_i = {
                'point_x_pos': x,
                'point_y_pos': y,
            }
            field_count = feature.GetFieldCount()
            for i in range(field_count):
                field = feature.GetField(i)
                field_name = feature.GetFieldDefnRef(i).GetName()
                shp_layer_dict_i[field_name] = field
            shp_layer_dict[flag] = shp_layer_dict_i
            flag += 1
        df = self.dic_to_df(shp_layer_dict,'point_idx')
        return df

    def show_df_all_columns(self):
        pd.set_option('display.max_columns', None)
        pass

    def print_head_n(self, df, n=10, pause_flag=0):
        self.show_df_all_columns()
        print(df.head(n))
        print('Dataframe length:', len(df))
        print('Dataframe columns length:', len(df.columns))
        if pause_flag == 1:
            pause()

    def remove_np_nan(self, arr, is_relplace=False):
        arr = np.array(arr)
        if is_relplace:
            arr = arr[~np.isnan(arr)]
        else:
            arr_cp = copy.copy(arr)
            arr_cp = arr_cp[~np.isnan(arr_cp)]
            return arr_cp
        pass

    def plot_colors_palette(self, cmap):
        sns.palplot(cmap)

    def group_consecutive_vals(self, in_list):
        # 连续值分组
        ranges = []
        for _, group in groupby(enumerate(in_list), lambda index_item: index_item[0] - index_item[1]):
            group = list(map(itemgetter(1), group))
            if len(group) > 1:
                ranges.append(list(range(group[0], group[-1] + 1)))
            else:
                ranges.append([group[0]])
        return ranges

    def listdir(self, fdir):
        '''
        Mac OS
        list the names of the files in the directory
        return sorted files list without '.DS_store'
        '''
        list_dir = []
        for f in sorted(os.listdir(fdir)):
            if f.startswith('.'):
                continue
            list_dir.append(f)
        return list_dir


    def drop_repeat_val_from_list(self, in_list):
        in_list = list(in_list)
        in_list = set(in_list)
        in_list = list(in_list)
        in_list.sort()

        return in_list

    def nan_correlation(self, val1_list, val2_list, method='pearson'):
        # pearson correlation of val1 and val2, which contain Nan
        if not len(val1_list) == len(val2_list):
            raise UserWarning('val1_list and val2_list must have the same length')
        val1_list_new = []
        val2_list_new = []
        for i in range(len(val1_list)):
            val1 = val1_list[i]
            val2 = val2_list[i]
            if np.isnan(val1):
                continue
            if np.isnan(val2):
                continue
            val1_list_new.append(val1)
            val2_list_new.append(val2)
        if len(val1_list_new) <= 3:
            r, p = np.nan, np.nan
        else:
            if method == 'pearson':
                r, p = stats.pearsonr(val1_list_new, val2_list_new)
            elif method == 'spearman':
                r, p = stats.spearmanr(val1_list_new, val2_list_new)
            elif method == 'kendall':
                r, p = stats.kendalltau(val1_list_new, val2_list_new)
            else:
                raise UserWarning('method must be pearson or spearman or kendall')

        return r, p

    def moving_window_correlation(self, arr1, arr2, window_size: int = 10, date_list: list = None):
        if not len(arr1) == len(arr2):
            raise ValueError('arr1 and arr2 must have the same length')
        if not date_list is None:
            if not len(arr1) == len(date_list):
                raise ValueError('arr and date_list must have the same length')
        if window_size <= 3:
            raise ValueError('window_size must be greater than 3')
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        corr_dict = {}
        for i in range(len(arr1)):
            if i + window_size >= len(arr1):
                break
            if not date_list is None:
                window_name = f'{date_list[i]}-{date_list[i + window_size]}'
            else:
                window_name = f'{i}-{i + window_size}'
            picked_arr1 = arr1[i:i + window_size]
            picked_arr2 = arr2[i:i + window_size]
            r, p = self.nan_correlation(picked_arr1, picked_arr2)
            corr_dict[window_name] = r
        return corr_dict

    def nan_line_fit(self, val1_list, val2_list):
        if not len(val1_list) == len(val2_list):
            raise UserWarning('val1_list and val2_list must have the same length')
        K = KDE_plot()
        val1_list_new = []
        val2_list_new = []
        for i in range(len(val1_list)):
            val1 = val1_list[i]
            val2 = val2_list[i]
            if np.isnan(val1):
                continue
            if np.isnan(val2):
                continue
            val1_list_new.append(val1)
            val2_list_new.append(val2)
        if len(val1_list_new) <= 3:
            a, b, r, p = np.nan, np.nan, np.nan, np.nan
        else:
            a, b, r, p = K.linefit(val1_list_new, val2_list_new)

        return a, b, r, p

    def count_num(self, arr, elements):
        arr = np.array(arr)
        unique, counts = np.unique(arr, return_counts=True)
        count_dic = dict(zip(unique, counts))
        if not elements in count_dic:
            num = 0
        else:
            num = count_dic[elements]
        return num

    def open_path_and_file(self, fpath):
        if sys.platform == "win32":
            os.startfile(fpath)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, fpath])
        # os.system('open {}'.format(fpath))

    def slide_window_correlation(self, x, y, window=15):
        time_series = []
        r_list = []
        for i in range(len(x)):
            if i + window >= len(x):
                continue
            x_new = x[i:i + window]
            y_new = y[i:i + window]
            # x_new = signal.detrend(x_new)
            # y_new = signal.detrend(y_new)
            # plt.plot(x_new)
            # plt.plot(y_new)
            # plt.grid(1)
            # plt.show()
            r, p = stats.pearsonr(x_new, y_new)
            r_list.append(r)
            time_series.append(i)
        time_series = np.array(time_series)
        time_series = time_series + int(window / 2)
        time_series = np.array(time_series, dtype=int)
        r_list = np.array(r_list)

        return time_series, r_list

    def get_df_unique_val_list(self, df, var_name):
        var_list = df[var_name]
        var_list = var_list.dropna()
        var_list = list(set(var_list))
        var_list.sort()
        var_list = tuple(var_list)
        return var_list

    def normalize(self, vals, norm_max=1., norm_min=-1., up_limit=None, bottom_limit=None):
        vals_max = np.nanmax(vals)
        vals_min = np.nanmin(vals)
        norm_list = []
        for v in vals:
            percentile = (v - vals_min) / (vals_max - vals_min)
            norm = percentile * (norm_max - norm_min) + norm_min
            norm_list.append(norm)
        norm_list = np.array(norm_list)
        if up_limit and bottom_limit:
            norm_list[norm_list > up_limit] = np.nan
            norm_list[norm_list < bottom_limit] = np.nan
        return norm_list

    def number_of_days_in_month(self, year=2019, month=2):
        return monthrange(year, month)[1]

    def dic_to_df(self, dic, key_col_str='__key__', col_order=None):
        '''
        :param dic:
        {
        row1:{col1:val1, col2:val2},
        row2:{col1:val1, col2:val2},
        row3:{col1:val1, col2:val2},
        }
        :param key_col_str: define a Dataframe column to store keys of dict
        :return: Dataframe
        '''
        data = []
        columns = []
        index = []
        if col_order == None:
            all_cols = []
            for key in dic:
                vals = dic[key]
                for col in vals:
                    all_cols.append(col)
            all_cols = list(set(all_cols))
            all_cols.sort()
        else:
            all_cols = col_order
        for key in tqdm(dic):
            vals = dic[key]
            if len(vals) == 0:
                continue
            vals_list = []
            col_list = []
            vals_list.append(key)
            col_list.append(key_col_str)
            for col in all_cols:
                if not col in vals:
                    val = np.nan
                else:
                    val = vals[col]
                vals_list.append(val)
                col_list.append(col)
            data.append(vals_list)
            columns.append(col_list)
            index.append(key)
        # df = pd.DataFrame(data=data, columns=columns[0], index=index)
        df = pd.DataFrame(data=data, columns=columns[0])
        return df

    def dic_to_df_different_columns(self, dic, key_col_str='__key__'):
        '''
        :param dic:
        {
        key1:{col1:val1, col2:val2},
        key2:{col1:val1, col2:val2},
        key3:{col1:val1, col2:val2},
        }
        :param key_col_str: define a Dataframe column to store keys of dict
        :return: Dataframe
        '''
        df_list = []
        key_list = []
        for key in tqdm(dic):
            dic_i = dic[key]
            key_list.append(key)
            # print(dic_i)
            new_dic = {k: [] for k in dic_i}
            for k in dic_i:
                new_dic[k].append(dic_i[k])
            df_i = pd.DataFrame.from_dict(data=new_dic)
            df_list.append(df_i)
        # print(len(data[0]))
        # df = pd.DataFrame(data=data, columns=columns, index=index)
        df = pd.concat(df_list)
        df[key_col_str] = key_list
        columns = df.columns.tolist()
        columns.remove(key_col_str)
        columns.insert(0, key_col_str)
        df = df[columns]
        return df

    def df_to_spatial_dic(self, df, col_name, reduce_method=None):
        pix_list = df['pix']
        val_list = df[col_name]
        set_pix_list = set(pix_list)
        if len(val_list) == len(set_pix_list):
            spatial_dic = dict(zip(pix_list, val_list))
            return spatial_dic
        else:
            if reduce_method == None:
                raise ValueError('reduce_method must be provided when there are multiple values for the same pix')
            df_group_dict = self.df_groupby(df,'pix')
            spatial_dic = {}
            for pix in df_group_dict:
                df_i = df_group_dict[pix]
                vals = df_i[col_name].tolist()
                reduced_vals = reduce_method(vals)
                spatial_dic[pix] = reduced_vals
            return spatial_dic

    def is_unique_key_in_df(self, df, unique_key):
        len_df = len(df)
        unique_key_list = self.get_df_unique_val_list(df, unique_key)
        len_unique_key = len(unique_key_list)
        if len_df == len_unique_key:
            return True
        else:
            return False

    def add_dic_to_df(self, df, dic, unique_key):
        if not self.is_unique_key_in_df(df, unique_key):
            raise UserWarning(f'{unique_key} is not a unique key')
        all_val_list = []
        all_col_list = []
        for i, row in df.iterrows():
            unique_key_ = row[unique_key]
            if not unique_key_ in dic:
                dic_i = np.nan
            else:
                dic_i = dic[unique_key_]
            col_list = []
            val_list = []
            for col in dic_i:
                val = dic_i[col]
                col_list.append(col)
                val_list.append(val)
            all_val_list.append(val_list)
            all_col_list.append(col_list)
            # val_list.append(val)
        all_val_list = np.array(all_val_list)
        all_col_list = np.array(all_col_list)
        all_val_list_T = all_val_list.T
        all_col_list_T = all_col_list.T
        for i in range(len(all_col_list_T)):
            df[all_col_list_T[i][0]] = all_val_list_T[i]
        return df

    def spatial_dics_to_df(self, spatial_dic_all):
        unique_keys = []
        for var_name in spatial_dic_all:
            dic_i = spatial_dic_all[var_name]
            for key in dic_i:
                unique_keys.append(key)
        unique_keys = list(set(unique_keys))
        unique_keys.sort()
        dic_all_transform = {}
        for key in unique_keys:
            dic_all_transform[key] = {}
        var_name_list = []
        for var_name in spatial_dic_all:
            var_name_list.append(var_name)
            dic_i = spatial_dic_all[var_name]
            for key in dic_i:
                val = dic_i[key]
                dic_all_transform[key].update({var_name: val})
        df = self.dic_to_df(dic_all_transform, 'pix')
        valid_var_name_list = []
        not_valid_var_name_list = []
        for var_name in var_name_list:
            if var_name in df.columns:
                valid_var_name_list.append(var_name)
            else:
                not_valid_var_name_list.append(var_name)
        df = df.dropna(how='all', subset=valid_var_name_list)
        not_valid_var_name_list.sort()
        for var_name in not_valid_var_name_list:
            df[var_name] = np.nan
        return df

    def add_spatial_dic_to_df(self, df, dic, key_name):
        val_list = []
        for i, row in df.iterrows():
            pix = row['pix']
            if not pix in dic:
                val = np.nan
            else:
                val = dic[pix]
            val_list.append(val)
        df[key_name] = val_list
        return df

    def join_df_list(self, df, df_list, key):
        # key must be unique
        if len(df) == 0:
            df = df_list[0]
            df_list = df_list[1:]
        if not self.is_unique_key_in_df(df, key):
            raise UserWarning(f'{key} in not an unique key')
        for df_i in df_list:
            df = df.join(df_i.set_index(key), on=key)
        return df

    def df_to_dic(self, df, key_str='__key__'):
        '''
        :param df: Dataframe
        :param key_str: Unique column name
        :return:
        '''
        columns = df.columns
        dic = {}
        for i, row in df.iterrows():
            key = row[key_str]
            if key in dic:
                raise UserWarning(f'"{key_str}" is not unique\ni.e. "{key}" is not unique')
            dic_i = {}
            for col in columns:
                val = row[col]
                dic_i[col] = val
            dic[key] = dic_i
        return dic

    def df_to_dic_non_unique_key(self, df, non_unique_colname, unique_colname):
        df_to_dict = {}
        col_name_list = self.get_df_unique_val_list(df, non_unique_colname)
        for key in tqdm(col_name_list, desc='df_to_dic_non_unique_key'):
            df_col_name = df[df[non_unique_colname] == key]
            df_year_dict = self.df_to_dic(df_col_name, unique_colname)
            df_to_dict[key] = df_year_dict
        return df_to_dict

    def add_pix_to_df_from_lon_lat(self, df):
        lon_list = []
        lat_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            lon = row['lon']
            lat = row['lat']
            lon = float(lon)
            lat = float(lat)
            lon_list.append(lon)
            lat_list.append(lat)
        pix_list = DIC_and_TIF().lon_lat_to_pix(lon_list, lat_list)
        df['pix'] = pix_list
        return df

    def add_lon_lat_to_df(self, df, D=None):
        lon_list = []
        lat_list = []
        if D is None:
            D = DIC_and_TIF()
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            lon, lat = D.pix_to_lon_lat(pix)
            lon_list.append(lon)
            lat_list.append(lat)
        df['lon'] = lon_list
        df['lat'] = lat_list

        return df

    def rename_dataframe_columns(self, df, old_name, new_name):
        new_name_dic = {
            old_name: new_name,
        }
        df = pd.DataFrame(df)
        df = df.rename(columns=new_name_dic)
        return df

    def change_df_col_dtype(self, df, col_name, dtype):
        series = df[col_name].tolist()
        series_dtype = np.array(series, dtype)
        df[col_name] = series_dtype
        return df

    def shasum(self, fpath, isprint=True):
        fr = open(fpath, 'rb')
        content_bytes = fr.read()
        readable_hash = hashlib.sha256(content_bytes).hexdigest()
        if isprint:
            print(fpath)
            print(readable_hash)
            print('--' * 8)
        return readable_hash

    def gen_time_stamps(self,
                        start=datetime.datetime(2020, 1, 1),
                        end=datetime.datetime(2020, 1, 2),
                        delta=datetime.timedelta(hours=0.5)
                        ):

        def daterange_i(start_date, end_date):
            while start_date < end_date:
                yield start_date
                start_date += delta

        half_time_stamps = []
        for tt in daterange_i(start, end):
            half_time_stamps.append(tt)
        return half_time_stamps

    def convert_val_to_time_series_obj(self, data, time_stamp_list, name='value'):
        index = pd.DatetimeIndex(time_stamp_list)
        time_series = pd.Series(data=data, index=index, name=name)
        return time_series

    def drop_n_std(self, vals, n=1):
        vals = np.array(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up = mean + n * std
        down = mean - n * std
        vals[vals > up] = np.nan
        vals[vals < down] = np.nan
        return vals

    def monthly_vals_to_annual_val(self, monthly_vals, grow_season=None, method='mean'):
        '''
        from
        [1,2,3....,46,47,48]
        to
        ([1,2,...11,12]
        [13,14,...23,24]
        .
        .
        [37,38,...47,48])
        to
        [1,2,3,4]
        :param monthly_vals:
        :param grow_season: default is 1-12
        :return:
        todo: add axis to np.nanmean()
        '''
        if grow_season == None:
            grow_season = list(range(12))
        else:
            grow_season = np.array(grow_season, dtype=int)
            grow_season = grow_season - 1
            if grow_season[0] < 0:
                raise UserWarning(f'Error grow_season:{grow_season}')
        monthly_vals = np.array(monthly_vals)
        monthly_vals_reshape = np.reshape(monthly_vals, (-1, 12))
        monthly_vals_reshape_T = monthly_vals_reshape.T
        monthly_vals_reshape_T_gs = Tools().pick_vals_from_1darray(monthly_vals_reshape_T, grow_season)
        monthly_vals_reshape_gs = monthly_vals_reshape_T_gs.T
        annual_val_list = []
        for one_year_vals in monthly_vals_reshape_gs:
            if self.is_all_nan(one_year_vals):
                annual_val = np.nan
                if method == 'array':
                    annual_val = np.array(one_year_vals)
            else:
                if method == 'mean':
                    annual_val = np.nanmean(one_year_vals)
                elif method == 'max':
                    annual_val = np.nanmax(one_year_vals)
                elif method == 'min':
                    annual_val = np.nanmin(one_year_vals)
                elif method == 'array':
                    annual_val = np.array(one_year_vals)
                elif method == 'sum':
                    annual_val = np.nansum(one_year_vals)
                else:
                    raise UserWarning(f'method:{method} error')
            annual_val_list.append(annual_val)
        annual_val_list = np.array(annual_val_list)

        return annual_val_list

    def monthly_to_annual_with_datetime_obj(self, vals, date_range, grow_season: list, method='mean'):
        if grow_season == None:
            grow_season = list(range(12))
        else:
            grow_season = np.array(grow_season, dtype=int)
            if grow_season[0] < 0:
                raise UserWarning(f'Error grow_season:{grow_season}')
        year_list = []
        for date in date_range:
            year = date.year
            if year not in year_list:
                year_list.append(year)
        year_list.sort()
        vals_dict = dict(zip(date_range, vals))
        annual_dict = {y: [] for y in year_list}
        for date in vals_dict:
            year = date.year
            mon = date.month
            if not mon in grow_season:
                continue
            annual_dict[year].append(vals_dict[date])

        vals_gs_annual = []
        for year in year_list:
            one_year_vals = annual_dict[year]
            if method == 'mean':
                annual_val = np.nanmean(one_year_vals)
            elif method == 'max':
                annual_val = np.nanmax(one_year_vals)
            elif method == 'min':
                annual_val = np.nanmin(one_year_vals)
            elif method == 'array':
                annual_val = np.array(one_year_vals)
            elif method == 'sum':
                annual_val = np.nansum(one_year_vals)
            else:
                raise UserWarning(f'method:{method} error')
            vals_gs_annual.append(annual_val)
        vals_gs_annual = np.array(vals_gs_annual)
        year_date_obj_list = [datetime.datetime(y, 1, 1) for y in year_list]
        return year_date_obj_list, vals_gs_annual

    def monthly_vals_to_date_dic(self, monthly_val, start_year, end_year):
        date_list = []
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                date = f'{y}{m:02d}'
                date_list.append(date)
        len_vals = len(monthly_val)
        len_date_list = len(date_list)
        if not len_vals == len_date_list:
            raise UserWarning('Date list is not matching value list')
        dic = dict(zip(date_list, monthly_val))

        return dic

    def month_index_to_date_obj(self, month_index, init_date_obj):
        year = init_date_obj.year
        month = init_date_obj.month
        end_month = month + month_index
        end_year = year + int(end_month / 12)
        end_month = end_month % 12
        if end_month == 0:
            end_month = 12
            end_year -= 1
        end_date_obj = datetime.datetime(end_year, end_month, 1)
        return end_date_obj

    def pick_gs_monthly_data(self, vals, gs):
        if len(vals) % 12 != 0:
            raise ValueError('the lenth of vals is not a multiple of 12')
        vals_year_number = len(vals) // 12
        start_year = 1982
        end_year = start_year + vals_year_number - 1
        vals_dict = self.monthly_vals_to_date_dic(vals, start_year, end_year)
        picked_dates = []
        for y in range(start_year, end_year + 1):
            for m in gs:
                picked_dates.append(f'{y:04d}{m:02d}')
        gs_vals = []
        for date in picked_dates:
            val = vals_dict[date]
            gs_vals.append(val)
        gs_vals = np.array(gs_vals)
        return gs_vals

    def unzip(self, zipfolder, outdir):
        # zipfolder = join(self.datadir,'zips')
        # outdir = join(self.datadir,'unzip')
        self.mkdir(outdir)
        for f in tqdm(self.listdir(zipfolder)):
            outdir_i = join(outdir, f.replace('.zip', ''))
            self.mkdir(outdir_i)
            fpath = join(zipfolder, f)
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(outdir_i)
            zip_ref.close()

    def intersect(self, x, y):
        x = set(x)
        y = set(y)
        z = x.intersection(y)
        return z

    def is_all_nan(self, vals, nodata=np.nan):
        if type(vals) == float:
            return vals == nodata
        vals = np.array(vals)
        if np.isnan(nodata):
            isnan_list = np.isnan(vals)
            isnan_list_set = set(isnan_list)
            isnan_list_set = list(isnan_list_set)
            if len(isnan_list_set) == 1:
                if isnan_list_set[0] == True:
                    return True
                else:
                    return False
            else:
                return False
        else:
            isnan_list = vals[vals==nodata]
            if len(isnan_list) == len(vals):
                return True
            else:
                return False

    def reverse_dic(self, dic):
        items = dic.items()
        df = pd.DataFrame.from_dict(data=items)
        unique_vals = self.get_df_unique_val_list(df, 1)
        dic_reverse = {}
        for v in unique_vals:
            dic_reverse[v] = []
        for key in dic:
            val = dic[key]
            dic_reverse[val].append(key)
        return dic_reverse

    def combine_df_columns(self, df, combine_col, new_col_name, method='mean'):
        combined_vals_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            vals_list = []
            for cols in combine_col:
                vals = row[cols]
                if type(vals) == float:
                    continue
                vals = np.array(vals)
                vals_list.append(vals)
            vals_list = np.array(vals_list)
            vals_list_mean = np.nanmean(vals_list, axis=0)
            combined_vals_list.append(vals_list_mean)
        df[new_col_name] = combined_vals_list
        return df

    def get_vals_std_up_down(self, vals):
        vals = np.array(vals)
        std = np.nanstd(vals)
        mean = np.nanmean(vals)
        up = mean + std
        down = mean - std
        return up, down

    def vals_to_time_sereis_annual(self, vals, yearlist: list = None, start_year: int = None):
        if yearlist == start_year == None:
            raise UserWarning('You must input at least one parameter of "year list" or "start year"')
        if start_year != None:
            yearlist = list(range(start_year, start_year + len(vals)))
            xval_ts = pd.Series(vals, index=yearlist)
            return xval_ts
        if yearlist != None:
            xval_ts = pd.Series(vals, index=yearlist)
            return xval_ts

    def hex_color_to_rgb(self, hex_color):
        '''
        auto generated by github copilot
        :param hex_color:
        :return:
        '''
        hex_color = hex_color.lstrip('#')
        rgb_list = list(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        rgb_list = [i / 255. for i in rgb_list]
        rgb_list.append(1)
        return tuple(rgb_list)

    def cross_list(self, *args, is_unique=False):
        # auto generate by github copilot
        cross_list = list(itertools.product(*args))
        cross_list = [x for x in cross_list if x[0] != x[1]]
        cross_list_unique = []
        for x in cross_list:
            x_list = list(x)
            x_list.sort()
            cross_list_unique.append(tuple(x_list))
        cross_list_unique = list(set(cross_list_unique))
        if is_unique:
            return cross_list_unique
        else:
            return cross_list

    def cross_select_dataframe(self, df, *args, is_unique=False):
        if len(args) == 1:
            arg = args[0]
            unique_value = self.get_df_unique_val_list(df, arg)
            cross_df_dict = {}
            for uv in unique_value:
                df_temp = df[df[arg] == uv]
                cross_df_dict[uv] = df_temp
            return cross_df_dict
        else:
            unique_value_list = []
            for arg in args:
                unique_value = self.get_df_unique_val_list(df, arg)
                unique_value_list.append(unique_value)
            cross_list = self.cross_list(*unique_value_list, is_unique=is_unique)
            cross_df_dict = {}
            for x in cross_list:
                df_copy = copy.copy(df)
                for xi in range(len(x)):
                    df_copy = df_copy[df_copy[args[xi]] == x[xi]]
                cross_df_dict[x] = df_copy
            return cross_df_dict

    def resample_nan(self, array, target_res, original_res, nan_value=-999999):
        array = array.astype(float)
        array[array == nan_value] = np.nan
        window_len = int(target_res / original_res)
        array_row_new = len(array) / window_len
        array_col_new = len(array[0]) / window_len
        array_row_new = int(array_row_new)
        array_col_new = int(array_col_new)
        matrix = []
        for i in range(array_row_new):
            row = array[i * window_len:(i + 1) * window_len]
            temp = []
            for j in range(array_col_new):
                row_T = row.T
                col_T = row_T[j * window_len:(j + 1) * window_len]
                matrix_i = col_T.T
                ## count the number of nan
                matrix_i_flat = matrix_i.flatten()
                nan_flag = np.isnan(matrix_i_flat)
                nan_number = self.count_num(nan_flag, True)
                nan_ratio = nan_number / len(matrix_i_flat)
                if nan_ratio > 0.5:
                    mean_matrix_i = np.nan
                else:
                    mean_matrix_i = np.nansum(matrix_i) / len(matrix_i_flat)
                    # temp.append(np.nanmean(mean_matrix_i))
                temp.append(mean_matrix_i)
            # print(temp)
            temp = np.array(temp)
            matrix.append(temp)
        matrix = np.array(matrix)
        # matrix[matrix == 0] = np.nan
        return matrix

    def cmap_blend(self, color_list, as_cmap=True, n_colors=6):
        # color_list = ['r', 'g', 'b']
        cmap = sns.blend_palette(color_list, as_cmap=as_cmap, n_colors=n_colors)
        return cmap

    def cmap_diverging(self, start_color_hue, end_color_hue, saturation=100, lightness=40):
        cmap = sns.diverging_palette(0, 120, s=saturation, l=40, as_cmap=True)
        return cmap

    def get_max_key_from_dict(self, input_dict):
        max_key = None
        max_value = -np.inf
        for key in input_dict:
            value = input_dict[key]
            if value > max_value:
                max_key = key
                max_value = value
        return max_key

    def days_number_of_year(self, year):
        if year % 4 == 0:
            if year % 100 == 0:
                if year % 400 == 0:
                    return 366
                else:
                    return 365
            else:
                return 366
        else:
            return 365

    def count_days_of_two_dates(self, date1, date2):
        date1 = datetime.datetime.strptime(date1, '%Y-%m-%d')
        date2 = datetime.datetime.strptime(date2, '%Y-%m-%d')
        delta = date2 - date1
        return delta.days

    def nc_to_tif(self, fname, var_name, outdir):
        try:
            ncin = Dataset(fname, 'r')
            print(ncin.variables.keys())

        except:
            raise UserWarning('File not supported: ' + fname)
        try:
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
        except:
            try:
                lat = ncin.variables['latitude'][:]
                lon = ncin.variables['longitude'][:]
            except:
                try:
                    lat = ncin.variables['lat_FULL'][:]
                    lon = ncin.variables['lon_FULL'][:]
                except:
                    raise UserWarning('lat or lon not found')
        shape = np.shape(lat)
        try:
            time = ncin.variables['time_counter'][:]
            basetime_str = ncin.variables['time_counter'].units
        except:
            time = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units

        basetime_unit = basetime_str.split('since')[0]
        basetime_unit = basetime_unit.strip()
        print(basetime_unit)
        print(basetime_str)
        if basetime_unit == 'days':
            timedelta_unit = 'days'
        elif basetime_unit == 'years':
            timedelta_unit = 'years'
        elif basetime_unit == 'month':
            timedelta_unit = 'month'
        elif basetime_unit == 'months':
            timedelta_unit = 'month'
        elif basetime_unit == 'seconds':
            timedelta_unit = 'seconds'
        elif basetime_unit == 'hours':
            timedelta_unit = 'hours'
        else:
            raise Exception('basetime unit not supported')
        basetime = basetime_str.strip(f'{timedelta_unit} since ')
        try:
            basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
        except:
            try:
                basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    try:
                        basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M')
                    except:
                        try:
                            basetime = datetime.datetime.strptime(basetime, '%Y-%m')
                        except:
                            raise UserWarning('basetime format not supported')
        data = ncin.variables[var_name]
        if len(shape) == 2:
            xx, yy = lon, lat
        else:
            xx, yy = np.meshgrid(lon, lat)
        for time_i in tqdm(range(len(time))):
            if basetime_unit == 'days':
                date = basetime + datetime.timedelta(days=int(time[time_i]))
            elif basetime_unit == 'years':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                date2 = f'{int(base_year + time[time_i])}-01-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'month' or basetime_unit == 'months':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                base_month = basetime.month
                date2 = f'{int(base_year + time[time_i] // 12)}-{int(base_month + time[time_i] % 12)}-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'seconds':
                date = basetime + datetime.timedelta(seconds=int(time[time_i]))
            elif basetime_unit == 'hours':
                date = basetime + datetime.timedelta(hours=int(time[time_i]))
            else:
                raise Exception('basetime unit not supported')
            time_str = time[time_i]
            mon = date.month
            year = date.year
            day = date.day
            outf_name = f'{year}{mon:02d}{day:02d}.tif'
            outpath = join(outdir, outf_name)
            if isfile(outpath):
                continue
            arr = data[time_i]
            arr = np.array(arr)
            lon_list = xx.flatten()
            lat_list = yy.flatten()
            val_list = arr.flatten()
            lon_list[lon_list > 180] = lon_list[lon_list > 180] - 360
            df = pd.DataFrame()
            df['lon'] = lon_list
            df['lat'] = lat_list
            df['val'] = val_list
            lon_list_new = df['lon'].tolist()
            lat_list_new = df['lat'].tolist()
            val_list_new = df['val'].tolist()
            DIC_and_TIF().lon_lat_val_to_tif(lon_list_new, lat_list_new, val_list_new, outpath)

    def uncertainty_err(self, vals):
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up, bottom = stats.t.interval(0.95, len(vals) - 1, loc=mean, scale=std / np.sqrt(len(vals)))
        err = mean - bottom
        return err, up, bottom

    def uncertainty_err_2d(self, vals, axis=0):
        vals = np.array(vals)
        if axis == 0:
            vals_T = vals.T
            vals_err = []
            for val in tqdm(vals_T, desc='uncertainty'):
                err, _, _ = self.uncertainty_err(val)
                vals_err.append(err)
            vals_err = np.array(vals_err)
        elif axis == 1:
            vals_T = vals
            vals_err = []
            for val in tqdm(vals_T, desc='uncertainty'):
                err, _, _ = self.uncertainty_err(val)
                vals_err.append(err)
            vals_err = np.array(vals_err)
        else:
            raise Exception('axis must be 0 or 1')
        return vals_err

    def df_bin(self, df, col, bins):
        df_copy = df.copy()
        df_copy[f'{col}_bins'] = pd.cut(df[col], bins=bins)
        df_group = df_copy.groupby([f'{col}_bins'])
        bins_name = df_group.groups.keys()
        bins_name_list = list(bins_name)
        bins_list_str = [str(i) for i in bins_name_list]
        # for name,df_group_i in df_group:
        #     vals = df_group_i[col].tolist()
        #     mean = np.nanmean(vals)
        #     err,_,_ = self.uncertainty_err(SM)
        #     # x_list.append(name)
        #     y_list.append(mean)
        #     err_list.append(err)
        return df_group, bins_list_str

    def df_bin_2d(self,df,val_col_name,col_name_x,col_name_y,bin_x,bin_y,method=np.nanmean,round_x=2,round_y=2):
        df_group_y, _ = self.df_bin(df, col_name_y, bin_y)
        matrix_dict = {}
        y_ticks_list = []
        x_ticks_dict = {}
        flag1 = 0
        for name_y, df_group_y_i in df_group_y:
            matrix_i = []
            y_ticks = (name_y[0].left + name_y[0].right) / 2
            y_ticks = np.round(y_ticks, round_y)
            y_ticks_list.append(y_ticks)
            df_group_x, _ = self.df_bin(df_group_y_i, col_name_x, bin_x)
            flag2 = 0
            for name_x, df_group_x_i in df_group_x:
                vals = df_group_x_i[val_col_name].tolist()
                rt_mean = method(vals)
                matrix_i.append(rt_mean)
                x_ticks = (name_x[0].left + name_x[0].right) / 2
                x_ticks = np.round(x_ticks, round_x)
                x_ticks_dict[x_ticks] = 0
                key = (flag1, flag2)
                matrix_dict[key] = rt_mean
                flag2 += 1
            flag1 += 1
        x_ticks_list = list(x_ticks_dict.keys())
        x_ticks_list.sort()
        return matrix_dict,x_ticks_list,y_ticks_list

    def plot_df_bin_2d_matrix(self,matrix_dict,vmin,vmax,x_ticks_list,y_ticks_list,cmap='RdBu',
                              is_only_return_matrix=False):
        keys = list(matrix_dict.keys())
        r_list = []
        c_list = []
        for r, c in keys:
            r_list.append(r)
            c_list.append(c)
        r_list = set(r_list)
        c_list = set(c_list)

        row = len(r_list)
        col = len(c_list)
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in matrix_dict:
                    val_pix = matrix_dict[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        matrix = np.array(spatial, dtype=float)
        matrix = matrix[::-1]
        if is_only_return_matrix:
            return matrix
        plt.imshow(matrix,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.xticks(range(len(c_list)), x_ticks_list)
        plt.yticks(range(len(r_list)), y_ticks_list[::-1])

    def plot_df_bin_2d_scatter(self,matrix_dict,vmin,vmax,x_ticks_list,y_ticks_list,cmap='RdBu',
                               is_x_quantile=False,
                               is_y_quantile=False,marker='s',**kwargs):
        for y,x in matrix_dict.keys():
            val = matrix_dict[(y,x)]
            if is_x_quantile:
                x_pos = x / (len(x_ticks_list) - 1) * 100
            else:
                x_pos = x_ticks_list[x]
            if is_y_quantile:
                y_pos = y / (len(y_ticks_list) - 1) * 100
            else:
                y_pos = y_ticks_list[y]
            plt.scatter(x_pos,y_pos,c=val,vmin=vmin,vmax=vmax,cmap=cmap,linewidths=0,marker=marker,**kwargs)

    def ANOVA_test(self, *args, method: str):
        if method == 'f_oneway':
            return stats.f_oneway(*args)
        elif method == 'ks':
            if len(args) == 2:
                return stats.ks_2samp(*args)
            else:
                raise ValueError('KS test args length must be 2')

    def drop_df_index(self, df):
        df = df.reset_index(drop=True)
        return df

    def date_to_DOY(self, date_list):
        '''
        :param date_list: list of datetime objects
        :return: list of DOY
        '''
        start_year = date_list[0].year
        start_date = datetime.datetime(start_year, 1, 1)
        date_delta = date_list - start_date + datetime.timedelta(days=1)
        DOY = [date.days for date in date_delta]
        return DOY

    def gen_colors(self, color_list_number, palette='Spectral'):
        color_list = sns.color_palette(palette, color_list_number)
        return color_list

    def del_columns(self, df, columns: list):
        df = df.drop(columns=columns, axis=1)
        return df

    def bootstrap_data(self, data: pd.DataFrame, n: int, ratio: float):
        if not 0 < ratio < 1:
            raise ValueError('ratio must be between 0 and 1')
        data_len = len(data)
        for i in range(n):
            bootstraped_data = data.sample(n=int(data_len * ratio), replace=True)
            yield bootstraped_data

    def set_Chinese_available_fonts(self):
        from matplotlib.font_manager import FontManager
        import subprocess
        fm = FontManager()
        mat_fonts = set(f.name for f in fm.ttflist)
        # print(mat_fonts)
        output = subprocess.check_output(
            'fc-list :lang=zh -f "%{family}\n"', shell=True)  # 获取字体列表
        output = output.decode('utf-8')

        zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n'))
        available = mat_fonts & zh_fonts
        print('*' * 10, '可用的字体', '*' * 10)
        for f in available:
            print(f)
        plt.rcParams["font.sans-serif"] = list(available)[0]

    def lag_correlation(self, earlier, later, lag, method='pearson'):
        '''
        earlier value can affect later value
        e.g. earlier SPEI, later NDVI
        :return: correlation
        '''
        if lag == 0:
            r, p = self.nan_correlation(earlier, later, method)
        else:
            later = later[lag:]
            earlier = earlier[:-lag]
            r, p = self.nan_correlation(earlier, later, method)
        return r, p

    def df_groupby(self, df, col):
        assert col in df.columns
        df_groupby = df.groupby(col, observed=True)
        df_group_dict = {}
        for name, group in df_groupby:
            df_group_dict[name] = group
        return df_group_dict

    def color_map_choice(self):
        color_map_list = [
            'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap',
            'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd',
            'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2',
            'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',
            'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r',
            'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral',
            'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd',
            'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg',
            'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper',
            'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r',
            'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar',
            'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
            'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r',
            'icefire', 'icefire_r', 'inferno', 'inferno_r', 'magma', 'magma_r', 'mako', 'mako_r',
            'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism',
            'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r',
            'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
            'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted',
            'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
        ]
        color_number = 0
        for c in color_map_list:
            if c.endswith('_r'):
                continue
            color_number += 1
        fig = plt.figure()
        flag = 0
        for c in color_map_list:
            if c.endswith('_r'):
                continue
            flag += 1
            ax = fig.add_subplot(10, 9, flag)
            print(c)
            pal = sns.color_palette(c, n_colors=11)
            n = len(pal)
            ax.imshow(np.arange(n).reshape(1, n),
                      cmap=mpl.colors.ListedColormap(list(pal)),
                      interpolation="nearest", aspect='auto')
            plt.xticks([])
            plt.yticks([])
            plt.title(c)
        plt.tight_layout()
        plt.show()

    def dict_zip(self, keys, vals, allow_duplicates_keys=False):
        assert len(keys) == len(vals)
        assert len(keys) > 0
        if not allow_duplicates_keys:
            assert len(set(keys)) == len(keys)
        return dict(zip(keys, vals))

    def df_drop_duplicates(self, df, cols:list,**kwargs):
        df = df.drop_duplicates(subset=cols,**kwargs)
        return df

    def sort_dict_by_value(self, input_dict,descending=True):
        return dict(sorted(input_dict.items(), key=lambda item: item[1], reverse=descending))

    def date_range(self,start_date_obj,end_date_obj):
        date_range = []
        for i in range((end_date_obj - start_date_obj).days):
            date_range.append(start_date_obj + datetime.timedelta(days=i))
        return date_range

    def pick_min_max_key_val_from_dict(self, dic,min_or_max):
        key_list = []
        val_list = []
        for key in dic:
            val = dic[key]
            key_list.append(key)
            val_list.append(val)
        if min_or_max == 'min':
            max_min_val_index = np.nanargmin(val_list)
            max_min_key = key_list[max_min_val_index]
        elif min_or_max == 'max':
            max_min_val_index = np.nanargmax(val_list)
            max_min_key = key_list[max_min_val_index]
        else:
            raise UserWarning('min_or_max must be "min" or "max"')
        return max_min_key

    def listdir_full(self, fdir, extension=None):
        fdir = Path(fdir)
        f_list = []
        for f in self.listdir(fdir):
            if extension:
                assert extension.startswith('.')
                if not f.endswith(extension):
                    continue
            fpath = fdir / f
            f_list.append(fpath)
        return f_list

    def doy_to_month(self,doy):
        '''
        :param doy: day of year
        :return: month
        '''
        base = datetime.datetime(2000,1,1)
        time_delta = datetime.timedelta(int(doy))
        date = base + time_delta
        month = date.month
        day = date.day
        if day > 15:
            month = month + 1
        if month >= 12:
            month = 12
        return month

    def get_GS_vals(self, vals:np.ndarray, onset_doy:float, offset_doy:float):
        # todo: check carefully
        onset_mon = self.doy_to_month(onset_doy)
        offset_mon = self.doy_to_month(offset_doy)
        assert len(vals) % 12 == 0
        vals_reshape = vals.reshape(-1, 12)

        if offset_mon > onset_mon:
            vals_reshape_gs = vals_reshape[:, onset_mon - 1:offset_mon - 1]
            return vals_reshape_gs

        elif offset_mon < onset_mon:
            gs_len = 12 - onset_mon + 1 + offset_mon

            first_year = vals_reshape[0][onset_mon - 1:].tolist()
            last_year = vals_reshape[-1][:onset_mon - 1].tolist()
            middle_year = vals_reshape[1:-1].flatten().tolist()

            vals_cut = first_year + middle_year + last_year
            vals_cut = np.array(vals_cut)
            vals_cut_reshape = vals_cut.reshape(-1, 12)
            vals_cut_reshape_gs = vals_cut_reshape[:, :gs_len]
            # vals_cut_reshape_gs_annual_mean = method(vals_cut_reshape_gs, axis=1)
            return vals_cut_reshape_gs

        elif offset_mon == onset_mon:
            vals_reshape_gs = vals_reshape[:,onset_mon-1]
            vals_reshape_gs = vals_reshape_gs.flatten().reshape(-1,1)
            return vals_reshape_gs
        else:
            print(offset_mon, onset_mon)
            print('errrrr')
            raise ValueError

        pass

    def reproject_coordinates(self,x,y,src_crs,dst_crs):
        # Create a transformer object
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

        # Reproject the coordinates
        new_x, new_y = transformer.transform(x, y)

        return new_x, new_y

    def shasum_string(self, input_string):
        # Create a SHA-256 hash object
        sha256_hash = hashlib.sha256()

        # Update the hash object with the bytes of the input string
        sha256_hash.update(input_string.encode('utf-8'))

        # Get the hexadecimal representation of the hash
        hex_dig = sha256_hash.hexdigest()

        return hex_dig


    def df_to_gpkg(self, df, outGPKGfn, lon_col='lon', lat_col='lat', layer_name='points', wkt=None):
        '''
        Save a pandas DataFrame to a GeoPackage point layer.

        DataFrame must include longitude/latitude columns and any number of attribute columns.
        '''

        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a pandas.DataFrame')
        if len(df) == 0:
            raise ValueError('df is empty')
        if lon_col not in df.columns or lat_col not in df.columns:
            raise KeyError(f'Missing required coordinate columns: {lon_col}, {lat_col}')
        if layer_name is None or str(layer_name).strip() == '':
            raise ValueError('layer_name cannot be empty')

        if outGPKGfn.endswith('.gpkg'):
            out_path = outGPKGfn
        else:
            out_path = outGPKGfn + '.gpkg'

        gpkg_driver = ogr.GetDriverByName('GPKG')
        if gpkg_driver is None:
            raise RuntimeError('GPKG driver is not available in your GDAL/OGR build')
        if os.path.exists(out_path):
            gpkg_driver.DeleteDataSource(out_path)

        out_data_source = gpkg_driver.CreateDataSource(out_path)
        if out_data_source is None:
            raise RuntimeError(f'Failed to create GeoPackage: {out_path}')

        spatial_ref = osr.SpatialReference()
        if wkt is None:
            wkt = DIC_and_TIF().wkt_84()
        spatial_ref.ImportFromWkt(wkt)

        out_layer = out_data_source.CreateLayer(str(layer_name), srs=spatial_ref, geom_type=ogr.wkbPoint)
        if out_layer is None:
            out_data_source = None
            raise RuntimeError(f'Failed to create layer: {layer_name}')

        attr_cols = [col for col in df.columns if col not in [lon_col, lat_col]]
        field_type_dict = {
            'float': ogr.OFTReal,
            'int': ogr.OFTInteger64,
            'str': ogr.OFTString,
            'bool': ogr.OFTInteger
        }

        col_type_dict = {}
        for col in attr_cols:
            series = df[col].dropna()
            if len(series) == 0:
                value_type_name = 'str'
            else:
                value = series.iloc[0]
                if isinstance(value, (bool, np.bool_)):
                    value_type_name = 'bool'
                elif isinstance(value, (int, np.integer)):
                    value_type_name = 'int'
                elif isinstance(value, (float, np.floating)):
                    value_type_name = 'float'
                else:
                    value_type_name = 'str'

            col_type_dict[col] = value_type_name
            out_layer.CreateField(ogr.FieldDefn(str(col), field_type_dict[value_type_name]))

        feature_defn = out_layer.GetLayerDefn()
        for _, row in df.iterrows():
            lon = row[lon_col]
            lat = row[lat_col]
            if pd.isna(lon) or pd.isna(lat):
                continue

            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(float(lon), float(lat))

            out_feature = ogr.Feature(feature_defn)
            out_feature.SetGeometry(point)

            for col in attr_cols:
                value = row[col]
                if pd.isna(value):
                    continue
                value_type_name = col_type_dict[col]
                if value_type_name == 'int':
                    out_feature.SetField(str(col), int(value))
                elif value_type_name == 'float':
                    out_feature.SetField(str(col), float(value))
                elif value_type_name == 'bool':
                    out_feature.SetField(str(col), int(bool(value)))
                else:
                    out_feature.SetField(str(col), str(value))

            out_layer.CreateFeature(out_feature)
            out_feature = None

        out_data_source = None
        return out_path


    def download_simple(self,url,outf):
        http = urllib3.PoolManager()
        r = http.request('GET', url, preload_content=False)
        body = r.read()
        with open(outf, 'wb') as f:
            f.write(body)

    def download_file(self,url, outf, num_threads=4, allow_tqdm=True,
                      custom_dns=None, username=None, password=None
                      ):
        '''
        This function is used to download large files in parallel.
        username and password are used for authentication [optional]
        custom_dns is used for custom dns server [optional]

        :param url: download url
        :param outf: output file path
        :param num_threads: concurrent download threads
        :param allow_tqdm: show progress bar
        :param custom_dns: custom dns server
        :param username: uesrname
        :param password: password
        :return: None
        '''

        outf = Path(outf)
        outf_name = outf.name
        domain = url.split('/')[2]

        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36'}
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        if not custom_dns is None:
            url = url.replace(domain,self.get_ip_from_custom_dns(domain,custom_dns))
            headers['Host'] = domain


        session = requests.Session()
        if username:
            session.auth = HTTPBasicAuth(username, password)
        response = session.get(url, headers=headers, stream=True, allow_redirects=True,verify=False)

        file_size = int(response.headers.get('content-length', 0))

        response_code = response.status_code
        if response_code != 200:
            raise ValueError(f"File not found at URL: {url}\nResponse code: {response_code}")

        if file_size == 0:
            raise ValueError(f"File size is 0 for URL: {url}")
        if allow_tqdm:
            progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, desc=outf_name)
        else:
            progress_bar = None
        lock = Lock()

        with open(outf, 'wb') as f:
            f.write(b'\0' * file_size)

        def download_chunk(start, end):
            headers = {'Range': f'bytes={start}-{end}',
                       'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36'
                       }
            if not custom_dns is None:
                headers['Host'] = domain
            response = session.get(url, headers=headers, stream=True,verify=False)

            with open(outf, 'rb+') as f:
                f.seek(start)
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
                        with lock:
                            if progress_bar:
                                progress_bar.update(len(chunk))

        chunk_size = file_size // num_threads
        threads = []

        for i in range(num_threads):
            start = i * chunk_size
            end = file_size - 1 if i == num_threads - 1 else (i + 1) * chunk_size - 1

            t = Thread(target=download_chunk, args=(start, end))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        if progress_bar:
            progress_bar.close()

    def get_ip_from_custom_dns(self,domain,dns_server='1.1.1.1'):
        resolver = dns.resolver.Resolver()
        resolver.nameservers = [dns_server]
        answer = resolver.resolve(domain, 'A')
        return answer[0].to_text()

    def switch_on_python_plot(self):
        import matplotlib
        matplotlib.use('TkAgg')

    def turn_on_python_plot(self):
        import matplotlib
        matplotlib.use('TkAgg')

    def is_iterable(self, obj):
        try:
            iter(obj)
            return True
        except TypeError:
            return False

    def memory_estimate(self):
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        print("RSS (MB):", mem.rss / 1024 ** 2)
        print("VMS (MB):", mem.vms / 1024 ** 2)
        # pause
        input('\33[7m' + "PRESS ENTER TO CONTINUE." + '\33[0m')


    def merge_dicts_list(self,dict_list):
        merged_dict = {}
        for d in dict_list:
            merged_dict.update(d)
        return merged_dict


    def save_distributed_array(self, large_array, outdir, n=10000,prefix='array',istqdm=True):
        '''
        :param large_array:
        :param outdir:
        :param n: save to each file every n sample
        :return:
        '''
        flag = 0
        temp_array = []
        if istqdm:
            iterator = tqdm(large_array, desc='saving...')
        else:
            iterator = large_array
        array_len = len(large_array)
        total_len = array_len // n + 1
        print('total files number:',total_len)
        for array_i in iterator:
            flag += 1
            arr = np.array(array_i)
            temp_array.append(arr)
            if flag % n == 0:
                digit_str = self.get_digit_str(total_len, int(flag / n))
                outf = join(outdir, f'{prefix}_{digit_str}.npy')
                np.save(outf, temp_array)
                temp_array = {}
        digit_str = self.get_digit_str(total_len, int(flag / n)+1)
        outf = join(outdir, f'{prefix}_{digit_str}.npy')
        np.save(outf, temp_array)

        pass



    def spatial_dict_to_df_multiple_value_flatten(self, spatial_dict, flatten_cols):

        '''
        flatten_cols will be flatten, other cols will be copied

        :param spatial_dict:
        spatial_dict = {
        row1:{flatten_col1:[val1,val2,val3],flatten_col2:[val1,val2,val3], other_col:val},
        row2:{flatten_col1:[val1,val2,val3],flatten_col2:[val1,val2,val3], other_col:val},
        row3:{flatten_col1:[val1,val2,val3],flatten_col2:[val1,val2,val3], other_col:val},
        }
        :param flatten_cols: [flatten_col1,flatten_col2]
        :return: Dataframe with columns: flatten_col1,flatten_col2, other_col
        '''

        col_list = []
        for pix in spatial_dict:
            keys = list(spatial_dict[pix].keys())
            col_list = keys
            break

        other_cols = []
        for col in col_list:
            if col not in flatten_cols:
                other_cols.append(col)

        vals_len_dict = {}
        for col in flatten_cols:
            for pix in spatial_dict:
                vals = spatial_dict[pix][col]
                vals_len = len(vals)
                vals_len_dict[pix] = vals_len
            break

        df = pd.DataFrame()

        pix_list = []
        for pix in spatial_dict:
            vals_len = vals_len_dict[pix]
            pix_list.extend([pix] * vals_len)
        df['pix'] = pix_list

        for col in flatten_cols:
            vals_all = []
            for pix in spatial_dict:
                vals = spatial_dict[pix][col]
                vals_all.extend(vals)
            df[col] = vals_all

        for col in other_cols:
            vals_all = []
            for pix in spatial_dict:
                val = spatial_dict[pix][col]
                vals_len = vals_len_dict[pix]
                vals_all.extend([val] * vals_len)
            df[col] = vals_all


        return df

    def df_bin_iterator(self, df, col, bins):
        df_copy = df.copy()
        df_copy[f'{col}_bins'] = pd.cut(df[col], bins=bins)
        df_group = df_copy.groupby([f'{col}_bins'])
        for name,df_group_i in df_group:
            name_str = str(name[0])
            yield df_group_i, name_str

    def get_digit_str(self,total_len,idx):
        digit = math.log(total_len, 10) + 1
        digit = int(digit)
        digit_str = f'{idx:0{digit}d}'
        return digit_str


class SMOOTH:
    '''
    一些平滑算法
    '''

    def __init__(self):

        pass

    def interp_1d(self, val):
        if len(val) == 0:
            return [None]

        # 1、插缺失值
        x = []
        val_new = []
        flag = 0
        for i in range(len(val)):
            if val[i] >= -10:
                flag += 1.
                index = i
                x = np.append(x, index)
                val_new = np.append(val_new, val[i])
        if flag / len(val) < 0.9:
            return [None]
        interp = interpolate.interp1d(x, val_new, kind='nearest', fill_value="extrapolate")

        xi = list(range(len(val)))
        yi = interp(xi)

        # 2、利用三倍sigma，去除离群值
        # print(len(yi))
        val_mean = np.mean(yi)
        sigma = np.std(yi)
        n = 3
        yi[(val_mean - n * sigma) > yi] = -999999
        yi[(val_mean + n * sigma) < yi] = 999999
        bottom = val_mean - n * sigma
        top = val_mean + n * sigma
        # plt.scatter(range(len(yi)),yi)
        # print(len(yi),123)
        # plt.scatter(range(len(yi)),yi)
        # plt.plot(yi)
        # plt.show()
        # print(len(yi))

        # 3、插离群值
        xii = []
        val_new_ii = []

        for i in range(len(yi)):
            if -999999 < yi[i] < 999999:
                index = i
                xii = np.append(xii, index)
                val_new_ii = np.append(val_new_ii, yi[i])

        interp_1 = interpolate.interp1d(xii, val_new_ii, kind='nearest', fill_value="extrapolate")

        xiii = list(range(len(val)))
        yiii = interp_1(xiii)

        # for i in range(len(yi)):
        #     if yi[i] == -999999:
        #         val_new_ii = np.append(val_new_ii, bottom)
        #     elif yi[i] == 999999:
        #         val_new_ii = np.append(val_new_ii, top)
        #     else:
        #         val_new_ii = np.append(val_new_ii, yi[i])

        return yiii

    def smooth_convolve(self, x, window_len=11, window='hanning'):
        """
        1d卷积滤波
        smooth the data using a window with requested size.
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the beginning and end part of the output signal.
        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
        output:
            the smoothed signal
        example:
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        see also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """
        x = np.array(x)

        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        # return y
        y_interpolated = y[(window_len // 2 - 1):-(window_len // 2)]
        if len(x) == len(y_interpolated):
            return y_interpolated
        elif len(x) < len(y_interpolated):
            return y_interpolated[1:]
        else:
            raise UserWarning('Need debug...')

    def smooth(self, x):
        # 后窗滤波
        # 滑动平均
        x = np.array(x)
        temp = 0
        new_x = []
        for i in range(len(x)):
            if i + 3 == len(x):
                break
            temp += x[i] + x[i + 1] + x[i + 2] + x[i + 3]
            new_x.append(temp / 4.)
            temp = 0
        return np.array(new_x)

    def smooth_interpolate(self, inx, iny, zoom):
        '''
        1d平滑差值
        :return:
        '''

        x_new = np.arange(min(inx), max(inx), ((max(inx) - min(inx)) / float(len(inx))) / float(zoom))
        func = interpolate.interp1d(inx, iny, kind='cubic')
        y_new = func(x_new)
        return x_new, y_new

    def mid_window_smooth(self, x, window=3):
        # 中滑动窗口滤波
        # 窗口为奇数

        if window < 0:
            raise IOError('window must be greater than 0')
        elif window == 0:
            return x
        else:
            pass

        if window % 2 != 1:
            raise IOError('window should be an odd number')

        x = np.array(x)

        new_x = []

        window_i = (window - 1) / 2
        # left = window - window_i
        for i in range(len(x)):
            left = i - window_i
            right = i + window_i
            if left < 0:
                left = 0
            if right >= len(x):
                right = len(x)
            picked_indx = list(range(int(left), int(right)))
            picked_value = Tools().pick_vals_from_1darray(x, picked_indx)
            picked_value_mean = np.nanmean(picked_value)
            new_x.append(picked_value_mean)

        #     if i - window < 0:
        #         new_x.append(x[i])
        #     else:
        #         temp = 0
        #         for w in range(window):
        #             temp += x[i - w]
        #         smoothed = temp / float(window)
        #         new_x = np.append(new_x, smoothed)
        return new_x

    def forward_window_smooth(self, x, window=3):
        # 前窗滤波
        # window = window-1
        # 不改变数据长度

        if window < 0:
            raise IOError('window must be greater than 0')
        elif window == 0:
            return x
        else:
            pass

        x = np.array(x)

        new_x = np.array([])
        # plt.plot(x)
        # plt.show()
        for i in range(len(x)):
            if i - window < 0:
                new_x = np.append(new_x, x[i])
            else:
                temp = 0
                for w in range(window):
                    temp += x[i - w]
                smoothed = temp / float(window)
                new_x = np.append(new_x, smoothed)
        return new_x

    def filter_3_sigma(self, arr_list):
        sum_ = []
        for i in arr_list:
            if i >= 0:
                sum_.append(i)
        sum_ = np.array(sum_)
        val_mean = np.mean(sum_)
        sigma = np.std(sum_)
        n = 3
        sum_[(val_mean - n * sigma) > sum_] = -999999
        sum_[(val_mean + n * sigma) < sum_] = -999999

        # for i in
        return sum_


    def hist_plot_smooth(self, arr, interpolate_window=6, **kwargs):
        weights = np.ones_like(arr) / float(len(arr))
        n1, x1, patch = plt.hist(arr, weights=weights, **kwargs)
        density1 = stats.gaussian_kde(arr)
        y1 = density1(x1)
        coe = max(n1) / max(y1)
        y1 = y1 * coe
        y1 = self.smooth_convolve(y1, window_len=interpolate_window)
        return x1, y1

        pass


class DIC_and_TIF:
    '''
    字典转tif
    tif转字典
    '''

    def __init__(self,
                 originX=-180.,
                 endX=180,
                 originY=90.,
                 endY=-90,
                 pixelsize=0.5,
                 tif_template=None):
        if tif_template:
            self.arr_template, self.originX, self.originY, self.pixelWidth, self.pixelHeight = \
                ToRaster().raster2array(tif_template)
        else:
            self.originX, self.originY, self.pixelWidth, self.pixelHeight = \
                originX, originY, pixelsize, -pixelsize
            r = int((endY - originY) / self.pixelHeight)
            c = int((endX - originX) / self.pixelWidth)
            self.arr_template = np.ones((r, c))
        pass

    def arr_to_tif(self, array, newRasterfn):
        grid_nan = np.isnan(array)
        grid = np.logical_not(grid_nan)
        array[np.logical_not(grid)] = -999999
        ToRaster().array2raster(newRasterfn, self.originX, self.originY, self.pixelWidth, self.pixelHeight, array)
        pass

    def arr_to_tif_GDT_Byte(self, array, newRasterfn):
        # template
        grid_nan = np.isnan(array)
        grid = np.logical_not(grid_nan)
        array[np.logical_not(grid)] = 255
        ToRaster().array2raster_GDT_Byte(newRasterfn, self.originX, self.originY, self.pixelWidth, self.pixelHeight,
                                         array)
        pass

    def spatial_arr_to_dic(self, arr):

        pix_dic = {}
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                pix = (i, j)
                val = arr[i][j]
                pix_dic[pix] = val

        return pix_dic

    def pix_dic_to_spatial_arr(self, spatial_dic):

        row = len(self.arr_template)
        col = len(self.arr_template[0])
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in spatial_dic:
                    val_pix = spatial_dic[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        spatial = np.array(spatial, dtype=float)
        return spatial

    def pix_dic_to_spatial_arr_mean(self, spatial_dic):

        mean_spatial_dic = {}
        for pix in tqdm(spatial_dic, desc='calculating spatial mean'):
            vals = spatial_dic[pix]
            if len(vals) == 0:
                mean = np.nan
            else:
                mean = np.nanmean(vals)
            mean_spatial_dic[pix] = mean

        spatial = self.pix_dic_to_spatial_arr(mean_spatial_dic)
        spatial = np.array(spatial, dtype=float)
        return spatial

    def pix_dic_to_spatial_arr_trend(self, spatial_dic):

        mean_spatial_dic = {}
        for pix in tqdm(spatial_dic, desc='calculating spatial trend'):
            vals = spatial_dic[pix]
            if len(vals) == 0:
                a = np.nan
            else:
                x = list(range(len(vals)))
                y = vals
                try:
                    a, _, _, _ = Tools().nan_line_fit(x, y)
                except:
                    a = np.nan
            mean_spatial_dic[pix] = a

        spatial = self.pix_dic_to_spatial_arr(mean_spatial_dic)
        spatial = np.array(spatial, dtype=float)
        return spatial

    def pix_dic_to_spatial_arr_ascii(self, spatial_dic):
        row = len(self.arr_template)
        col = len(self.arr_template[0])
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in spatial_dic:
                    val_pix = spatial_dic[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        spatial = np.array(spatial)
        return spatial

    def pix_dic_to_tif(self, spatial_dic, out_tif):

        spatial = self.pix_dic_to_spatial_arr(spatial_dic)
        # spatial = np.array(spatial)
        self.arr_to_tif(spatial, out_tif)

    def pix_dic_to_shp(self, spatial_dic, outshp, temp_dir):
        pix_to_lon_lat_dic = self.spatial_tif_to_lon_lat_dic(temp_dir)
        inlist = []
        for pix in spatial_dic:
            lon, lat = pix_to_lon_lat_dic[pix]
            val = spatial_dic[pix]
            if np.isnan(val):
                continue
            inlist.append((lon, lat, val))
        Tools().point_to_shp(inlist, outshp)

        pass

    def pix_dic_to_tif_every_time_stamp(self, spatial_dic, outdir, filename_list: list = None):
        Tools().mkdir(outdir)
        ## check values number
        vals_number_list = []
        for pix in spatial_dic:
            vals = spatial_dic[pix]
            vals_number = len(vals)
            if vals_number not in vals_number_list:
                vals_number_list.append(vals_number)
        vals_number_list = list(set(vals_number_list))
        if len(vals_number_list) != 1:
            print('vals number not equal')
            raise
        vals_number = vals_number_list[0]
        if not vals_number == len(filename_list):
            raise ValueError('vals number not equal to filename number')
        ## get number digits
        vals_number_str = str(vals_number)
        vals_number_digits = len(vals_number_str)
        n = vals_number_digits
        for i in tqdm(range(vals_number)):
            spatial_dic_i = {}
            for pix in spatial_dic:
                val = spatial_dic[pix][i]
                spatial_dic_i[pix] = val
            if filename_list is not None:
                fname = str(filename_list[i]) + '.tif'
            else:
                fname = f'{i:0{n}d}.tif'
            fpath = join(outdir, fname)
            self.pix_dic_to_tif(spatial_dic_i, fpath)

    def pix_dic_to_tif_every_time_stamp_dict(self, spatial_dic, outdir):
        '''
        :param spatial_dic: {pix1:{key1:value,key2:value},
                             pix2:{key1:value,key2:value}}
        :param outdir:
        :return:
        '''
        Tools().mkdir(outdir)
        ## check values number
        keys_number_list = []
        for pix in spatial_dic:
            dict_i = spatial_dic[pix]
            keys = list(dict_i.keys())
            for key in keys:
                if key not in keys_number_list:
                    keys_number_list.append(key)
        keys_number_list.sort()
        for key in tqdm(keys_number_list):
            spatial_dic_i = {}
            for pix in spatial_dic:
                if not key in spatial_dic[pix]:
                    continue
                val = spatial_dic[pix][key]
                spatial_dic_i[pix] = val
            fname = f'{key}.tif'
            fpath = join(outdir, fname)
            self.pix_dic_to_tif(spatial_dic_i, fpath)

    def spatial_tif_to_lon_lat_dic(self, temp_dir=None):
        if temp_dir is None:
            arr = self.arr_template
            pix_to_lon_lat_dic = {}
            for i in tqdm(list(range(len(arr))), desc='tif_to_lon_lat_dic'):
                for j in range(len(arr[0])):
                    pix = (i, j)
                    lon = self.originX + self.pixelWidth * j
                    lat = self.originY + self.pixelHeight * i
                    pix_to_lon_lat_dic[pix] = tuple([lon, lat])
            return pix_to_lon_lat_dic
        else:
            this_class_dir = os.path.join(temp_dir, 'DIC_and_TIF')
            Tools().mkdir(this_class_dir, force=True)
            outf_conf = f'{self.originX}_{self.originY}_{self.pixelWidth}_{self.pixelHeight}'
            outf_conf = outf_conf.encode('utf-8')
            hash_outf = hashlib.sha256(outf_conf).hexdigest()
            outf = os.path.join(this_class_dir, f'{hash_outf}.npy')
            if os.path.isfile(outf):
                print(f'loading {outf}')
                dic = Tools().load_npy(outf)
                return dic
            else:
                arr = self.arr_template
                pix_to_lon_lat_dic = {}
                for i in tqdm(list(range(len(arr))), desc='tif_to_lon_lat_dic'):
                    for j in range(len(arr[0])):
                        pix = (i, j)
                        lon = self.originX + self.pixelWidth * j
                        lat = self.originY + self.pixelHeight * i
                        pix_to_lon_lat_dic[pix] = tuple([lon, lat])
                print('saving')
                np.save(outf, pix_to_lon_lat_dic)
                return pix_to_lon_lat_dic

    def spatial_tif_to_dic(self, tif):

        arr = ToRaster().raster2array(tif)[0]
        arr = np.array(arr, dtype=float)
        arr = Tools().mask_999999_arr(arr, warning=False)
        dic = self.spatial_arr_to_dic(arr)
        return dic

    def spatial_tif_to_arr(self, tif):
        arr = ToRaster().raster2array(tif)[0]
        arr = np.array(arr, dtype=float)
        arr = Tools().mask_999999_arr(arr, warning=False)
        return arr

    def void_spatial_dic(self):
        arr = self.arr_template
        void_dic = {}
        for row in range(len(arr)):
            for col in range(len(arr[row])):
                key = (row, col)
                void_dic[key] = []
        return void_dic

    def void_spatial_dic_nan(self):
        arr = self.arr_template
        void_dic = {}
        for row in range(len(arr)):
            for col in range(len(arr[row])):
                key = (row, col)
                void_dic[key] = np.nan
        return void_dic

    def void_spatial_dic_zero(self):
        arr = self.arr_template
        void_dic = {}
        for row in range(len(arr)):
            for col in range(len(arr[row])):
                key = (row, col)
                void_dic[key] = 0.
        return void_dic

    def void_spatial_dic_ones(self):
        arr = self.arr_template
        void_dic = {}
        for row in range(len(arr)):
            for col in range(len(arr[row])):
                key = (row, col)
                void_dic[key] = 1.
        return void_dic

    def plot_back_ground_arr(self, rasterized_world_tif, ax=None, **kwargs):
        arr = ToRaster().raster2array(rasterized_world_tif)[0]
        ndv = ToRaster().get_ndv(rasterized_world_tif)
        back_ground = []
        for i in range(len(arr)):
            temp = []
            for j in range(len(arr[0])):
                val = arr[i][j]
                if val == ndv:
                    temp.append(np.nan)
                    continue
                if val != 1:
                    temp.append(np.nan)
                else:
                    temp.append(1.)
            back_ground.append(temp)
        back_ground = np.array(back_ground)
        if ax == None:
            plt.imshow(back_ground, 'gray', vmin=0, vmax=1.4, zorder=-1, **kwargs)
        else:
            ax.imshow(back_ground, 'gray', vmin=0, vmax=1.4, zorder=-1, **kwargs)

    def plot_back_ground_arr_north_sphere(self, rasterized_world_tif, ax=None, **kwargs):
        ndv = ToRaster().get_ndv(rasterized_world_tif)
        arr = ToRaster().raster2array(rasterized_world_tif)[0]
        back_ground = []
        for i in range(len(arr)):
            temp = []
            for j in range(len(arr[0])):
                val = arr[i][j]
                if val == ndv:
                    temp.append(np.nan)
                    continue
                if val != 1:
                    temp.append(np.nan)
                else:
                    temp.append(1.)
            back_ground.append(temp)
        back_ground = np.array(back_ground)
        if ax == None:
            plt.imshow(back_ground[:int(len(arr) / 2)], 'gray', vmin=0, vmax=1.4, zorder=-1, **kwargs)
        else:
            ax.imshow(back_ground[:int(len(arr) / 2)], 'gray', vmin=0, vmax=1.4, zorder=-1, **kwargs)

    def mask_ocean_dic(self):
        arr = self.arr_template
        ocean_dic = {}
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                val = arr[i][j]
                if val < -99999:
                    continue
                else:
                    ocean_dic[(i, j)] = 1
        return ocean_dic

    def show_pix(self, pix, window_pix=20):
        dic_temp = {}
        c, r = pix
        for ci in range(c - window_pix, c + window_pix):
            for ri in range(r - window_pix, r + window_pix):
                pix_new = (ci, ri)
                dic_temp[pix_new] = 10
        arr = self.pix_dic_to_spatial_arr(dic_temp)
        # plt.figure()
        self.plot_back_ground_arr()
        plt.imshow(arr, cmap='gray', vmin=0, vmax=100, zorder=99)
        plt.title(str(pix))

    def china_pix(self, pix):
        # only for 0.5 spatial resolution
        r, c = pix
        china_r = list(range(75, 150))
        china_c = list(range(550, 620))
        if r in china_r:
            if c in china_c:
                return True
            else:
                return False
        else:
            return False

    def per_pix_animate(self, per_pix_dir, interval_t=10, condition=''):

        import matplotlib.animation as animation

        def plot_back_ground_arr():
            arr = self.arr_template
            back_ground = []
            for i in range(len(arr)):
                temp = []
                for j in range(len(arr[0])):
                    val = arr[i][j]
                    if val < -90000:
                        temp.append(100)
                    else:
                        temp.append(70)
                back_ground.append(temp)
            back_ground = np.array(back_ground)
            return back_ground

        back_ground = plot_back_ground_arr()

        def init():
            line.set_ydata([np.nan] * len(x))
            return line,

        def show_pix(pix, background_arr):
            c, r = pix
            selected_pix = []
            for ci in range(c - 5, c + 5):
                for ri in range(r - 5, r + 5):
                    pix_new = (ci, ri)
                    selected_pix.append(pix_new)
            for pix in selected_pix:
                background_arr[pix] = -999
            return background_arr

        fdir = per_pix_dir
        dic = Tools().load_npy_dir(fdir, condition=condition)

        # selected_pix_sort = []
        # for pix in tqdm(dic):
        #     if not self.china_pix(pix):
        #         continue
        #     selected_pix_sort.append(pix)
        # selected_pix_sort.sort()

        flag = 0
        china_pix = []
        china_pix_val = {}
        min_max_v = []
        for pix in dic:
            val = dic[pix]
            val = np.array(val)
            if len(val) == 0:
                continue
            val[val < -9999] = np.nan
            china_pix_val[flag] = val
            vmin_init = np.nanmin(val)
            vmax_init = np.nanmax(val)
            min_max_v.append((vmin_init, vmax_init))
            china_pix.append(pix)
            flag += 1
        min_max_set_dic = {}

        vmin_list = []
        vmax_list = []
        for i in range(len(min_max_v)):
            vmin_list.append(min_max_v[i][0])
            vmax_list.append(min_max_v[i][1])
            vmin_set = np.min(vmin_list)
            vmax_set = np.max(vmax_list)
            min_max_set_dic[i] = (vmin_set, vmax_set)
        # exit()
        fig = plt.figure()
        ax2 = fig.add_subplot(212)
        ax1 = fig.add_subplot(211)
        # print(dic[china_pix[0]])
        x = list(range(len(china_pix_val[0])))
        val_init = china_pix_val[0]
        val_init[val_init < -999] = np.nan
        line, = ax1.plot(list(range(len(x))), val_init)
        # if vmin_init == None:
        #     vmin_init = np.nanmin(val_init)
        # if vmax_init == None:
        #     vmax_init = np.nanmax(val_init)

        im = ax2.imshow(back_ground, cmap='gray', vmin=0, vmax=100, zorder=99)

        def animate(i):
            back_ground_copy = copy.copy(back_ground)
            val_in = china_pix_val[i]
            val_in[val_in < -999] = 0
            line.set_ydata(val_in)
            ax1.set_title(china_pix[i])
            vmin_, vmax_ = min_max_set_dic[i]
            # if vmin == None:
            #     vmin = vmin_
            # if vmax == None:
            #     vmax = vmax_
            if not np.isnan(vmin_) and not np.isnan(vmax_):
                ax1.set_ylim(vmin_, vmax_)
            im_arr_in = show_pix(china_pix[i], back_ground_copy)
            im.set_array(im_arr_in)
            return line,

        ani = animation.FuncAnimation(
            fig, animate, init_func=init, interval=interval_t, blit=False, frames=len(china_pix))

        plt.show()

        pass

    def lon_lat_val_to_tif(self, lon_list, lat_list, val_list, outtif):
        lonlist_set = list(set(lon_list))
        latlist_set = list(set(lat_list))
        lonlist_set.sort()
        latlist_set.sort()
        latlist_set = latlist_set[::-1]
        originX = min(lonlist_set)
        originY = max(latlist_set)
        pixelWidth = (lonlist_set[-1] - lonlist_set[0]) / (len(lonlist_set) - 1)
        pixelHeight = (latlist_set[-1] - latlist_set[0]) / (len(latlist_set) - 1)
        spatial_dic = {}
        for i in range(len(lon_list)):
            lon = lon_list[i]
            lat = lat_list[i]
            val = val_list[i]
            r = abs(round((lat - originY) / pixelHeight))
            c = abs(round((lon - originX) / pixelWidth))
            r = int(r)
            c = int(c)
            spatial_dic[(r, c)] = val
        spatial = []
        row = len(latlist_set)
        col = len(lonlist_set)
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in spatial_dic:
                    val_pix = spatial_dic[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)
        spatial = np.array(spatial, dtype=float)
        longitude_start = originX
        latitude_start = originY
        ToRaster().array2raster(outtif, longitude_start, latitude_start, pixelWidth, pixelHeight, spatial)

    def lon_lat_ascii_to_arr(self, lon_list, lat_list, val_list):
        lonlist_set = list(set(lon_list))
        latlist_set = list(set(lat_list))
        lonlist_set.sort()
        latlist_set.sort()
        latlist_set = latlist_set[::-1]
        originX = min(lonlist_set)
        originY = max(latlist_set)
        pixelWidth = lonlist_set[1] - lonlist_set[0]
        pixelHeight = latlist_set[1] - latlist_set[0]
        spatial_dic = {}
        for i in range(len(lon_list)):
            lon = lon_list[i]
            lat = lat_list[i]
            val = val_list[i]
            r = abs(int((lat - originY) / pixelHeight))
            c = abs(int((lon - originX) / pixelWidth))
            spatial_dic[(r, c)] = val
        spatial = []
        row = abs(int((max(latlist_set) - min(latlist_set)) / pixelHeight))
        col = abs(int((max(lonlist_set) - min(lonlist_set)) / pixelWidth))
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in spatial_dic:
                    val_pix = spatial_dic[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)
        spatial = np.array(spatial)
        array, longitude_start, latitude_start, pixelWidth, pixelHeight = \
            spatial, originX, originY, pixelWidth, pixelHeight
        return array, longitude_start, latitude_start, pixelWidth, pixelHeight

    def unify_raster(self, in_tif, out_tif, ndv=-999999):
        '''
        Unify raster to the extend of global (-180 180 90 -90)
        '''
        insert_value = ndv
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(in_tif)
        # insert values to row
        top_line_num = abs((90. - originY) / pixelHeight)
        bottom_line_num = abs((90. + originY + pixelHeight * len(array)) / pixelHeight)
        top_line_num = int(round(top_line_num, 0))
        bottom_line_num = int(round(bottom_line_num, 0))
        nan_array_insert = np.ones_like(array[0]) * insert_value
        top_array_insert = []
        for i in range(top_line_num):
            top_array_insert.append(nan_array_insert)
        bottom_array_insert = []
        for i in range(bottom_line_num):
            bottom_array_insert.append(nan_array_insert)
        bottom_array_insert = np.array(bottom_array_insert)
        if len(top_array_insert) != 0:
            arr_temp = np.insert(array, obj=0, values=top_array_insert, axis=0)
        else:
            arr_temp = array
        if len(bottom_array_insert) != 0:
            array_unify_top_bottom = np.vstack((arr_temp, bottom_array_insert))
        else:
            array_unify_top_bottom = arr_temp

        # insert values to column
        left_line_num = abs((-180. - originX) / pixelWidth)
        right_line_num = abs((180. - (originX + pixelWidth * len(array[0]))) / pixelWidth)
        left_line_num = int(round(left_line_num, 0))
        right_line_num = int(round(right_line_num, 0))
        left_array_insert = []
        right_array_insert = []
        for i in range(left_line_num):
            left_array_insert.append(insert_value)
        for i in range(right_line_num):
            right_array_insert.append(insert_value)

        array_unify_left_right = []
        for i in array_unify_top_bottom:
            if len(left_array_insert) != 0:
                arr_temp = np.insert(i, obj=0, values=left_array_insert, axis=0)
            else:
                arr_temp = i
            if len(right_array_insert) != 0:
                array_temp1 = np.hstack((arr_temp, right_array_insert))
            else:
                array_temp1 = arr_temp
            array_unify_left_right.append(array_temp1)
        array_unify_left_right = np.array(array_unify_left_right)
        newRasterfn = out_tif
        ToRaster().array2raster(newRasterfn, -180, 90, pixelWidth, pixelHeight, array_unify_left_right, ndv=ndv)

    def unify_raster1(self, in_tif, out_tif, res, srcSRS, dstSRS):  # todo: need to be tested
        row = len(self.arr_template)
        col = len(self.arr_template[0])
        sY = self.originY
        sX = self.originX
        eY = sY + row * self.pixelHeight
        eX = sX + col * self.pixelWidth
        extent = [sX, eY, eX, sY]
        dataset = gdal.Open(in_tif)
        gdal.Warp(out_tif, dataset, srcSRS=srcSRS, dstSRS=dstSRS, outputBounds=extent)

    def resample_reproj(self, in_tif, out_tif, res, srcSRS, dstSRS):
        dataset = gdal.Open(in_tif)
        gdal.Warp(out_tif, dataset, xRes=res, yRes=res, srcSRS=srcSRS, dstSRS=dstSRS)

    def gen_srs_from_wkt(self, proj_wkt):
        '''
        proj_wkt example:
        prj_info = PROJCS["Homolosine",
                GEOGCS["WGS 84",
                    DATUM["WGS_1984",
                        SPHEROID["WGS 84",6378137,298.257223563,
                            AUTHORITY["EPSG","7030"]],
               AUTHORITY["EPSG","6326"]],
                    PRIMEM["Greenwich",0,
                        AUTHORITY["EPSG","8901"]],
                    UNIT["degree",0.0174532925199433,
                        AUTHORITY["EPSG","9122"]],
                    AUTHORITY["EPSG","4326"]],
                PROJECTION["Interrupted_Goode_Homolosine"],
                UNIT["Meter",1]]
        '''
        inRasterSRS = osr.SpatialReference()
        inRasterSRS.ImportFromWkt(proj_wkt)
        return inRasterSRS

    def lon_lat_to_pix(self, lon_list, lat_list, isInt=True):
        pix_list = []
        for i in range(len(lon_list)):
            lon = lon_list[i]
            lat = lat_list[i]
            if lon > 180 or lon < -180:
                success_lon = 0
                c = np.nan
            else:
                c = (lon - self.originX) / self.pixelWidth
                if isInt == True:
                    c = int(c)
                success_lon = 1

            if lat > 90 or lat < -90:
                success_lat = 0
                r = np.nan
            else:
                r = (lat - self.originY) / self.pixelHeight
                if isInt == True:
                    r = int(r)
                success_lat = 1
            if success_lat == 1 and success_lon == 1:
                pix_list.append((r, c))
            else:
                pix_list.append(np.nan)

        pix_list = tuple(pix_list)
        return pix_list

    def gen_ones_background_tif(self, outtif):
        # outtif = T.path_join(this_root,'conf','ones_background.tif')
        arr = self.void_spatial_dic_ones()
        self.pix_dic_to_tif(arr, outtif)
        pass

    def gen_land_background_tif(self, shp, outtif, pix_size=0.5):
        # outtif = T.path_join(this_root,'conf','land.tif')
        # shp = T.path_join(this_root,'shp','world.shp')
        ToRaster().shp_to_raster(shp, outtif, pix_size)
        self.unify_raster(outtif, outtif)
        pass

    def pix_to_lon_lat(self, pix):
        r, c = pix
        lat = self.originY + (self.pixelHeight * r)
        lon = self.originX + (self.pixelWidth * c)
        return lon, lat

    def plot_sites_location(self, lon_list, lat_list, background_tif=None, inshp=None, out_background_tif=None,
                            pixel_size=None, text_list=None, colorlist=None, isshow=False, **kwargs):
        pix_list = self.lon_lat_to_pix(lon_list, lat_list, isInt=False)
        lon_list = []
        lat_list = []
        for lon, lat in pix_list:
            lon_list.append(lon)
            lat_list.append(lat)
        if background_tif:
            self.plot_back_ground_arr(background_tif)
        if inshp:
            if out_background_tif == None:
                raise 'please set out_background_tif path'
            if pixel_size == None:
                raise 'please set pixel_size (e.g. 0.5, unit: deg)'
            if os.path.isfile(out_background_tif):
                print(out_background_tif, 'available')
                self.plot_back_ground_arr(out_background_tif)
            else:
                print(out_background_tif, 'generating...')
                background_tif = ToRaster().shp_to_raster(inshp, out_background_tif, pixel_size=pixel_size, ndv=255)
                ToRaster().unify_raster(background_tif, background_tif, GDT_Byte=True)
                print('done')
                self.plot_back_ground_arr(background_tif)

        if colorlist:
            plt.scatter(lat_list, lon_list, c=colorlist, **kwargs)
            plt.colorbar()
        else:
            plt.scatter(lat_list, lon_list, **kwargs)
        if not text_list == None:
            for i in range(len(lon_list)):
                plt.text(lat_list[i], lon_list[i], text_list[i])
        if isshow:
            plt.show()

    def plot_df_spatial_pix(self, df, global_land_tif):
        pix_list = df['pix'].tolist()
        pix_list = list(set(pix_list))
        spatial_dict = {pix: 1 for pix in pix_list}
        arr = self.pix_dic_to_spatial_arr(spatial_dict)
        self.plot_back_ground_arr(global_land_tif)
        plt.imshow(arr)

    def rad(self, d):
        return d * math.pi / 180

    def GetDistance(self, lng1, lat1, lng2, lat2):
        radLat1 = self.rad(lat1)
        radLat2 = self.rad(lat2)
        a = radLat1 - radLat2
        b = self.rad(lng1) - self.rad(lng2)
        s = 2 * math.asin(math.sqrt(
            math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
        s = s * 6378.137 * 1000
        distance = round(s, 4)
        return distance

        #### from https://kite.com/python/answers/how-to-find-the-distance-between-two-lat-long-coordinates-in-python
        # R = 6373.0
        # dlon = lng2 - lng1
        # dlat = lat2 - lat1
        # a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        # c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        # distance = R * c
        # # print distance
        # # exit()
        # return distance
        pass

    def calculate_pixel_area(self):
        pix_list = self.void_spatial_dic()
        pixel_size = self.pixelWidth
        area_dict = {}
        for pix in tqdm(pix_list, desc='calculate_pixel_area'):
            lon, lat = self.pix_to_lon_lat(pix)
            upper_left_lon = lon - pixel_size / 2
            upper_left_lat = lat + pixel_size / 2
            upper_right_lon = lon + pixel_size / 2
            upper_right_lat = lat + pixel_size / 2
            lower_left_lon = lon - pixel_size / 2
            lower_left_lat = lat - pixel_size / 2
            lower_right_lon = lon + pixel_size / 2
            lower_right_lat = lat - pixel_size / 2
            upper_left_to_upper_right = self.GetDistance(upper_left_lon, upper_left_lat, upper_right_lon,
                                                         upper_right_lat)
            upper_left_to_lower_left = self.GetDistance(upper_left_lon, upper_left_lat, lower_left_lon, lower_left_lat)
            area = upper_left_to_upper_right * upper_left_to_lower_left
            area_dict[pix] = area
        return area_dict

    def wkt_84(self):
        wkt = '''GEOGCRS["WGS 84",
    ENSEMBLE["World Geodetic System 1984 ensemble",
        MEMBER["World Geodetic System 1984 (Transit)"],
        MEMBER["World Geodetic System 1984 (G730)"],
        MEMBER["World Geodetic System 1984 (G873)"],
        MEMBER["World Geodetic System 1984 (G1150)"],
        MEMBER["World Geodetic System 1984 (G1674)"],
        MEMBER["World Geodetic System 1984 (G1762)"],
        MEMBER["World Geodetic System 1984 (G2139)"],
        ELLIPSOID["WGS 84",6378137,298.257223563,
            LENGTHUNIT["metre",1]],
        ENSEMBLEACCURACY[2.0]],
    PRIMEM["Greenwich",0,
        ANGLEUNIT["degree",0.0174532925199433]],
    CS[ellipsoidal,2],
        AXIS["geodetic latitude (Lat)",north,
            ORDER[1],
            ANGLEUNIT["degree",0.0174532925199433]],
        AXIS["geodetic longitude (Lon)",east,
            ORDER[2],
            ANGLEUNIT["degree",0.0174532925199433]],
    USAGE[
        SCOPE["Horizontal component of 3D system."],
        AREA["World."],
        BBOX[-90,-180,90,180]],
    ID["EPSG",4326]]'''
        return wkt


class MULTIPROCESS:

    def __init__(self, func, params, istqdm=False,istqdm_inner=False):
        self.func = func
        self.params = params
        self.istqdm = istqdm
        self.istqdm_inner = istqdm_inner
        copyreg.pickle(types.MethodType, self._pickle_method)
        pass

    def _pickle_method(self, m):
        if m.__self__ is None:
            return getattr, (m.__self__.__class__, m.__func__.__name__)
        else:
            return getattr, (m.__self__, m.__func__.__name__)

    def func_wrapper(self, *args, **kwargs):
        if not self.istqdm_inner:
            tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        return self.func(*args, **kwargs)

    def run(self, process=4, process_or_thread='p', **kwargs):
        assert process > 0
        if process_or_thread == 'p':
            # pool = multiprocessing.Pool(process)
            pool = ProcessPool(nodes=process)
        elif process_or_thread == 't':
            pool = TPool(process)
        else:
            raise IOError('process_or_thread key error, input keyword such as "p" or "t"')
        if self.istqdm:
            results = list(tqdm(pool.imap(self.func_wrapper, self.params), total=len(self.params), **kwargs))
        else:
            results = pool.map(self.func_wrapper, self.params)
        pool.close()
        pool.join()
        if process_or_thread == 'p':
            pool.clear()
        return results


class KDE_plot:

    def __init__(self):

        pass

    def reverse_colourmap(self, cmap, name='my_cmap_r'):
        """
        In:
        cmap, name
        Out:
        my_cmap_r
        Explanation:
        t[0] goes from 0 to 1
        row i:   x  y0  y1 -> t[0] t[1] t[2]
                       /
                      /
        row i+1: x  y0  y1 -> t[n] t[1] t[2]
        so the inverse should do the same:
        row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                       /
                      /
        row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
        """
        reverse = []
        k = []

        for key in cmap._segmentdata:
            k.append(key)
            channel = cmap._segmentdata[key]
            data = []

            for t in channel:
                data.append((1 - t[0], t[2], t[1]))
            reverse.append(sorted(data))

        LinearL = dict(list(zip(k, reverse)))
        my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
        return my_cmap_r

    def makeColours(self, vals, cmap, reverse=0):
        norm = []
        for i in vals:
            norm.append((i - np.min(vals)) / (np.max(vals) - np.min(vals)))
        colors = []
        cmap = plt.get_cmap(cmap)
        if reverse:
            cmap = self.reverse_colourmap(cmap)
        else:
            cmap = cmap

        for i in norm:
            colors.append(cmap(i))
        return colors

    def linefit(self, x, y):
        '''
        最小二乘法拟合直线
        :param x:
        :param y:
        :return:
        '''
        N = float(len(x))
        sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
        for i in range(0, int(N)):
            sx += x[i]
            sy += y[i]
            sxx += x[i] * x[i]
            syy += y[i] * y[i]
            sxy += x[i] * y[i]
        a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
        b = (sy - a * sx) / N
        r = -(sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
        r, p = stats.pearsonr(x, y)
        return a, b, r, p

    def plot_fit_line(self, a, b, r, p, X, ax=None, title='', is_label=True, is_formula=True, line_color='k', **argvs):
        '''
        画拟合直线 y=ax+b
        画散点图 X,Y
        :param a:
        :param b:
        :param X:
        :param Y:
        :param i:
        :param title:
        :return:
        '''
        x = np.linspace(min(X), max(X), 10)
        y = a * x + b
        #
        # plt.subplot(2,2,i)
        # plt.scatter(X,Y,marker='o',s=5,c = 'grey')
        # plt.plot(X,Y)
        c = line_color
        if is_label == True:
            if is_formula == True:
                label = 'y={:0.2f}x+{:0.2f}\nr={:0.2f}\np={:0.2f}'.format(a, b, r, p)
            else:
                label = 'r={:0.2f}'.format(r)
        else:
            label = None

        if ax == None:
            if not 'linewidth' in argvs:
                plt.plot(x, y, linestyle='dashed', c=c, alpha=0.7, label=label, **argvs)
            else:
                plt.plot(x, y, linestyle='dashed', c=c, alpha=0.7, label=label, **argvs)
        else:
            if not 'linewidth' in argvs:
                ax.plot(x, y, linestyle='dashed', c=c, alpha=0.7, label=label, **argvs)
            else:
                ax.plot(x, y, linestyle='dashed', c=c, alpha=0.7, label=label, **argvs)

    def plot_scatter(self, val1, val2,
                     plot_fit_line=False,
                     max_n=30000,
                     is_plot_1_1_line=False,
                     cmap='ocean',
                     reverse=0, s=0.3,
                     title='', ax=None,
                     silent=False, is_KDE=True,
                     fit_line_c=None,
                     is_equal=False,
                     x_y_lim=None,
                     **kwargs):
        val1 = np.array(val1)
        val2 = np.array(val2)
        if not silent:
            print('data length is {}'.format(len(val1)))
        if len(val1) > max_n:
            val_range_index = list(range(len(val1)))
            val_range_index = random.sample(val_range_index, max_n)  # 从val中随机选择n个点，目的是加快核密度算法
            new_val1 = []
            new_val2 = []
            for i in val_range_index:
                new_val1.append(val1[i])
                new_val2.append(val2[i])
            val1 = new_val1
            val2 = new_val2
            if not silent:
                print('data length is modified to {}'.format(len(val1)))
        else:
            val1 = val1
            val2 = val2

        kde_val = np.array([val1, val2])
        if not silent:
            print('doing kernel density estimation... ')
        new_v1 = []
        new_v2 = []
        for vals_12 in kde_val.T:
            # print(vals_12)
            v1, v2 = vals_12
            if np.isnan(v1):
                continue
            if np.isnan(v2):
                continue
            new_v1.append(v1)
            new_v2.append(v2)
        val1, val2 = new_v1, new_v2
        kde_val = np.array([new_v1, new_v2])
        if is_KDE:
            densObj = kde(kde_val)
            dens_vals = densObj.evaluate(kde_val)
            colors = self.makeColours(dens_vals, cmap, reverse=reverse)
        else:
            colors = None
        if ax == None:
            plt.figure()
            plt.title(title)
            plt.scatter(val1, val2, c=colors, linewidths=0, s=s, **kwargs)
        else:
            ax.set_title(title)
            ax.scatter(val1, val2, c=colors, linewidths=0, s=s, **kwargs)

        if x_y_lim:
            xmin, xmax, ymin, ymax = x_y_lim
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        if is_equal:
            plt.axis('equal')
        if plot_fit_line:
            a, b, r, p = self.linefit(val1, val2)
            if is_plot_1_1_line:
                plt.plot([np.min([val1, val2]), np.max([val1, val2])], [np.min([val1, val2]), np.max([val1, val2])],
                         '--', c='k')
            self.plot_fit_line(a, b, r, p, val1, line_color=fit_line_c)
            # plt.legend()
            return a, b, r, p

    def plot_scatter_hex(self, x, y, kind="hex", color="#4CB391", xlim=None, ylim=None, gridsize=80):
        df_temp = pd.DataFrame()
        df_temp['x'] = x
        df_temp['y'] = y
        if not xlim == None:
            df_temp = df_temp[df_temp['x'] > xlim[0]]
            df_temp = df_temp[df_temp['x'] < xlim[1]]
        if not ylim == None:
            df_temp = df_temp[df_temp['y'] > ylim[0]]
            df_temp = df_temp[df_temp['y'] < ylim[1]]
        X = df_temp['x'].values
        Y = df_temp['y'].values
        sns.jointplot(x=X, y=Y, kind="hex", color="#4CB391", xlim=xlim, ylim=ylim, gridsize=gridsize)

    def cmap_with_transparency(self, cmap, min_alpha=0., max_alpha=0.5):
        ncolors = 256
        color_array = plt.get_cmap(cmap)(range(ncolors))

        # change alpha values
        color_array[:, -1] = np.linspace(min_alpha, max_alpha, ncolors)

        # create a colormap object
        map_object = LinearSegmentedColormap.from_list(name=f'{cmap}_transparency', colors=color_array)

        # register this new colormap with matplotlib
        plt.register_cmap(cmap=map_object)
        return f'{cmap}_transparency'
        # show some example data
        # f, ax = plt.subplots()
        # h = ax.imshow(np.random.rand(100, 100), cmap='rainbow_alpha')
        # plt.colorbar(mappable=h)

        pass


class Pre_Process:

    def __init__(self):
        pass

    def run(self):

        pass

    def data_transform(self, fdir, outdir, n=10000):
        n = int(n)
        Tools().mkdir(outdir)
        # per_pix_data
        flist = Tools().listdir(fdir)
        date_list = []
        for f in flist:
            if f.endswith('.tif'):
                date = f.split('.')[0]
                date_list.append(date)
        date_list.sort()
        all_array = []
        for d in tqdm(date_list, 'loading...'):
            # for d in date_list:
            for f in flist:
                if f.endswith('.tif'):
                    if f.split('.')[0] == d:
                        # print(d)
                        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir, f))
                        array = np.array(array, dtype=float)
                        all_array.append(array)

        row = len(all_array[0])
        col = len(all_array[0][0])
        all_array = np.array(all_array)
        all_array_T = all_array.T
        void_dic = {}
        void_dic_list = []
        for r in tqdm(list(range(row))):
            for c in range(col):
                time_series = all_array_T[c][r]
                time_series = np.array(time_series)
                # if np.nanstd(time_series) == 0:
                #     continue
                # time_series[time_series<-99999] = np.nan
                # time_series = time_series[~np.isnan(time_series)]
                # if len(time_series) == 0:
                #     continue
                void_dic_list.append((r, c))
                void_dic[(r, c)] = time_series
        flag = 0
        temp_dic = {}
        for key in tqdm(void_dic_list, 'saving...'):
            flag += 1
            # print('saving ',flag,'/',len(void_dic)/100000)
            arr = void_dic[key]
            arr = np.array(arr)
            temp_dic[key] = arr
            if flag % n == 0:
                # print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + '/per_pix_dic_%03d' % (flag / n), temp_dic)
                temp_dic = {}
        np.save(outdir + '/per_pix_dic_%03d' % 0, temp_dic)

    def data_transform_with_date_list(self, fdir, outdir, date_list, n=10000):
        n = int(n)
        Tools().mkdir(outdir)
        outdir = outdir + '/'
        template_f = os.path.join(fdir, Tools().listdir(fdir)[0])
        template_arr = ToRaster().raster2array(template_f)[0]
        void_arr = np.ones_like(template_arr) * np.nan
        all_array = []
        invalid_f_num = 0
        for d in tqdm(date_list, 'loading...'):
            f = os.path.join(fdir, d)
            if os.path.isfile(f):
                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
                array = np.array(array, dtype=float)
                all_array.append(array)
            else:
                all_array.append(void_arr)
                invalid_f_num += 1
        print('\n', 'invalid_f_num:', invalid_f_num)
        # exit()

        row = len(all_array[0])
        col = len(all_array[0][0])
        all_array = np.array(all_array)
        all_array_T = all_array.T
        void_dic = {}
        void_dic_list = []
        for r in tqdm(list(range(row))):
            for c in range(col):
                time_series = all_array_T[c][r]
                time_series = np.array(time_series)
                # if np.nanstd(time_series) == 0:
                #     continue
                # time_series[time_series<-99999] = np.nan
                # time_series = time_series[~np.isnan(time_series)]
                # if len(time_series) == 0:
                #     continue
                void_dic_list.append((r, c))
                void_dic[(r, c)] = time_series

        # for i in void_dic_list:
        #     print(i)
        # exit()
        flag = 0
        temp_dic = {}
        for key in tqdm(void_dic_list, 'saving...'):
            flag += 1
            # print('saving ',flag,'/',len(void_dic)/100000)
            arr = void_dic[key]
            arr = np.array(arr)
            temp_dic[key] = arr
            if flag % n == 0:
                # print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + 'per_pix_dic_%03d' % (flag / n), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

    def kernel_cal_anomaly(self, params):
        fdir, f, save_dir = params
        fpath = Tools().path_join(fdir, f)
        pix_dic = Tools().load_npy(fpath)
        anomaly_pix_dic = {}
        for pix in pix_dic:
            ####### one pix #######
            vals = pix_dic[pix]
            vals = np.array(vals)
            vals = Tools().mask_999999_arr(vals, warning=False)
            # 清洗数据
            climatology_means = []
            climatology_std = []
            # vals = signal.detrend(vals)
            for m in range(1, 13):
                one_mon = []
                for i in range(len(pix_dic[pix])):
                    mon = i % 12 + 1
                    if mon == m:
                        one_mon.append(pix_dic[pix][i])
                mean = np.nanmean(one_mon)
                std = np.nanstd(one_mon)
                climatology_means.append(mean)
                climatology_std.append(std)

            # 算法1
            # pix_anomaly = {}
            # for m in range(1, 13):
            #     for i in range(len(pix_dic[pix])):
            #         mon = i % 12 + 1
            #         if mon == m:
            #             this_mon_mean_val = climatology_means[mon - 1]
            #             this_mon_std_val = climatology_std[mon - 1]
            #             if this_mon_std_val == 0:
            #                 anomaly = -999999
            #             else:
            #                 anomaly = (pix_dic[pix][i] - this_mon_mean_val) / float(this_mon_std_val)
            #             key_anomaly = i
            #             pix_anomaly[key_anomaly] = anomaly
            # arr = pandas.Series(pix_anomaly)
            # anomaly_list = arr.to_list()
            # anomaly_pix_dic[pix] = anomaly_list

            # 算法2
            pix_anomaly = []
            for i in range(len(vals)):
                mon = i % 12
                std_ = climatology_std[mon]
                mean_ = climatology_means[mon]
                if std_ == 0:
                    anomaly = 0  ##### 修改gpp
                else:
                    anomaly = (vals[i] - mean_) / std_

                pix_anomaly.append(anomaly)
            # pix_anomaly = Tools().interp_1d_1(pix_anomaly,-100)
            # plt.plot(pix_anomaly)
            # plt.show()
            pix_anomaly = np.array(pix_anomaly)
            anomaly_pix_dic[pix] = pix_anomaly
        save_f = Tools().path_join(save_dir, f)
        np.save(save_f, anomaly_pix_dic)

    def z_score_climatology(self, vals):
        pix_anomaly = []
        climatology_means = []
        climatology_std = []
        for m in range(1, 13):
            one_mon = []
            for i in range(len(vals)):
                mon = i % 12 + 1
                if mon == m:
                    one_mon.append(vals[i])
            mean = np.nanmean(one_mon)
            std = np.nanstd(one_mon)
            climatology_means.append(mean)
            climatology_std.append(std)
        for i in range(len(vals)):
            mon = i % 12
            std_ = climatology_std[mon]
            mean_ = climatology_means[mon]
            if std_ == 0:
                anomaly = 0
            else:
                anomaly = (vals[i] - mean_) / std_
            pix_anomaly.append(anomaly)
        pix_anomaly = np.array(pix_anomaly)
        return pix_anomaly

    def climatology_anomaly(self, vals):
        '''
        juping
        :param vals:
        :return:
        '''
        pix_anomaly = []
        climatology_means = []
        for m in range(1, 13):
            one_mon = []
            for i in range(len(vals)):
                mon = i % 12 + 1
                if mon == m:
                    one_mon.append(vals[i])
            mean = np.nanmean(one_mon)
            std = np.nanstd(one_mon)
            climatology_means.append(mean)
        for i in range(len(vals)):
            mon = i % 12
            mean_ = climatology_means[mon]
            anomaly = vals[i] - mean_
            pix_anomaly.append(anomaly)
        pix_anomaly = np.array(pix_anomaly)
        return pix_anomaly

    def climatology_percentage(self, vals):
        '''
        percentage
        :param vals:
        :return:
        '''
        pix_percentage = []
        climatology_means = []
        for m in range(1, 13):
            one_mon = []
            for i in range(len(vals)):
                mon = i % 12 + 1
                if mon == m:
                    one_mon.append(vals[i])
            mean = np.nanmean(one_mon)
            climatology_means.append(mean)
        for i in range(len(vals)):
            mon = i % 12
            mean_ = climatology_means[mon]
            percentage = vals[i] / mean_ * 100 - 100
            pix_percentage.append(percentage)
        pix_percentage = np.array(pix_percentage)
        return pix_percentage

    def climotology_mean_std(self, vals):
        result_dict = {}
        climatology_means = []
        climatology_std = []
        for m in range(1, 13):
            one_mon = []
            for i in range(len(vals)):
                mon = i % 12 + 1
                if mon == m:
                    one_mon.append(vals[i])
            mean = np.nanmean(one_mon)
            std = np.nanstd(one_mon)
            climatology_means.append(mean)
            climatology_std.append(std)
            result_dict[m] = {'mean': mean, 'std': std}
        return result_dict

    def z_score(self, vals):
        vals = np.array(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        if std == 0:
            anomaly = 0
        else:
            anomaly = (vals - mean) / std
        return anomaly

    def cal_anomaly_juping(self, vals):
        mean = np.nanmean(vals)
        vals = np.array(vals)
        anomaly = vals - mean
        return anomaly

    def cal_relative_change(self, vals):
        relative_change_list = []
        mean = np.nanmean(vals)
        for v in vals:
            relative_change = (v - mean) / v
            relative_change_list.append(relative_change)
        relative_change_list = np.array(relative_change_list)
        return relative_change_list

    def cal_anomaly(self, fdir, save_dir):
        # fdir = this_root + 'NDVI/per_pix/'
        # save_dir = this_root + 'NDVI/per_pix_anomaly/'
        Tools().mkdir(save_dir)
        flist = Tools().listdir(fdir)
        # flag = 0
        params = []
        for f in flist:
            # print(f)
            params.append([fdir, f, save_dir])

        # for p in params:
        #     print(p[1])
        #     self.kernel_cal_anomaly(p)
        MULTIPROCESS(self.kernel_cal_anomaly, params).run(process=4, process_or_thread='p',
                                                          desc='calculating anomaly...')

    def clean_per_pix(self, fdir, outdir, mode='linear'):
        # mode = climatology
        Tools().mkdir(outdir)
        for f in tqdm(Tools().listdir(fdir)):
            dic = Tools().load_npy(fdir + '/' + f)
            clean_dic = {}
            for pix in dic:
                val = dic[pix]
                val = np.array(val, dtype=float)
                val[val < -9999] = np.nan
                if mode == 'linear':
                    new_val = Tools().interp_nan(val, kind='linear')
                elif mode == 'climatology':
                    new_val = Tools().interp_nan_climatology(val)
                else:
                    raise UserWarning('mode error')
                if len(new_val) == 1:
                    continue
                # plt.plot(val)
                # plt.show()
                clean_dic[pix] = new_val
            np.save(outdir + '/' + f, clean_dic)
        pass

    def detrend(self, fdir, outdir):
        Tools().mkdir(outdir)
        for f in tqdm(Tools().listdir(fdir), desc='detrend...'):
            dic = Tools().load_npy(join(fdir, f))
            dic_detrend = Tools().detrend_dic(dic)
            outf = join(outdir, f)
            Tools().save_npy(dic_detrend, outf)
        pass

    def compose_tif_list(self, flist, outf, less_than=-9999, method='mean'):
        # less_than -9999, mask as np.nan
        if len(flist) == 0:
            return
        tif_template = flist[0]
        void_dic = DIC_and_TIF(tif_template=tif_template).void_spatial_dic()
        for f in tqdm(flist, desc='transforming...'):
            if not f.endswith('.tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
            for r in range(len(array)):
                for c in range(len(array[0])):
                    pix = (r, c)
                    val = array[r][c]
                    void_dic[pix].append(val)
        spatial_dic = {}
        for pix in tqdm(void_dic, desc='calculating mean...'):
            vals = void_dic[pix]
            vals = np.array(vals, dtype=float)
            vals[vals < less_than] = np.nan
            if method == 'mean':
                compose_val = np.nanmean(vals)
            elif method == 'max':
                compose_val = np.nanmax(vals)
            elif method == 'sum':
                compose_val = np.nansum(vals)
            else:
                raise UserWarning(f'{method} is invalid, should be "mean" "max" or "sum"')
            spatial_dic[pix] = compose_val
        DIC_and_TIF(tif_template=tif_template).pix_dic_to_tif(spatial_dic, outf)

    def get_year_month_day(self, fname, date_fmt='yyyymmdd'):
        try:
            if date_fmt == 'yyyymmdd':
                fname_split = fname.split('.')
                if not len(fname_split) == 2:
                    raise
                date = fname_split[0]
                if not len(date) == 8:
                    raise
                date_int = int(date)
                y = date[:4]
                m = date[4:6]
                d = date[6:]
                y = int(y)
                m = int(m)
                d = int(d)
                date_obj = datetime.datetime(y, m, d)  # check date availability
                return y, m, d
            elif date_fmt == 'doy':
                fname_split = fname.split('.')
                if not len(fname_split) == 2:
                    raise
                date = fname_split[0]
                if not len(date) == 7:
                    raise
                y = date[:4]
                doy = date[4:]
                doy = int(doy)
                date_base = datetime.datetime(int(y), 1, 1)
                time_delta = datetime.timedelta(doy - 1)
                date_obj = date_base + time_delta
                y = date_obj.year
                m = date_obj.month
                d = date_obj.day
                return y, m, d
        except:
            if date_fmt == 'yyyymmdd':
                raise UserWarning(
                    f'------\nfname must be yyyymmdd.tif e.g. 19820101.tif\nplease check your fname "{fname}"')
            elif date_fmt == 'doy':
                raise UserWarning(
                    f'------\nfname must be yyyyddd.tif e.g. 1982001.tif\nplease check your fname "{fname}"')

    def monthly_compose(self, indir, outdir, date_fmt='yyyymmdd', method='mean'):
        '''
        :param method: "mean", "max" or "sum"
        :param date_fmt: 'yyyymmdd' or 'doy'
        :return:
        '''
        Tools().mkdir(outdir)
        year_list = []
        month_list = []
        for f in Tools().listdir(indir):
            y, m, d = self.get_year_month_day(f, date_fmt=date_fmt)
            year_list.append(y)
            month_list.append(m)
        year_list = Tools().drop_repeat_val_from_list(year_list)
        month_list = Tools().drop_repeat_val_from_list(month_list)
        compose_path_dic = {}
        for y in year_list:
            for m in month_list:
                date = (y, m)
                compose_path_dic[date] = []
        for f in Tools().listdir(indir):
            y, m, d = self.get_year_month_day(f, date_fmt=date_fmt)
            date = (y, m)
            compose_path_dic[date].append(join(indir, f))
        for date in compose_path_dic:
            flist = compose_path_dic[date]
            y, m = date
            print(f'{y}{m:02d}')
            outfname = f'{y}{m:02d}.tif'
            outpath = join(outdir, outfname)
            self.compose_tif_list(flist, outpath, method=method)

    def time_series_dic_to_tif(self, spatial_dic, tif_template, outf_list):
        for i in tqdm(range(len(outf_list))):
            outf = outf_list[i]
            spatial_dic_i = {}
            for pix in spatial_dic:
                vals = spatial_dic[pix]
                val = vals[i]
                spatial_dic_i[pix] = val
            arr = DIC_and_TIF(tif_template=tif_template).pix_dic_to_spatial_arr(spatial_dic_i)
            DIC_and_TIF(tif_template=tif_template).arr_to_tif(arr, outf)


class Plot:
    def __init__(self):

        pass

    def plot_line_with_gradient_error_band(self, x, y, yerr, color_gradient_n=100, c=None,
                                           pow=2, min_alpha=0, max_alpha=1, **kwargs):
        x = np.array(x)
        y = np.array(y)
        yerr = np.array(yerr)
        alpha_range_ = np.linspace(min_alpha, math.pow(max_alpha, int(pow)), int(color_gradient_n / 2))
        alpha_range_ = alpha_range_ ** pow
        alpha_range__ = alpha_range_[::-1]
        alpha_range = np.hstack((alpha_range_, alpha_range__))
        bottom = []
        top = []
        for i in range(len(x)):
            b = y[i] - yerr[i]
            t = y[i] + yerr[i]
            bins_i = np.linspace(b, t, color_gradient_n)
            bottom_i = []
            top_i = []
            for j in range(len(bins_i)):
                if j + 1 >= len(bins_i):
                    break
                bottom_i.append(bins_i[j])
                top_i.append(bins_i[j + 1])
            bottom.append(bottom_i)
            top.append(top_i)
        bottom = np.array(bottom)
        top = np.array(top)
        bottom = bottom.T
        top = top.T
        for i in range(color_gradient_n - 1):
            plt.fill_between(x, bottom[i], top[i], alpha=alpha_range[i], zorder=-99,
                             color=c, edgecolor=None, **kwargs)
        pass

    def plot_line_with_error_bar(self, x, y, yerr, c=None, alpha=0.2, **kwargs):

        x = np.array(x)
        y = np.array(y)
        yerr = np.array(yerr)
        y1 = y + yerr
        y2 = y - yerr
        plt.fill_between(x, y1, y2, zorder=-99,
                         color=c, edgecolor=None, alpha=0.2, **kwargs)

    def plot_hist_smooth(self, arr, interpolate_window=5, range=None, **kwargs):
        weights = np.ones_like(arr) / float(len(arr))

        n1, x1, patch = plt.hist(arr, weights=weights, range=range, **kwargs)
        arr = np.array(arr)
        if range is None:
            min_v = np.nanmin(arr)
            max_v = np.nanmax(arr)
        else:
            min_v = range[0]
            max_v = range[1]
        arr[arr > max_v] = np.nan
        arr[arr < min_v] = np.nan
        arr = arr[~np.isnan(arr)]
        density1 = stats.gaussian_kde(arr)
        y1 = density1(x1)
        coe = np.max(n1) / np.max(y1)
        y1 = y1 * coe
        y1 = SMOOTH().smooth_convolve(y1, interpolate_window)
        return x1, y1

    def plot_hist_smooth_colorful(self, arr, palette='PiYG',color_range=None,hist_range=None, **kwargs):

        if hist_range is None:
            min_v = np.nanmin(arr)
            max_v = np.nanmax(arr)
        else:
            min_v = hist_range[0]
            max_v = hist_range[1]
        arr[arr > max_v] = np.nan
        arr[arr < min_v] = np.nan
        arr = arr[~np.isnan(arr)]

        weights = np.ones_like(arr) / float(len(arr))
        n1,x1,patches = plt.hist(arr,weights=weights,range=hist_range,**kwargs)
        color_list = Tools().gen_colors(len(patches),palette=palette)

        if not color_range is None:
            range_min = color_range[0]
            range_max = color_range[-1]
        else:
            range_min = np.nanmin(arr)
            range_max = np.nanmax(arr)

        left = 0
        right = 0
        for i, p in enumerate(patches):
            x_pos = p.xy[0]
            if x_pos < range_min:
                left += 1
            elif x_pos > range_max:
                right += 1
            p.set_facecolor(color_list[i])

        color_list_new = Tools().gen_colors(len(patches) - left - right, palette=palette)
        for i,p in enumerate(patches):
            x_pos = p.xy[0]
            if x_pos < range_min:
                p.set_facecolor(color_list[0])
                continue
            elif x_pos > range_max:
                p.set_facecolor(color_list[-1])
                continue
            else:
                p.set_facecolor(color_list_new[i-left])

    def multi_step_pdf(self, df, pdf_colname, bin_colname, bins, pdf_bin_n=40, pdf_range=None, discard_limit_n=10,
                       blend_color_list=('#ABD3E0', '#EFDDE2', '#DC9AA2')):
        flag = 0
        df_group, bins_list_str = Tools().df_bin(df, bin_colname, bins)
        color_len = 0
        for name, df_group_i in df_group:
            vals = df_group_i[pdf_colname].tolist()
            vals = np.array(vals)
            if len(vals) <= discard_limit_n:
                continue
            color_len += 1
        # exit()
        color_list = Tools().cmap_blend(blend_color_list, as_cmap=False, n_colors=color_len)
        alpha_list = np.linspace(0.5, 1, color_len)
        fig, ax = plt.subplots(1)
        delta_y = 0.1
        label_list = []
        y_tick_list = []
        vals_mean_list = []
        for name, df_group_i in df_group:
            label = str(name.left) + '-' + str(name.right)
            vals = df_group_i[pdf_colname].tolist()
            vals = np.array(vals)
            print(bin_colname, label, len(vals))
            if len(vals) <= discard_limit_n:
                continue
            vals_mean = np.nanmedian(vals)
            vals_mean_list.append(vals_mean)
            x, y = Plot().plot_hist_smooth(vals, bins=pdf_bin_n, alpha=0., range=pdf_range, color=color_list[flag],
                                           linewidth=2)
            # ax.plot(x, y, label=label, color=color_list[flag])
            ax.plot(x, y + delta_y * flag, color='k', zorder=100)
            y_tick_list.append(delta_y * flag)
            ax.fill(x, y + delta_y * flag, color=color_list[flag], label=label, zorder=-flag)
            label_list.append(label)
            flag += 1
        # plt.legend()
        plt.xlabel(pdf_colname)
        plt.ylabel(bin_colname)
        plt.yticks(y_tick_list, label_list)
        y_tick_list = np.array(y_tick_list)
        plt.scatter(vals_mean_list, y_tick_list + 0.05, color='k', zorder=100)
        # plt.tight_layout()
        # plt.show()

    def plot_ortho(self, fpath, ax=None, cmap=None, vmin=None, vmax=None, is_plot_colorbar=True, is_reproj=True,is_discrete=False,colormap_n=11):
        '''
        :param fpath: tif file
        :param is_reproj: if True, reproject file from 4326 to ortho
        '''
        color_list = [
            '#844000',
            '#fc9831',
            '#fffbd4',
            '#86b9d2',
            '#064c6c',
        ]
        # Blue represents high values, and red represents low values.
        if ax == None:
            ax = plt.subplot(1, 1, 1)
        if cmap is None:
            cmap = Tools().cmap_blend(color_list[::-1])
        if not is_reproj:
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        else:
            fpath_ortho = self.ortho_reproj(fpath, fpath + '_ortho-reproj.tif')
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_ortho)
            os.remove(fpath_ortho)
        originY1 = copy.copy(originY)
        arr = Tools().mask_999999_arr(arr, warning=False)
        arr_m = ma.masked_where(np.isnan(arr), arr)
        originX = 0
        originY = originY * 2
        lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
        lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
        lon_list, lat_list = np.meshgrid(lon_list, lat_list)
        m = Basemap(projection='ortho', lon_0=0, lat_0=90., ax=ax, resolution='l')
        ret = m.pcolormesh(lon_list, lat_list, arr_m, cmap=cmap, zorder=99, vmin=vmin, vmax=vmax)
        clip_circle = mpatches.Circle(xy=(originY1, originY1), radius=originY1 * np.cos(np.pi / 6),
                                      facecolor='None', edgecolor='k', zorder=100, lw=1.5)
        clip_circle1 = mpatches.Circle(xy=(originY1, originY1), radius=originY1,
                                       facecolor='None', edgecolor='w', zorder=100, lw=10)
        ax.add_patch(clip_circle)
        ax.add_patch(clip_circle1)
        m.drawparallels(np.arange(30., 90., 30.), zorder=99, dashes=[2, 2], linewidth=0.5)
        meridict = m.drawmeridians(np.arange(0., 420., 60.), zorder=99, latmax=90, dashes=[2, 2], linewidth=0.5)
        for obj in meridict:
            line = meridict[obj][0][0]
            line.set_clip_path(clip_circle.get_path(), clip_circle.get_transform())
        limb = m.drawmapboundary(fill_color='#EFEFEF', zorder=0)
        limb.set_clip_path(clip_circle.get_path(), clip_circle.get_transform())
        coastlines = m.drawcoastlines(zorder=99, linewidth=0.5)
        coastlines.set_clip_path(clip_circle.get_path(), clip_circle.get_transform())
        ret.set_clip_path(clip_circle.get_path(), clip_circle.get_transform())
        polys = m.fillcontinents(color='#B1B0B1', lake_color='#EFEFEF')
        for poly in polys:
            poly.set_clip_path(clip_circle.get_path(), clip_circle.get_transform())
        if is_plot_colorbar:
            if is_discrete:
                bounds = np.linspace(vmin, vmax, colormap_n)
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
                # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                cax, kw = mpl.colorbar.make_axes(ax, location='bottom', pad=0, shrink=0.5)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds,
                                                 orientation='horizontal')
            else:
                cbar = plt.colorbar(ret, ax=ax, shrink=0.5, location='bottom', pad=0)

        return m, ret

    def plot_ortho_significance_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5, s=20, c='k', marker='x',
                                        zorder=100, res=2):
        fpath_clip = fpath_p + 'clip.tif'
        fpath_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath_p)
        D_clip = DIC_and_TIF(tif_template=fpath_p)
        D_clip_lon_lat_pix_dict = D_clip.spatial_tif_to_lon_lat_dic(temp_root)
        fpath_clip_spatial_dict_clipped = {}
        for pix in fpath_spatial_dict:
            lon, lat = D_clip_lon_lat_pix_dict[pix]
            if lat <= 30 + res:
                continue
            fpath_clip_spatial_dict_clipped[pix] = fpath_spatial_dict[pix]
        DIC_and_TIF().pix_dic_to_tif(fpath_clip_spatial_dict_clipped, fpath_clip)
        fpath_resample = fpath_clip + 'resample.tif'
        ToRaster().resample_reproj(fpath_clip, fpath_resample, res=res,srcSRS=None, dstSRS=None)
        fpath_resample_ortho = fpath_resample + 'ortho.tif'
        self.ortho_reproj(fpath_resample, fpath_resample_ortho, res=res * 100000)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample_ortho)

        arr = Tools().mask_999999_arr(arr, warning=False)
        arr[arr > sig_level] = np.nan
        D_resample = DIC_and_TIF(tif_template=fpath_resample_ortho)

        os.remove(fpath_clip)
        os.remove(fpath_resample_ortho)
        os.remove(fpath_resample)

        spatial_dict = D_resample.spatial_arr_to_dic(arr)
        lon_lat_pix_dict = D_resample.spatial_tif_to_lon_lat_dic(temp_root)

        lon_list = []
        lat_list = []
        for pix in spatial_dict:
            val = spatial_dict[pix]
            if np.isnan(val):
                continue
            lon, lat = lon_lat_pix_dict[pix]
            lon_list.append(lon)
            lat_list.append(lat)
        lon_list = np.array(lon_list)
        lat_list = np.array(lat_list)
        lon_list = lon_list - originX
        lat_list = lat_list + originY
        lon_list = lon_list + pixelWidth / 2
        lat_list = lat_list + pixelHeight / 2
        # m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=0.5)
        m.scatter(lon_list, lat_list, latlon=False, s=s, c=c, zorder=zorder, marker=marker, ax=ax,linewidths=linewidths)

        return m

    def plot_Robinson_significance_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5, s=20,
                                        c='k', marker='x',
                                        zorder=100, res=2):

        fpath_clip = fpath_p + 'clip.tif'
        fpath_spatial_dict = DIC_and_TIF(tif_template=fpath_p).spatial_tif_to_dic(fpath_p)
        D_clip = DIC_and_TIF(tif_template=fpath_p)
        D_clip_lon_lat_pix_dict = D_clip.spatial_tif_to_lon_lat_dic(temp_root)
        fpath_clip_spatial_dict_clipped = {}
        for pix in fpath_spatial_dict:
            lon, lat = D_clip_lon_lat_pix_dict[pix]
            fpath_clip_spatial_dict_clipped[pix] = fpath_spatial_dict[pix]
        DIC_and_TIF(tif_template=fpath_p).pix_dic_to_tif(fpath_clip_spatial_dict_clipped, fpath_clip)
        fpath_resample = fpath_clip + 'resample.tif'
        ToRaster().resample_reproj(fpath_clip, fpath_resample, res=res)
        fpath_resample_ortho = fpath_resample + 'Robinson.tif'
        self.Robinson_reproj(fpath_resample, fpath_resample_ortho, res=res * 100000)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample_ortho)

        arr = Tools().mask_999999_arr(arr, warning=False)
        arr[arr > sig_level] = np.nan
        D_resample = DIC_and_TIF(tif_template=fpath_resample_ortho)
        #
        os.remove(fpath_clip)
        os.remove(fpath_resample_ortho)
        os.remove(fpath_resample)

        spatial_dict = D_resample.spatial_arr_to_dic(arr)
        lon_lat_pix_dict = D_resample.spatial_tif_to_lon_lat_dic(temp_root)

        lon_list = []
        lat_list = []
        for pix in spatial_dict:
            val = spatial_dict[pix]
            if np.isnan(val):
                continue
            lon, lat = lon_lat_pix_dict[pix]
            lon_list.append(lon)
            lat_list.append(lat)
        lon_list = np.array(lon_list)
        lat_list = np.array(lat_list)
        lon_list = lon_list - originX
        lat_list = lat_list + originY
        lon_list = lon_list + pixelWidth / 2
        lat_list = lat_list + pixelHeight / 2
        # m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=0.5)
        m.scatter(lon_list, lat_list, latlon=False, s=s, c=c, zorder=zorder, marker=marker, ax=ax,
                  linewidths=linewidths)

        return m

    def ortho_wkt(self):
        wkt = '''
        PROJCRS["North_Pole_Orthographic",
    BASEGEOGCRS["WGS 84",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["North_Pole_Orthographic",
        METHOD["Orthographic (Spherical)"],
        PARAMETER["Latitude of natural origin",90,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8801]],
        PARAMETER["Longitude of natural origin",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["False easting",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1]],
    USAGE[
        SCOPE["Not known."],
        AREA["Northern hemisphere."],
        BBOX[0,-180,90,180]],
    ID["ESRI",102035]]'''
        return wkt

    def ortho_reproj(self, fpath, outf, res=50000):
        wkt = self.ortho_wkt()
        wkt84 = DIC_and_TIF().wkt_84()
        dstSRS = DIC_and_TIF().gen_srs_from_wkt(wkt)
        srcSRS = DIC_and_TIF().gen_srs_from_wkt(wkt84)
        ToRaster().resample_reproj(fpath, outf, res, srcSRS,dstSRS)
        return outf

    def plot_Robinson(self, fpath, ax=None, cmap=None, vmin=None, vmax=None, is_plot_colorbar=True, is_reproj=True,res=25000,is_discrete=False,colormap_n=11):
        '''
        :param fpath: tif file
        :param is_reproj: if True, reproject file from 4326 to Robinson
        :param res: resolution, meter
        '''
        color_list = [
            '#844000',
            '#fc9831',
            '#fffbd4',
            '#86b9d2',
            '#064c6c',
        ]
        # Blue represents high values, and red represents low values.
        if ax == None:
            # plt.figure(figsize=(10, 10))
            ax = plt.subplot(1, 1, 1)
        if cmap is None:
            cmap = Tools().cmap_blend(color_list)
        if not is_reproj:
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        else:
            fpath_robinson = self.Robinson_reproj(fpath, fpath + '_robinson-reproj.tif',res=res)
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_robinson)
            os.remove(fpath_robinson)
        originY1 = copy.copy(originY)
        arr = Tools().mask_999999_arr(arr, warning=False)
        arr_m = ma.masked_where(np.isnan(arr), arr)
        originX = 0
        originY = originY * 2
        lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
        lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
        lon_list, lat_list = np.meshgrid(lon_list, lat_list)
        m = Basemap(projection='robin', lon_0=0, lat_0=90., ax=ax, resolution='i')
        ret = m.pcolormesh(lon_list, lat_list, arr_m, cmap=cmap, zorder=99, vmin=vmin, vmax=vmax,)
        m.drawparallels(np.arange(-60., 90., 30.), zorder=99, dashes=[8, 8], linewidth=.5)
        m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        meridict = m.drawmeridians(np.arange(0., 420., 60.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        for obj in meridict:
            line = meridict[obj][0][0]
        coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        polys = m.fillcontinents(color='#D1D1D1', lake_color='#EFEFEF',zorder=90)
        if is_plot_colorbar:
            if is_discrete:
                bounds = np.linspace(vmin, vmax, colormap_n)
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
                # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05,shrink=0.5)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, orientation='horizontal')
            else:
                cbar = plt.colorbar(ret, ax=ax, shrink=0.5, location='bottom', pad=0.05)
        return m, ret

    def Robinson_reproj(self, fpath, outf, res=50000):
        wkt = self.Robinson_wkt()
        srs = DIC_and_TIF().gen_srs_from_wkt(wkt)
        ToRaster().resample_reproj(fpath, outf, res, dstSRS=srs)
        return outf

    def Robinson_wkt(self):
        wkt = '''
        PROJCRS["Sphere_Robinson",
    BASEGEOGCRS["Unknown datum based upon the Authalic Sphere",
        DATUM["Not specified (based on Authalic Sphere)",
            ELLIPSOID["Sphere",6371000,0,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["Sphere_Robinson",
        METHOD["Robinson"],
        PARAMETER["Longitude of natural origin",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["False easting",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1]],
    USAGE[
        SCOPE["Not known."],
        AREA["World."],
        BBOX[-90,-180,90,180]],
    ID["ESRI",53030]]'''
        return wkt

    def plot_China_Albers(self, fpath, in_shpfile,shp_provinces_f,ax=None, cmap=None, vmin=None, vmax=None, is_plot_colorbar=True, is_reproj=True,res=10000,is_discrete=False,colormap_n=11):
        '''
        :param fpath: tif file
        :param is_reproj: if True, reproject file from 4326 to Robinson
        :param res: resolution, meter
        '''
        color_list = [
            '#844000',
            '#fc9831',
            '#fffbd4',
            '#86b9d2',
            '#064c6c',
        ]
        # Blue represents high values, and red represents low values.
        if ax == None:
            # plt.figure(figsize=(10, 10))
            ax = plt.subplot(1, 1, 1)
        if cmap is None:
            cmap = Tools().cmap_blend(color_list)
        arr_deg, originX_deg, originY_deg, pixelWidth_deg, pixelHeight_deg = ToRaster().raster2array(fpath)
        llcrnrlon = originX_deg
        urcrnrlat = originY_deg
        urcrnrlon = originX_deg + pixelWidth_deg * arr_deg.shape[1]
        llcrnrlat = originY_deg + pixelHeight_deg * arr_deg.shape[0]
        arr_deg = Tools().mask_999999_arr(arr_deg, warning=False)
        arr_m = ma.masked_where(np.isnan(arr_deg), arr_deg)
        # exit()
        lon_list = np.arange(originX_deg, originX_deg +  pixelWidth_deg * arr_deg.shape[1], pixelWidth_deg)
        lat_list = np.arange(originY_deg, originY_deg + pixelHeight_deg * arr_deg.shape[0], pixelHeight_deg)
        lat_list = lat_list + pixelHeight_deg / 2
        lon_list = lon_list + pixelWidth_deg / 2

        m = Basemap(projection='aea', ax=ax, resolution='i',
                    llcrnrlon=80, llcrnrlat=14, urcrnrlon=140, urcrnrlat=52,
                    lon_0=105,lat_0=0,lat_1=25,lat_2=47)
        lon_matrix = []
        lat_matrix = []
        for lon in tqdm(lon_list):
            lon_matrix_i = []
            lat_matrix_i = []
            for lat in lat_list:
                # print(lon,lat)
                lon_projtran, lat_projtran = m.projtran(lon,lat)
                lon_matrix_i.append(lon_projtran)
                lat_matrix_i.append(lat_projtran)

            lon_matrix.append(lon_matrix_i)
            lat_matrix.append(lat_matrix_i)
        lon_matrix = np.array(lon_matrix)
        lat_matrix = np.array(lat_matrix)

        ret = m.pcolormesh(lon_matrix, lat_matrix, arr_deg.T, cmap=cmap, zorder=99, vmin=vmin, vmax=vmax)
        shp_f = in_shpfile
        m.readshapefile(shp_f,'a', drawbounds=True, linewidth=0.5, color='k', zorder=100)
        m.readshapefile(shp_provinces_f, 'ooo', drawbounds=True, linewidth=0.3, color='k', zorder=100)
        # m.drawparallels(np.arange(-60., 90., 20.), zorder=99, dashes=[8, 8], linewidth=.5)
        # m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        # meridict = m.drawmeridians(np.arange(0., 420., 20.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        # meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        plt.axis('off')

        # for obj in meridict:
        #     line = meridict[obj][0][0]
        # coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        # polys = m.fillcontinents(color='#D1D1D1', lake_color='#EFEFEF',zorder=90)
        if is_plot_colorbar:
            if is_discrete:
                bounds = np.linspace(vmin, vmax, colormap_n)
                # norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05,shrink=0.5)
                # cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, orientation='horizontal')
            else:
                cbar = plt.colorbar(ret, ax=ax, shrink=0.5, location='bottom', pad=0.05)
        return m, ret

    def plot_China_Albers_significance_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5, s=20,
                                        c='k', marker='x', zorder=101, res=1.5):

        fpath_spatial_dict = DIC_and_TIF(tif_template=fpath_p).spatial_tif_to_dic(fpath_p)
        D_clip = DIC_and_TIF(tif_template=fpath_p)
        D_clip_lon_lat_pix_dict = D_clip.spatial_tif_to_lon_lat_dic(temp_root)
        fpath_resample = fpath_p + 'resample.tif'
        ToRaster().resample_reproj(fpath_p, fpath_resample, res=res)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample)

        arr = Tools().mask_999999_arr(arr, warning=False)
        arr[arr > sig_level] = np.nan
        D_resample = DIC_and_TIF(tif_template=fpath_resample)
        os.remove(fpath_resample)

        spatial_dict = D_resample.spatial_arr_to_dic(arr)
        lon_lat_pix_dict = D_resample.spatial_tif_to_lon_lat_dic(temp_root)

        lon_list = []
        lat_list = []
        for pix in spatial_dict:
            val = spatial_dict[pix]
            if np.isnan(val):
                continue
            lon, lat = lon_lat_pix_dict[pix]
            lon = lon + pixelWidth / 2
            lat = lat + pixelHeight / 2
            lon_projtran, lat_projtran = m.projtran(lon,lat)
            lon_list.append(lon_projtran)
            lat_list.append(lat_projtran)
        m.scatter(lon_list,lat_list, latlon=False, s=s, c=c, zorder=zorder, marker=marker, ax=ax,
                  linewidths=linewidths)
        return m


class ToRaster:
    def __init__(self):

        pass

    def raster2array(self, rasterfn):
        '''
        create array from raster
        Agrs:
            rasterfn: tiff file path
        Returns:
            array: tiff data, an 2D array
        '''
        raster = gdal.Open(rasterfn)
        geotransform = raster.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
        array = np.asarray(array)
        del raster
        return array, originX, originY, pixelWidth, pixelHeight

    def get_ndv(self, rasterfn):
        raster = gdal.Open(rasterfn)
        NDV = raster.GetRasterBand(1).GetNoDataValue()
        del raster
        return NDV

    def array2raster_GDT_Byte(self, newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = longitude_start
        originY = latitude_start
        # open geotiff
        driver = gdal.GetDriverByName('GTiff')
        if os.path.exists(newRasterfn):
            os.remove(newRasterfn)
        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
        # Add Color Table
        # outRaster.GetRasterBand(1).SetRasterColorTable(ct)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        # Write Date to geotiff
        outband = outRaster.GetRasterBand(1)
        ndv = 255
        outband.SetNoDataValue(ndv)
        outband.WriteArray(array)
        outRasterSRS = osr.SpatialReference()
        # outRasterSRS.ImportFromEPSG(4326)
        wkt_84 = DIC_and_TIF().wkt_84()
        outRasterSRS.ImportFromWkt(wkt_84)
        outRaster.SetProjection(wkt_84)
        # Close Geotiff
        outband.FlushCache()
        del outRaster

    def array2raster(self, newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array, ndv=-999999):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = longitude_start
        originY = latitude_start
        # open geotiff
        driver = gdal.GetDriverByName('GTiff')
        if os.path.exists(newRasterfn):
            os.remove(newRasterfn)
        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
        # outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_UInt16)
        # ndv = 255
        # Add Color Table
        # outRaster.GetRasterBand(1).SetRasterColorTable(ct)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        # Write Date to geotiff
        outband = outRaster.GetRasterBand(1)

        outband.SetNoDataValue(ndv)
        outband.WriteArray(array)
        outRasterSRS = osr.SpatialReference()
        # outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(DIC_and_TIF().wkt_84())
        # Close Geotiff
        outband.FlushCache()
        del outRaster

    def raster2array_multiple_bands(self, rasterfn):
        '''
        create array from raster
        Agrs:
            rasterfn: tiff file path
        Returns:
            array: tiff data, an 2D array
        '''
        raster = gdal.Open(rasterfn)
        geotransform = raster.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        band_num = raster.RasterCount
        band_list = []
        band_name_list = []
        for band in range(1,band_num + 1):
            band = raster.GetRasterBand(band)
            band_name = band.GetDescription()
            band_name_list.append(band_name)
            array = band.ReadAsArray()
            array = np.asarray(array)
            band_list.append(array)
        del raster
        return band_list,band_name_list, originX, originY, pixelWidth, pixelHeight

    def array2raster_polar(self, newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array,
                           ndv=-999999):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = longitude_start
        originY = latitude_start
        # open geotiff
        driver = gdal.GetDriverByName('GTiff')
        if os.path.exists(newRasterfn):
            os.remove(newRasterfn)
        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
        # Add Color Table
        # outRaster.GetRasterBand(1).SetRasterColorTable(ct)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        # Write Date to geotiff
        outband = outRaster.GetRasterBand(1)

        outband.SetNoDataValue(ndv)
        outband.WriteArray(array)
        # outRasterSRS.ImportFromEPSG(4326)
        # outRaster.SetProjection(outRasterSRS.ExportToWkt())
        # ref = osr.SpatialReference()
        # outRasterSRS = osr.SpatialReference()
        # ref_chr = r"PROJCS[\"NSIDC EASE-Grid North\",GEOGCS[\"Unspecified datum based upon the International 1924 Authalic Sphere\",DATUM[\"Not_specified_based_on_International_1924_Authalic_Sphere\",SPHEROID[\"International 1924 Authalic Sphere\",6371228,0,AUTHORITY[\"EPSG\",\"7057\"]],TOWGS84[-9036842.762,25067.525,0,9036842.763000002,0,-25067.525],AUTHORITY[\"EPSG\",\"6053\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4053\"]],PROJECTION[\"Lambert_Azimuthal_Equal_Area\"],PARAMETER[\"latitude_of_center\",90],PARAMETER[\"longitude_of_center\",0],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"X\",EAST],AXIS[\"Y\",NORTH],AUTHORITY[\"EPSG\",\"3408\"]]"
        # ref_chr = r'PROJCS["Grenada 1953 / British West Indies Grid",GEOGCS["Grenada 1953",DATUM["Grenada_1953",SPHEROID["Clarke 1880 (RGS)",6378249.145,293.465,AUTHORITY["EPSG","7012"]],TOWGS84[72,213.7,93,0,0,0,0],AUTHORITY["EPSG","6603"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4603"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-62],PARAMETER["scale_factor",0.9995],PARAMETER["false_easting",400000],PARAMETER["false_northing",0],AUTHORITY["EPSG","2003"],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
        # ref_chr = r'PROJCS["North_Pole_Lambert_Azimuthal_Equal_Area",GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Latitude_Of_Origin",90],UNIT["Meter",1],AUTHORITY["EPSG","9122"]]'
        # ref_chr = "PROJCS['North_Pole_Lambert_Azimuthal_Equal_Area',GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Lambert_Azimuthal_Equal_Area'],PARAMETER['False_Easting',0.0],PARAMETER['False_Northing',0.0],PARAMETER['Central_Meridian',0.0],PARAMETER['Latitude_Of_Origin',90.0],UNIT['Meter',1.0]]"
        # ref_chr = ref_chr.replace('\\','')
        # print ref_chr
        # "PROJCS['North_Pole_Lambert_Azimuthal_Equal_Area',GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Lambert_Azimuthal_Equal_Area'],PARAMETER['False_Easting',0.0],PARAMETER['False_Northing',0.0],PARAMETER['Central_Meridian',0.0],PARAMETER['Latitude_Of_Origin',90.0],UNIT['Meter',1.0]]"
        # outRasterSRS.ImportFromWkt(ref_chr)
        # print outRasterSRS
        # outRaster.SetProjection(outRasterSRS.ExportToWkt())
        # outRaster.SetProjection(ref_chr)
        outband.FlushCache()
        del outRaster

    def shp_to_raster(self, in_shp, output_raster, pixel_size, in_raster_template=None, ndv=-999999):
        input_shp = ogr.Open(in_shp)
        shp_layer = input_shp.GetLayer()
        if in_raster_template:
            raster = gdal.Open(in_raster_template)
            ulx, xres, xskew, uly, yskew, yres = raster.GetGeoTransform()
            lrx = ulx + (raster.RasterXSize * xres)
            lry = uly + (raster.RasterYSize * yres)
            xmin, xmax, ymin, ymax = ulx, lrx, lry, uly
        else:
            xmin, xmax, ymin, ymax = shp_layer.GetExtent()
        ds = gdal.Rasterize(output_raster, in_shp, xRes=pixel_size, yRes=pixel_size,
                            burnValues=1, noData=ndv, outputBounds=[xmin, ymin, xmax, ymax],
                            outputType=gdal.GDT_Byte)
        ds = None
        return output_raster

    def clip_array(self, in_raster, out_raster, in_shp):
        ds = gdal.Open(in_raster)
        ds_clip = gdal.Warp(out_raster, ds, cutlineDSName=in_shp, cropToCutline=True, dstNodata=np.nan)
        ds_clip = None

    # def clip_array(self, in_raster, out_raster, in_shp):
    #     in_array, originX, originY, pixelWidth, pixelHeight = self.raster2array(in_raster)
    #     input_shp = ogr.Open(in_shp)
    #     shp_layer = input_shp.GetLayer()
    #     xmin, xmax, ymin, ymax = shp_layer.GetExtent()
    #     in_shp_encode = in_shp.encode('utf-8')
    #     originX_str = str(originX)
    #     originY_str = str(originY)
    #     pixelWidth_str = str(pixelWidth)
    #     pixelHeight_str = str(pixelHeight)
    #     originX_str = originX_str.encode('utf-8')
    #     originY_str = originY_str.encode('utf-8')
    #     pixelWidth_str = pixelWidth_str.encode('utf-8')
    #     pixelHeight_str = pixelHeight_str.encode('utf-8')
    #     m1 = hashlib.md5(originX_str + originY_str + pixelWidth_str + pixelHeight_str + in_shp_encode)
    #     md5filename = m1.hexdigest() + '.tif'
    #     temp_dir = 'temporary_directory/'
    #     Tools().mkdir(temp_dir)
    #     temp_out_raster = temp_dir + md5filename
    #     if not os.path.isfile(temp_out_raster):
    #         self.shp_to_raster(in_shp, temp_out_raster, pixelWidth, in_raster_template=in_raster, )
    #     rastered_mask_array = self.raster2array(temp_out_raster)[0]
    #     in_mask_arr = np.array(rastered_mask_array)
    #     in_mask_arr[in_mask_arr < -9999] = False
    #     in_mask_arr = np.array(in_mask_arr, dtype=bool)
    #     in_array[~in_mask_arr] = np.nan
    #     lon_list = [xmin, xmax]
    #     lat_list = [ymin, ymax]
    #     pix_list = DIC_and_TIF(tif_template=in_raster).lon_lat_to_pix(lon_list, lat_list, isInt=False)
    #     pix1, pix2 = pix_list
    #     in_array = in_array[pix2[0]:pix1[0]]
    #     in_array = in_array.T
    #     in_array = in_array[pix1[1]:pix2[1]]
    #     in_array = in_array.T
    #     longitude_start, latitude_start = xmin, ymax
    #     self.array2raster(out_raster, longitude_start, latitude_start, pixelWidth, pixelHeight, in_array)

    def mask_array(self, in_raster, out_raster, in_mask_raster):
        in_arr, originX, originY, pixelWidth, pixelHeight = self.raster2array(in_raster)
        in_mask_arr, originX, originY, pixelWidth, pixelHeight = self.raster2array(in_mask_raster)
        in_arr = np.array(in_arr, dtype=float)
        in_mask_arr = np.array(in_mask_arr)
        in_mask_arr[in_mask_arr < -9999] = False
        in_mask_arr = np.array(in_mask_arr, dtype=bool)
        in_arr[~in_mask_arr] = np.nan
        longitude_start, latitude_start, pixelWidth, pixelHeight = originX, originY, pixelWidth, pixelHeight
        self.array2raster(out_raster, longitude_start, latitude_start, pixelWidth, pixelHeight, in_arr)

    def resample_reproj(self, in_tif, out_tif, res, srcSRS=None, dstSRS=None):
        dataset = gdal.Open(in_tif)
        gdal.Warp(out_tif, dataset, xRes=res, yRes=res, srcSRS=srcSRS, dstSRS=dstSRS)

    def resample_majority(self, tif_path,out_path, new_res):
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tif_path)
        col_num,row_num = np.shape(arr)
        print('origin shape:',col_num,row_num)
        new_col_num = abs(round(col_num * (pixelWidth/new_res))) + 1
        new_row_num = abs(round(row_num * (pixelHeight/new_res))) + 1

        print('new shape:',new_col_num-1,new_row_num-1)
        new_col_range = np.linspace(0,col_num,new_col_num)
        new_col_range_int = [int(round(i)) for i in new_col_range]

        new_row_range = np.linspace(0,row_num,new_row_num)
        new_row_range_int = [int(round(i)) for i in new_row_range]

        arr_new = np.ones((new_col_num-1,new_row_num-1)) * np.nan
        for i in tqdm(range(len(new_col_range_int))):
            if i == len(new_col_range_int)-1:
                break
            col_i_left = new_col_range_int[i]
            col_i_right = new_col_range_int[i+1]
            col_vals = arr[col_i_left:col_i_right]
            for j in range(len(new_row_range_int)):
                if j == len(new_row_range_int)-1:
                    break
                col_j_left = new_row_range_int[j]
                col_j_right = new_row_range_int[j+1]
                col_vals_T = col_vals.T
                col_j_vals = col_vals_T[col_j_left:col_j_right]
                values, counts = np.unique(col_j_vals, return_counts=True)
                ind = np.argmax(counts)
                new_val = values[ind]

                arr_new[i,j] = new_val

        ToRaster().array2raster(out_path, originX, originY, new_res, -new_res, arr_new)

class HANTS:

    def __init__(self):
        '''
        HANTS algorithm for time series smoothing
        '''
        pass

    def __left_consecutive_index(self, values_list, invalid_value):
        left_consecutive_non_valid_index = []
        for i in range(len(values_list)):
            if values_list[i] == invalid_value:
                left_consecutive_non_valid_index.append(i)
            else:
                break
        return left_consecutive_non_valid_index

    def __right_consecutive_index(self, values_list, invalid_value):
        right_consecutive_non_valid_index = []
        for i in range(len(values_list) - 1, -1, -1):
            if values_list[i] == invalid_value:
                right_consecutive_non_valid_index.append(i)
            else:
                break
        return right_consecutive_non_valid_index

    def hants_interpolate(self, values_list, dates_list, valid_range, nan_value=np.nan,valid_ratio=0.8,silent=True):
        '''
        :param values_list: 1D, list of values, multi years
        :param dates_list:  1D, list of dates, corresponding to values_list
        :param valid_range: tuple, (low,high), valid range of values
        :param nan_value: float, nan value
        :return: Dict, {year:DOY_values}
        '''
        values_list = np.array(values_list)
        values_list[np.isnan(values_list)] = valid_range[0]
        values_dict = Tools().dict_zip(dates_list, values_list)
        annual_date_dict = {}
        for date in values_dict:
            year = date.year
            if year not in annual_date_dict:
                annual_date_dict[year] = []
            annual_date_dict[year].append(date)
        invalid_year_list = []
        valid_year_list = []
        for year in annual_date_dict:
            date_list_i = annual_date_dict[year]
            # print(date_list_i)
            # exit()
            if len(date_list_i) < len(date_list_i) * valid_ratio:
                if not silent:
                    print(f'{year} has less than {valid_ratio*100:.1f}% valid data, skip')
                invalid_year_list.append(year)
            else:
                valid_year_list.append(year)
        if len(valid_year_list) == 0:
            if not silent:
                print(f'All years have less than {valid_ratio*100:1f}% valid data, skip')
            return {}
        for year in invalid_year_list:
            del annual_date_dict[year]
        # exit()
        _values_list = []
        _dates_list = []
        for year in annual_date_dict:
            date_list_i = []
            for date in annual_date_dict[year]:
                date_list_i.append(date)
            date_list_i.sort()
            value_list_i = []
            for date in date_list_i:
                value_list_i.append(values_dict[date])
            date_list_i = np.array(date_list_i)
            value_list_i = np.array(value_list_i)
            _values_list.append(value_list_i)
            _dates_list.append(date_list_i)
        # _values_list = np.array(_values_list)
        # _dates_list = np.array(_dates_list)
        _values_list_1 = []
        for v in _values_list:
            for vi in v:
                _values_list_1.append(vi)
        std_i = np.nanstd(_values_list_1)
        std = 2. * std_i  # larger than twice the standard deviation of the input data is rejected
        interpolated_values_list = []
        for i, values in enumerate(_values_list):
            dates = _dates_list[i]
            xnew, ynew = self.__interp_values_to_DOY(values, dates)
            # print(xnew)
            # print(ynew)
            # print('---')
            # plt.plot(xnew, ynew, 'o')
            # plt.show()
            interpolated_values_list.append(ynew)

        interpolated_values_list = np.array(interpolated_values_list)
        # print(interpolated_values_list)
        # exit()
        results = self.__hants(sample_count=365, inputs=interpolated_values_list, low=valid_range[0],
                                  high=valid_range[1],
                                  fit_error_tolerance=std)
        results_new = []
        for i in range(len(results)):
            results_i = results[i]
            left_consecutive_non_valid_index = self.__left_consecutive_index(interpolated_values_list[i],
                                                                             valid_range[0])
            # print(left_consecutive_non_valid_index)
            right_consecutive_non_valid_index = self.__right_consecutive_index(interpolated_values_list[i],
                                                                               valid_range[0])
            results_i_new = []
            for j in range(len(results_i)):
                if j in left_consecutive_non_valid_index:
                    results_i_new.append(nan_value)
                elif j in right_consecutive_non_valid_index:
                    results_i_new.append(nan_value)
                else:
                    if results_i[j] < valid_range[0]:
                        results_i_new.append(nan_value)
                        continue
                    elif results_i[j] > valid_range[1]:
                        results_i_new.append(nan_value)
                        continue
                    else:
                        results_i_new.append(results_i[j])
            results_new.append(results_i_new)
        results_dict = Tools().dict_zip(valid_year_list, results_new)
        return results_dict

    def __date_list_to_DOY(self, date_list):
        '''
        :param date_list: list of datetime objects
        :return: list of DOY
        '''
        start_year = date_list[0].year
        start_date = datetime.datetime(start_year, 1, 1)
        date_delta = date_list - start_date + datetime.timedelta(days=1)
        DOY = [date.days for date in date_delta]
        return DOY

    def __interp_values_to_DOY(self, values, date_list):
        DOY = self.__date_list_to_DOY(date_list)
        inx = DOY
        iny = values
        x_new = list(range(1, 366))
        func = interpolate.interp1d(inx, iny, fill_value="extrapolate")
        y_new = func(x_new)
        return x_new, y_new

    def __makediag3d(self, M):
        b = np.zeros((M.shape[0], M.shape[1] * M.shape[1]))
        b[:, ::M.shape[1] + 1] = M
        return b.reshape((M.shape[0], M.shape[1], M.shape[1]))

    def __get_starter_matrix(self, base_period_len, sample_count, frequencies_considered_count):
        nr = min(2 * frequencies_considered_count + 1,
                 sample_count)  # number of 2*+1 frequencies, or number of input images
        mat = np.zeros(shape=(nr, sample_count))
        mat[0, :] = 1
        ang = 2 * np.pi * np.arange(base_period_len) / base_period_len
        cs = np.cos(ang)
        sn = np.sin(ang)
        # create some standard sinus and cosinus functions and put in matrix
        i = np.arange(1, frequencies_considered_count + 1)
        ts = np.arange(sample_count)
        for column in range(sample_count):
            index = np.mod(i * ts[column], base_period_len)
            # index looks like 000, 123, 246, etc, until it wraps around (for len(i)==3)
            mat[2 * i - 1, column] = cs.take(index)
            mat[2 * i, column] = sn.take(index)
        return mat

    def __hants(self, sample_count, inputs,
                frequencies_considered_count=3,
                outliers_to_reject='Hi',
                low=0., high=255,
                fit_error_tolerance=5.,
                delta=0.1):
        """
        Function to apply the Harmonic analysis of time series applied to arrays
        sample_count    = nr. of images (total number of actual samples of the time series)
        base_period_len    = length of the base period, measured in virtual samples
                (days, dekads, months, etc.)
        frequencies_considered_count    = number of frequencies to be considered above the zero frequency
        inputs     = array of input sample values (e.g. NDVI values)
        ts    = array of size sample_count of time sample indicators
                (indicates virtual sample number relative to the base period);
                numbers in array ts maybe greater than base_period_len
                If no aux file is used (no time samples), we assume ts(i)= i,
                where i=1, ..., sample_count
        outliers_to_reject  = 2-character string indicating rejection of high or low outliers
                select from 'Hi', 'Lo' or 'None'
        low   = valid range minimum
        high  = valid range maximum (values outside the valid range are rejeced
                right away)
        fit_error_tolerance   = fit error tolerance (points deviating more than fit_error_tolerance from curve
                fit are rejected)
        dod   = degree of overdeterminedness (iteration stops if number of
                points reaches the minimum required for curve fitting, plus
                dod). This is a safety measure
        delta = small positive number (e.g. 0.1) to suppress high amplitudes
        """
        # define some parameters
        base_period_len = sample_count  #

        # check which setting to set for outlier filtering
        if outliers_to_reject == 'Hi':
            sHiLo = -1
        elif outliers_to_reject == 'Lo':
            sHiLo = 1
        else:
            sHiLo = 0

        nr = min(2 * frequencies_considered_count + 1,
                 sample_count)  # number of 2*+1 frequencies, or number of input images

        # create empty arrays to fill
        outputs = np.zeros(shape=(inputs.shape[0], sample_count))

        mat = self.__get_starter_matrix(base_period_len, sample_count, frequencies_considered_count)

        # repeat the mat array over the number of arrays in inputs
        # and create arrays with ones with shape inputs where high and low values are set to 0
        mat = np.tile(mat[None].T, (1, inputs.shape[0])).T
        p = np.ones_like(inputs)
        p[(low >= inputs) | (inputs > high)] = 0
        nout = np.sum(p == 0, axis=-1)  # count the outliers for each timeseries

        # prepare for while loop
        ready = np.zeros((inputs.shape[0]), dtype=bool)  # all timeseries set to false

        dod = 1  # (2*frequencies_considered_count-1)  # Um, no it isn't :/
        noutmax = sample_count - nr - dod
        for _ in range(sample_count):
            if ready.all():
                break
            # print '--------*-*-*-*',it.value, '*-*-*-*--------'
            # multiply outliers with timeseries
            za = np.einsum('ijk,ik->ij', mat, p * inputs)

            # multiply mat with the multiplication of multiply diagonal of p with transpose of mat
            diag = self.__makediag3d(p)
            A = np.einsum('ajk,aki->aji', mat, np.einsum('aij,jka->ajk', diag, mat.T))
            # add delta to suppress high amplitudes but not for [0,0]
            A = A + np.tile(np.diag(np.ones(nr))[None].T, (1, inputs.shape[0])).T * delta
            A[:, 0, 0] = A[:, 0, 0] - delta

            # solve linear matrix equation and define reconstructed timeseries
            # zr = np.linalg.solve(A, za)
            zr = np.linalg.solve(A, za[..., None])[..., 0]
            outputs = np.einsum('ijk,kj->ki', mat.T, zr)

            # calculate error and sort err by index
            err = p * (sHiLo * (outputs - inputs))
            rankVec = np.argsort(err, axis=1, )

            # select maximum error and compute new ready status
            maxerr = np.diag(err.take(rankVec[:, sample_count - 1], axis=-1))
            ready = (maxerr <= fit_error_tolerance) | (nout == noutmax)

            # if ready is still false
            if not ready.all():
                j = rankVec.take(sample_count - 1, axis=-1)

                p.T[j.T, np.indices(j.shape)] = p.T[j.T, np.indices(j.shape)] * ready.astype(
                    int)  # *check
                nout += 1

        return outputs


class Dataframe_per_value_transform:

    def __init__(self, df, variable_list, start_year, end_year):
        self.start_year = start_year
        self.end_year = end_year
        self.variable_list = variable_list
        self.df_in = df
        self.dataframe_per_value()
        pass

    def init_void_dataframe(self):
        void_spatial_dict = DIC_and_TIF().void_spatial_dic()
        year_list = list(range(self.start_year, self.end_year + 1))
        year_list_all = []
        pix_list_all = []
        for pix in void_spatial_dict:
            for year in year_list:
                year_list_all.append(year)
                pix_list_all.append(pix)
        df = pd.DataFrame()
        df['year'] = year_list_all
        df['pix'] = pix_list_all
        return df, year_list

    def dataframe_per_value(self):
        variable_list = self.variable_list
        df_, year_list = self.init_void_dataframe()
        pix_list = Tools().get_df_unique_val_list(df_, 'pix')
        nan_list = [np.nan] * len(year_list)
        all_data = {}
        for col in variable_list:
            spatial_dict = Tools().df_to_spatial_dic(self.df_in, col)
            all_data[col] = spatial_dict
        for var_i in tqdm(variable_list):
            spatial_dict_i = all_data[var_i]
            val_list_all = []
            for pix in pix_list:
                if not pix in spatial_dict_i:
                    val_list_all.extend(nan_list)
                    continue
                vals = spatial_dict_i[pix]
                if type(vals) == float:
                    val_list_all.extend(nan_list)
                    continue
                if not len(vals) == len(year_list):
                    val_list_all.extend(nan_list)
                    continue
                for i, v in enumerate(vals):
                    val_list_all.append(v)
            df_[var_i] = val_list_all
        df = df_.dropna(subset=variable_list, how='all')
        self.df = df

class Decorator:

    @staticmethod
    def shutup_gdal(func):
        def wrapper(*args, **kwargs):
            gdal.PushErrorHandler('CPLQuietErrorHandler')
            warnings.simplefilter(action='ignore', category=FutureWarning)
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def plt_position(x_offset=1200, y_offset=-1000):
        from screeninfo import get_monitors
        monitors = get_monitors()
        print('------from plt_position decorator-------')
        for m in monitors:
            print(m)
        print('------from plt_position decorator-------')
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                base_x = x_offset
                base_y = y_offset
                fig = func(*args, **kwargs)
                manager = plt.get_current_fig_manager()
                if mpl.get_backend() == 'TkAgg':
                    manager.window.wm_geometry(f"+{base_x}+{base_y}")
                else:
                    pass
                plt.show()
                return fig
            return wrapper
        return decorator

    @staticmethod
    def shutup_np(func):
        def wrapper(*args, **kwargs):
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            return func(*args, **kwargs)

        return wrapper
    pass

    @staticmethod
    def shutup_tqdm(func):
        def wrapper(*args, **kwargs):
            tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
            return func(*args, **kwargs)

        return wrapper


class Tif_loader_Mem:

    def __init__(self,flist,memory_allocate,dtype=None,nodata=None,mute=False):
        self.flist = flist
        self.memory_allocate = memory_allocate
        self.profile = self.get_image_profiles(flist[0])
        if nodata==None:
            pass
        else:
            self.profile.update(nodata=nodata)
        # pprint(profile)
        self.h = self.profile['height']
        self.w = self.profile['width']
        if dtype == None:
            self.dtype = self.profile['dtype']
        else:
            self.dtype = dtype
        self.profile.update(dtype=self.dtype)
        # pprint(self.profile)
        self.available_rows = self.get_available_rows(self.memory_allocate, len(flist), self.h, self.w, dtype=self.dtype)
        self.iter_length = math.ceil(self.h / self.available_rows)
        self.block_index_list = list(range(math.ceil(self.h / self.available_rows)))
        if not mute:
            print('input file size:', f'h:{self.h},w:{self.w}')
            print('input file count:', len(self.flist))
            print('output block size:', f'h:{self.available_rows},w:{self.w}')
            print('output block count:', self.iter_length)
            print('------------------')
        pass

    def array_iterator(self):
        idx = 0
        for row in range(0, self.h, self.available_rows):
            patch_concat_list = []
            for fpath in self.flist:
                with rasterio.open(fpath) as src:
                    patch = src.read(
                        window=((row, row + self.available_rows),(0, self.w))
                    )
                    patch_concat_list.append(patch)
            patch_concat = np.concatenate(patch_concat_list, axis=0)
            patch_concat_list = []

            window = Window(col_off=0, row_off=self.available_rows * idx, width=self.w, height=self.available_rows)
            new_transform = rasterio.windows.transform(window, src.transform)

            profile_new = self.profile.copy()
            profile_new['height'] = self.available_rows
            profile_new['transform'] = new_transform
            transform = profile_new['transform']
            idx += 1
            yield patch_concat, profile_new

    def array_iterator_index(self,idx):

        row_list = list(range(0, self.h, self.available_rows))
        row = row_list[idx]
        patch_concat_list = []
        for fpath in self.flist:
        # for fpath in tqdm(self.flist):
            with rasterio.open(fpath) as src:
                patch = src.read(
                    window=((row, row + self.available_rows),(0, self.w))
                )
                patch_concat_list.append(patch)
        patch_concat = np.concatenate(patch_concat_list, axis=0)
        # print(patch_concat.shape)
        # exit()
        patch_concat_list = []

        window = Window(col_off=0, row_off=self.available_rows*idx, width=self.w,height=patch_concat.shape[1])
        new_transform = rasterio.windows.transform(window, src.transform)

        profile_new = self.profile.copy()
        profile_new['height'] = patch_concat.shape[1]
        profile_new['transform'] = new_transform
        transform = profile_new['transform']
        return patch_concat,profile_new


    def get_available_rows(self,mem_allocate,file_num,image_height,image_width,band_num=1,dtype=np.float32):
        # mem_allocate: GiB
        if mem_allocate > 512:
            Warning(f'Are you sure to allocate {mem_allocate}GiB memory?')
            print(f'Are you sure to allocate {mem_allocate}GiB memory?')
            pause()
        mem_allocate = self.GiByte_to_Byte(mem_allocate)

        memory_info = psutil.virtual_memory()
        total_mem  = self.sizeof_fmt(memory_info.total)
        sys_available_mem = memory_info.available
        if mem_allocate * 2 > memory_info.total:
            print('Memory not enough!!!','\ntotal mem:',total_mem,'\navailable mem:',self.sizeof_fmt(sys_available_mem))
            exit()
        if mem_allocate * 2 > memory_info.available:
            print('Memory Stress!!!','\ntotal mem:',total_mem,'\navailable mem:',self.sizeof_fmt(sys_available_mem))
            pause()
        array_init = np.zeros((1, 1), dtype=dtype)
        obj_mem = sys.getsizeof(array_init) - 128
        available_rows = int(mem_allocate / obj_mem / file_num / image_width)
        # print(mem_allocate / obj_mem / file_num / image_width)
        if available_rows < 1:
            raise Exception('memory not enough, please allocate more memory')
        if available_rows > image_height:
            print(f'Do not need that much memory, available_rows:{available_rows}, image_height:{image_height}')
            available_rows = image_height
        return available_rows

    def sizeof_fmt(self, num, suffix="B"):
        for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f}Yi{suffix}"

    def GiByte_to_Byte(self, GiByte):
        try:
            GiByte = float(GiByte)
        except:
            raise Exception('input error')
        return int(GiByte * 1024 * 1024 * 1024)

    def get_image_profiles(self,fpath):
        with rasterio.open(fpath) as src:
            profile = src.profile
            return profile

    def transform_to_block(self,outdir,njob=8):
        Tools().mkdir(outdir)
        flist = self.flist
        band_name_list = []
        for fpath in flist:
            fpath_obj = Path(fpath)
            band_name = fpath_obj.name
            band_name_list.append(band_name)
        if njob == 1:
            for idx in tqdm(self.block_index_list,desc='transform to block'):
                patch_concat,profile_new = self.array_iterator_index(idx)
                outf = join(outdir,f'{self.get_digit_str(self.iter_length,idx)}.tif')
                RasterIO_Func().write_tif_multi_bands(patch_concat, outf, profile_new, band_name_list)
        else:
            params_list = []
            for idx in self.block_index_list:
                params = [outdir,idx,band_name_list]
                params_list.append(params)
            MULTIPROCESS(self.kernel_transform_to_block,params_list).run(process=njob)
        pass

    def kernel_transform_to_block(self,params):
        outdir,idx,band_name_list = params
        patch_concat, profile_new = self.array_iterator_index(idx)
        outf = join(outdir, f'{self.get_digit_str(self.iter_length, idx)}.tif')
        RasterIO_Func().write_tif_multi_bands(patch_concat, outf, profile_new, band_name_list)

    def transform_to_spatial_dict(self,outdir,njob=8):
        Tools().mkdir(outdir)
        flist = self.flist
        band_name_list = []
        for fpath in flist:
            fpath_obj = Path(fpath)
            band_name = fpath_obj.name
            band_name_list.append(band_name)
        if njob == 1:
            for idx in tqdm(self.block_index_list,desc='transform to spatial dict'):
                patch_concat,profile_new = self.array_iterator_index(idx)
                row_size = patch_concat.shape[1]
                col_size = patch_concat.shape[2]
                spatial_dict = {}
                for r in range(row_size):
                    for c in range(col_size):
                        vals = patch_concat[:,r,c]
                        spatial_dict[(r+idx*self.available_rows,c)] = vals
                outf = join(outdir,f'{self.get_digit_str(self.iter_length,idx)}.npy')
                Tools().save_npy(spatial_dict,outf)
        else:
            params_list = []
            for idx in self.block_index_list:
                params = [outdir,idx]
                params_list.append(params)
            MULTIPROCESS(self.kernel_transform_to_spatial_dict,params_list).run(process=njob)

    def kernel_transform_to_spatial_dict(self,params):
        outdir, idx = params
        patch_concat, profile_new = self.array_iterator_index(idx)
        row_size = patch_concat.shape[1]
        col_size = patch_concat.shape[2]
        spatial_dict = {}
        for r in range(row_size):
            for c in range(col_size):
                vals = patch_concat[:, r, c]
                spatial_dict[(r + idx * self.available_rows, c)] = vals
        outf = join(outdir, f'{self.get_digit_str(self.iter_length, idx)}.npy')
        Tools().save_npy(spatial_dict, outf)

    def get_digit_str(self,total_len,idx):
        digit = math.log(total_len, 10) + 1
        digit = int(digit)
        digit_str = f'{idx:0{digit}d}'
        return digit_str

    def check_tifs(self):
        failed_flist = []
        for fpath in self.flist:
            try:
                with rasterio.open(fpath) as src:
                    profile = src.profile
                    # print(profile)
            except Exception as e:
                print(f'check tif error:{fpath}')
                print(e)
                print('----')
                failed_flist.append(fpath)
        return failed_flist
        pass

    def reduce(self,method,njob=8):
        flist = self.flist
        nodata=self.profile['nodata']

        band_name_list = []
        for fpath in flist:
            fpath_obj = Path(fpath)
            band_name = fpath_obj.name
            band_name_list.append(band_name)
        if njob == 1:
            results_2darray_list = []
            for idx in tqdm(self.block_index_list, desc=f'{method.__name__}'):
                params = [idx,method,nodata]
                results_2darray_i = self.kernel_reduce(params)
                results_2darray_list.append(results_2darray_i)
        else:
            params_list = []
            for idx in self.block_index_list:
                params = [idx,method,nodata]
                params_list.append(params)
            results_2darray_list = MULTIPROCESS(self.kernel_reduce,params_list).run(process=njob, desc=f'{method.__name__}')

        results_2darray = np.concatenate(results_2darray_list,axis=0)
        profile = copy.copy(self.profile)
        profile['count'] = 1
        return results_2darray,profile


    def kernel_reduce(self,params):
        idx,method,nodata=params
        patch_concat, profile_new = self.array_iterator_index(idx)
        patch_concat_2d = method(patch_concat, axis=0, where=patch_concat != nodata)
        return patch_concat_2d
    
class Tif_loader_Height:

    def __init__(self,flist,block_height,dtype=None,nodata=None,mute=False):
        self.flist = flist
        # self.image_height = image_height
        self.profile = self.get_image_profiles(flist[0])
        if nodata==None:
            pass
        else:
            self.profile.update(nodata=nodata)
        # pprint(profile)
        self.h = self.profile['height']
        self.w = self.profile['width']
        if dtype == None:
            self.dtype = self.profile['dtype']
        else:
            self.dtype = dtype
        self.profile.update(dtype=self.dtype)
        # pprint(self.profile)
        self.available_rows = block_height
        self.iter_length = math.ceil(self.h / self.available_rows)
        self.block_index_list = list(range(math.ceil(self.h / self.available_rows)))
        if not mute:
            print('input file size:', f'h:{self.h},w:{self.w}')
            print('input file count:', len(self.flist))
            print('output block size:', f'h:{self.available_rows},w:{self.w}')
            print('output block count:', self.iter_length)
            print('------------------')
        pass

    def array_iterator(self):
        idx = 0
        for row in range(0, self.h, self.available_rows):
            patch_concat_list = []
            for fpath in self.flist:
                with rasterio.open(fpath) as src:
                    patch = src.read(
                        window=((row, row + self.available_rows),(0, self.w))
                    )
                    patch_concat_list.append(patch)
            patch_concat = np.concatenate(patch_concat_list, axis=0)
            patch_concat_list = []

            window = Window(col_off=0, row_off=self.available_rows * idx, width=self.w, height=self.available_rows)
            new_transform = rasterio.windows.transform(window, src.transform)

            profile_new = self.profile.copy()
            profile_new['height'] = self.available_rows
            profile_new['transform'] = new_transform
            transform = profile_new['transform']
            idx += 1
            yield patch_concat, profile_new

    def array_iterator_index(self,idx):

        row_list = list(range(0, self.h, self.available_rows))
        row = row_list[idx]
        patch_concat_list = []
        for fpath in self.flist:
        # for fpath in tqdm(self.flist):
            with rasterio.open(fpath) as src:
                patch = src.read(
                    window=((row, row + self.available_rows),(0, self.w))
                )
                patch_concat_list.append(patch)
        patch_concat = np.concatenate(patch_concat_list, axis=0)
        # print(patch_concat.shape)
        # exit()
        patch_concat_list = []

        window = Window(col_off=0, row_off=self.available_rows*idx, width=self.w,height=patch_concat.shape[1])
        new_transform = rasterio.windows.transform(window, src.transform)

        profile_new = self.profile.copy()
        profile_new['height'] = patch_concat.shape[1]
        profile_new['transform'] = new_transform
        transform = profile_new['transform']
        return patch_concat,profile_new

    def sizeof_fmt(self, num, suffix="B"):
        for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f}Yi{suffix}"


    def get_image_profiles(self,fpath):
        with rasterio.open(fpath) as src:
            profile = src.profile
            return profile

    def transform_to_block(self,outdir,njob=8,band_name_list=None,istqdm=False):
        Tools().mkdir(outdir,True)
        flist = self.flist
        if band_name_list == None:
            band_name_list = []
            for fpath in flist:
                fpath_obj = Path(fpath)
                band_name = fpath_obj.name
                band_name_list.append(band_name)
        else:
            assert len(band_name_list) == len(flist)
        if njob == 1:
            if istqdm:
                block_index_list_iter = tqdm(self.block_index_list,desc='transform to block')
            else:
                block_index_list_iter = self.block_index_list
            for idx in block_index_list_iter:
                patch_concat,profile_new = self.array_iterator_index(idx)
                outf = join(outdir,f'{self.get_digit_str(self.iter_length,idx)}.tif')
                outf_is_ok = False
                if isfile(outf):
                    outf_is_ok = self.check_blocls(outf,band_name_list)
                    if outf_is_ok:
                        continue
                RasterIO_Func().write_tif_multi_bands(patch_concat, outf, profile_new, band_name_list)
        else:
            params_list = []
            for idx in self.block_index_list:
                params = [outdir,idx,band_name_list]
                params_list.append(params)
            MULTIPROCESS(self.kernel_transform_to_block,params_list,istqdm=istqdm).run(process=njob)
        pass

    def kernel_transform_to_block(self,params):
        outdir,idx,band_name_list = params
        outf = join(outdir, f'{self.get_digit_str(self.iter_length, idx)}.tif')
        outf_is_ok = False
        if isfile(outf):
            outf_is_ok = self.check_blocls(outf, band_name_list)
            if outf_is_ok:
                return
        patch_concat, profile_new = self.array_iterator_index(idx)
        RasterIO_Func().write_tif_multi_bands(patch_concat, outf, profile_new, band_name_list)


    def check_blocls(self,fpath,bands_description):
        try:
            bands_description_read = RasterIO_Func().read_tif_band_names(fpath)
        except:
            print('Old file is error, will overwrite',fpath)
            return False
        band_name1 = bands_description_read[0]
        band_name_origin_1 = bands_description[0]
        try:
            if band_name1 != band_name_origin_1:
                return False
        except:
            return False
        return True

    def transform_to_spatial_dict(self,outdir,njob=8):
        Tools().mkdir(outdir)
        flist = self.flist
        band_name_list = []
        for fpath in flist:
            fpath_obj = Path(fpath)
            band_name = fpath_obj.name
            band_name_list.append(band_name)
        if njob == 1:
            for idx in tqdm(self.block_index_list,desc='transform to spatial dict'):
                patch_concat,profile_new = self.array_iterator_index(idx)
                row_size = patch_concat.shape[1]
                col_size = patch_concat.shape[2]
                spatial_dict = {}
                for r in range(row_size):
                    for c in range(col_size):
                        vals = patch_concat[:,r,c]
                        spatial_dict[(r+idx*self.available_rows,c)] = vals
                outf = join(outdir,f'{self.get_digit_str(self.iter_length,idx)}.npy')
                Tools().save_npy(spatial_dict,outf)
        else:
            params_list = []
            for idx in self.block_index_list:
                params = [outdir,idx]
                params_list.append(params)
            MULTIPROCESS(self.kernel_transform_to_spatial_dict,params_list).run(process=njob)

    def kernel_transform_to_spatial_dict(self,params):
        outdir, idx = params
        patch_concat, profile_new = self.array_iterator_index(idx)
        row_size = patch_concat.shape[1]
        col_size = patch_concat.shape[2]
        spatial_dict = {}
        for r in range(row_size):
            for c in range(col_size):
                vals = patch_concat[:, r, c]
                spatial_dict[(r + idx * self.available_rows, c)] = vals
        outf = join(outdir, f'{self.get_digit_str(self.iter_length, idx)}.npy')
        Tools().save_npy(spatial_dict, outf)

    def get_digit_str(self,total_len,idx):
        digit = math.log(total_len, 10) + 1
        digit = int(digit)
        digit_str = f'{idx:0{digit}d}'
        return digit_str

    def check_tifs(self):
        failed_flist = []
        for fpath in self.flist:
            try:
                with rasterio.open(fpath) as src:
                    profile = src.profile
                    # print(profile)
            except Exception as e:
                print(f'check tif error:{fpath}')
                print(e)
                print('----')
                failed_flist.append(fpath)
        return failed_flist

    def reduce(self,method,njob=8):
        flist = self.flist
        nodata=self.profile['nodata']

        band_name_list = []
        for fpath in flist:
            fpath_obj = Path(fpath)
            band_name = fpath_obj.name
            band_name_list.append(band_name)
        if njob == 1:
            results_2darray_list = []
            for idx in self.block_index_list:
                params = [idx,method,nodata]
                results_2darray_i = self.kernel_reduce(params)
                results_2darray_list.append(results_2darray_i)
        else:
            params_list = []
            for idx in self.block_index_list:
                params = [idx,method,nodata]
                params_list.append(params)
            results_2darray_list = MULTIPROCESS(self.kernel_reduce,params_list).run(process=njob, desc=f'{method.__name__}')

        results_2darray = np.concatenate(results_2darray_list,axis=0)
        profile = copy.copy(self.profile)
        profile['count'] = 1
        return results_2darray,profile


    def kernel_reduce(self,params):
        idx, method, nodata = params
        patch_concat, profile_new = self.array_iterator_index(idx)
        patch_concat_2d = method(patch_concat)
        return patch_concat_2d

class RasterIO_Func:

    def __init__(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        pass

    def write_tif(self, array, outf, profile):
        profile['count'] = 1
        with rasterio.open(outf, "w", **profile) as dst:
            dst.write(array, 1)

    def read_tif_band_names(self,fpath):
        with rasterio.open(fpath) as dataset:
            bands_name = dataset.descriptions
            return bands_name

    def profile_template(self):
        profile = {'blockxsize': 432,
                 'blockysize': 224,
                 'compress': 'packbits',
                 'count': 1,
                 'crs': CRS().from_wkt('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
                                     'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
                                     'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
                                     'AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'),
                 'driver': 'GTiff',
                 'dtype': 'uint16',
                 'height': 2160,
                 'interleave': 'pixel',
                 'nodata': None,
                 'tiled': True,
                 'transform': Affine(0.08333333333333333, 0.0, -180.0,
                       0.0, -0.08333333333333333, 90.0),
                 'width': 4320}

        return profile

    def cal_row_col_from_coordinates(self,x_list,y_list,fpath):
        row_list = []
        col_list = []
        with rasterio.open(fpath) as src:
            profile = src.profile
            for i in range(len(x_list)):
                x = x_list[i]
                y = y_list[i]
                originX = profile['transform'][2]
                originY = profile['transform'][5]
                pixelWidth = profile['transform'][0]
                pixelHeight = profile['transform'][4]
                col = int((x - originX) / pixelWidth)
                row = int((y - originY) / pixelHeight)
                row_list.append(row)
                col_list.append(col)
        return row_list, col_list

    def extract_value_from_tif_by_row_col(self,row_list,col_list,fpath):
        value_list = []
        with rasterio.open(fpath) as src:
            profile = src.profile
            data = src.read()
            for i in range(len(row_list)):
                col = row_list[i]
                row = col_list[i]
                value = data[:, row, col]
                value_list.append(value)
        value_list = np.array(value_list)
        return value_list

    def extract_value_from_tif_by_x_y(self,x_list,y_list,fpath):
        value_list = []
        with rasterio.open(fpath) as src:
            profile = src.profile
            xy_point_list = list(zip(x_list,y_list))
            for value in src.sample(xy_point_list):
                value_list.append(value)
            value_list = np.array(value_list)
            return value_list

    def write_tif_multi_bands(self, array_3d, outf, profile, bands_description: list = None):
        dimension = array_3d.ndim
        if dimension == 2:
            array_3d = array_3d[np.newaxis, ... ]
        profile.update(count=array_3d.shape[0])
        with rasterio.open(outf, "w", **profile) as dst:
            for i in range(array_3d.shape[0]):
                dst.write(array_3d[i], i + 1)
                if bands_description is not None:
                    dst.set_band_description(i + 1, bands_description[i])

    def read_tif(self,fpath):
        with rasterio.open(fpath) as src:
            data = src.read()
            profile = src.profile
            data = data.squeeze()
            return data,profile

    def crop_tif(self,fpath,outf,in_shp):
        with rasterio.open(fpath) as src:
            shapes = gpd.read_file(in_shp).geometry
            subset, subset_transform = mask(src, shapes, crop=True)

            profile = src.profile
            profile.update({
                "height": subset.shape[1],
                "width": subset.shape[2],
                "transform": subset_transform
            })

        with rasterio.open(outf, "w", **profile) as dst:
            dst.write(subset)

    def mosaic_arrays(self,array_list,profile_list,istqdm=True):
        datasets = []

        if istqdm:
            iter_obj = tqdm(zip(array_list, profile_list),total=len(array_list),desc='mosaic')
        else:
            iter_obj = zip(array_list, profile_list)

        for arr, prof in iter_obj:
            if arr.ndim == 2 and prof["count"] == 1:
                arr = arr[np.newaxis, :, :]

            if arr.ndim == 2:
                prof.update(count=1)
            elif arr.ndim == 3:
                prof.update(count=arr.shape[0])
            else:
                raise ValueError("Invalid array dimensions for rasterio write")
            memfile = MemoryFile()
            with memfile.open(**prof) as dataset:
                dataset.write(arr)
            datasets.append(memfile.open())
        mosaic, mosaic_transform = merge(datasets)
        out_profile = profile_list[0].copy()
        out_profile.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": mosaic_transform
        })
        return mosaic,out_profile

    def mosaic_tifs(self,flist,outf,bigtiff="NO",istqdm=True):
        array_list = []
        profile_list = []
        if istqdm:
            iter_obj = tqdm(flist,desc='read tifs')
        else:
            iter_obj = flist
        bands_description = None
        for fpath in iter_obj:
            array,profile = self.read_tif(fpath)
            array_list.append(array)
            profile_list.append(profile)
            bands_description = self.read_tif_band_names(fpath)
        mosaic,out_profile = self.mosaic_arrays(array_list,profile_list,istqdm)
        if bigtiff == "YES":
            out_profile.update(bigtiff=bigtiff)
        self.write_tif_multi_bands(mosaic, outf, out_profile,bands_description=bands_description)

        pass


    def get_tif_bounds(self,fpath):
        with rasterio.open(fpath) as src:
            profile = src.profile
            crs = profile['crs']
            crs_str = crs.to_string()
            ImageHeight = profile['height']
            ImageWidth = profile['width']
            PixelWidth = profile['transform'][0]
            PixelHeight = profile['transform'][4]

            originX = profile['transform'][2]
            originY = profile['transform'][5]
            endX = originX + ImageWidth * PixelWidth
            endY = originY + ImageHeight * PixelHeight

        ll_point = (originX, originY)
        lr_point = (endX, originY)
        ur_point = (endX, endY)
        ul_point = (originX, endY)
        return ll_point,lr_point,ur_point,ul_point


    def reproject_tif(self,fpath,outf,dst_crs,dst_crs_res=None):
        with rasterio.open(fpath) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,resolution=dst_crs_res
            )
            profile = src.profile
            profile.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height
            })
            with rasterio.open(outf, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )

    def build_pyramid(self,
            tif_path,
            levels=(2, 4, 8, 16),
            resampling="average",
            compress="lzw",
            bigtiff="No",
    ):

        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"File not found: {tif_path}")
        if bigtiff == 'YES':
            with rasterio.open(tif_path) as src:
                band_descriptions = src.descriptions
                profile = src.profile
                is_bigtiff = profile.get("bigtiff") == "YES"
                if not is_bigtiff:
                    data = src.read()
                    profile.update(
                        tiled=True,
                        compress=compress,
                        bigtiff=bigtiff,
                    )
                    with rasterio.open(tif_path, "w", **profile) as dst:
                        dst.write(data)
                        for i in range(data.shape[0]):
                            dst.set_band_description(i + 1, band_descriptions[i])

        with rasterio.open(tif_path, 'r+') as ds:
            resampling_method = getattr(Resampling, resampling)
            ds.build_overviews(levels, resampling_method)
            ds.update_tags(ns='rio_overview', resampling=resampling)

    def clip_tif_by_bounds(self,in_tif,out_tif,bounds):


        with rasterio.open(in_tif) as src:
            minx, miny, maxx, maxy = bounds
            input_bounds = box(minx, miny, maxx, maxy)
            geo_df = gpd.GeoDataFrame({'geometry': input_bounds}, index=[0],
                                      crs=src.crs)

            # Get the geometry coordinates in the format rasterio mask function expects
            # which is a list of GeoJSON-like objects
            geoms = [mapping(g) for g in geo_df.geometry.values]

            # Clip the raster
            out_image, out_transform = mask(dataset=src, shapes=geoms, crop=True)

            # Update metadata for the output file
            # src.pro
            profile = src.profile
            profile.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": src.crs
            })

            with rasterio.open(out_tif, "w", **profile) as dest:
                dest.write(out_image)

        pass

    def get_tif_bounds_(self,fpath):
        # rename to get_tif_bounds
        with rasterio.open(fpath) as src:
            profile = src.profile
        # pprint(profile)
        # exit()
        crs = profile['crs']
        originX = profile['transform'][2]
        originY = profile['transform'][5]
        pixelWidth = profile['transform'][0]
        pixelHeight = profile['transform'][4]
        # endX = originX + array.shape[1] * pixelWidth
        endX = originX + profile['width'] * pixelWidth
        endY = originY + profile['height'] * pixelHeight

        minx, miny, maxx, maxy = originX, endY, endX, originY
        return minx, miny, maxx, maxy

    def clip_tif_by_tif(self,in_tif,out_tif,clip_tif):
        bounds = self.get_tif_bounds_(clip_tif)
        self.clip_tif_by_bounds(in_tif,out_tif,bounds)

    def pick_3darray_by_bands_description(self,fpath,selected_band_description):
        tif_bands_description = self.read_tif_band_names(fpath)
        idx_list = []
        for idx,name in enumerate(tif_bands_description):
            if name in selected_band_description:
                idx_list.append(idx)
        if len(idx_list) != len(selected_band_description):
            raise ValueError('band description does not match selected band description')
        data3d, profile = self.read_tif(fpath)
        data3d_selected = data3d[idx_list,:,:]
        profile['count'] = len(selected_band_description)
        return data3d_selected, profile

    def tif_vals_iterator(self,fpath):
        data3d, _ = self.read_tif(fpath)
        dimension = data3d.shape
        if len(dimension) == 3:
            for r in range(data3d.shape[1]):
                for c in range(data3d.shape[2]):
                    vals = data3d[:, r, c]
                    yield r, c, vals
        elif len(dimension) == 2:
            for r in range(data3d.shape[0]):
                for c in range(data3d.shape[1]):
                    vals = data3d[r, c]
                    yield r, c, vals
        else:
            raise ValueError('dimension must be 2 or 3')
        pass

    def gen_void_array_like(self,fpath):
        profile = self.read_raster_profile(fpath)
        width = profile['width']
        height = profile['height']
        count = profile['count']
        nodata = profile['nodata']
        dtype = profile['dtype']
        void_array = np.ones((count, height, width),dtype=dtype) * nodata
        return void_array

    def tif_vals_iterator_total_count(self,fpath):
        profile = self.read_raster_profile(fpath)
        width = profile['width']
        height = profile['height']
        total_num = width * height
        return total_num

    def tif_vals_iterator_tqdm(self,fpath,desc=None):
        profile = self.read_raster_profile(fpath)
        width = profile['width']
        height = profile['height']
        total_num = width * height
        iterator = self.tif_vals_iterator(fpath)
        iterator_tqdm = tqdm(iterator,total=total_num,desc=desc)
        return iterator_tqdm

    def array_3d_vals_iterator(self,data3d):
        for r in range(data3d.shape[1]):
            for c in range(data3d.shape[2]):
                vals = data3d[:, r, c]
                yield r, c, vals

        pass

    def read_raster_profile(self,fpath):
        with rasterio.open(fpath) as src:
            profile = src.profile
            return profile



    def tif_to_spatial_dict(self,fpath):
        data,profile = self.read_tif(fpath)

        height = profile['height']
        width = profile['width']
        count = profile['count']
        bands_description = self.read_tif_band_names(fpath)
        profile['bands_description'] = bands_description
        dimension = len(data.shape)
        if dimension == 2:
            row_num = data.shape[0]
            col_num = data.shape[1]
            spatial_dict = {}
            for r in range(row_num):
                for c in range(col_num):
                    val = data[r, c]
                    pix = (r,c)
                    spatial_dict[pix] = val
            return spatial_dict,profile

        elif dimension == 3:
            row_num = data.shape[1]
            col_num = data.shape[2]
            spatial_dict = {}
            for r in range(row_num):
                for c in range(col_num):
                    val = data[:, r, c]
                    pix = (r, c)
                    spatial_dict[pix] = val
            return spatial_dict,profile
        else:
            raise ValueError("Invalid array dimensions for rasterio write")

    def spatial_dict_to_tif(self,*args,**kwargs):
        DIC_and_DF().spatial_dict_to_tif(*args,**kwargs)

class Block_Handler:
    '''
    Handle 3d-raster blocks
    '''

    def __init__(self,block_flist):
        self.block_flist = block_flist
        originX_most, originY_most, endX_most, endY_most, pixelsize_f = self.get_bounds()

        self.originX = originX_most
        self.originY = originY_most
        self.endX = endX_most
        self.endY = endY_most
        self.pixelsize = pixelsize_f


    def get_pix_key(self,pix_key_outdir=None):
        global_originX = self.originX
        global_originY = self.originY
        global_endX = self.endX
        global_endY = self.endY
        # for fpath in tqdm(self.block_flist):
        pix_key_dict = {}
        for fpath in self.block_flist:
            fname = Path(fpath).name
            with rasterio.open(fpath) as src:
                profile = src.profile
                ImageHeight = profile['height']
                ImageWidth = profile['width']
                PixelWidth = profile['transform'][0]
                PixelHeight = profile['transform'][4]

                originX = profile['transform'][2]
                originY = profile['transform'][5]
                endX = originX + ImageWidth * PixelWidth
                endY = originY + ImageHeight * PixelHeight

                if originX < global_originX:
                    raise ValueError('Sub-region OriginX must be greater than global originX')
                if originY > global_originY:
                    raise ValueError('Sub-region OriginY must be less than global originY')
                if endX > global_endX:
                    raise ValueError('Sub-region endX must be less than global endX')
                if endY < global_endY:
                    raise ValueError('Sub-region endY must be greater than global endY')

                row = ImageHeight
                col = ImageWidth
                offset_x = int((originX - global_originX) / PixelWidth)
                offset_y = int((global_originY - originY) / abs(PixelHeight))
                pix_key_array = np.zeros((2,row,col),dtype=np.int32)
                for r in range(row):
                    for c in range(col):
                        global_r = offset_y + r
                        global_c = offset_x + c
                        pix_key_array[0][r][c] = global_r
                        pix_key_array[1][r][c] = global_c
                pix_key_dict[fpath] = pix_key_array
            if pix_key_outdir:
                assert isdir(pix_key_outdir)
                pix_key_outdir = Path(pix_key_outdir)
                outf = pix_key_outdir / f'pix_key_{fname}'
                bands_description = ['row','col']
                profile['count'] = 2
                RasterIO_Func().write_tif_multi_bands(pix_key_array, outf, profile, bands_description)
        return pix_key_dict

    def get_bounds(self):
        originX_list = []
        originY_list = []
        endX_list = []
        endY_list = []
        crs_list = []
        for fpath in self.block_flist:
            with rasterio.open(fpath) as src:
                profile = src.profile
                crs = profile['crs']
                crs_str = crs.to_string()
                if not crs_str in crs_list:
                    crs_list.append(crs_str)
                ImageHeight = profile['height']
                ImageWidth = profile['width']
                PixelWidth = profile['transform'][0]
                PixelHeight = profile['transform'][4]

                originX = profile['transform'][2]
                originY = profile['transform'][5]
                endX = originX + ImageWidth * PixelWidth
                endY = originY + ImageHeight * PixelHeight

                originX_list.append(originX)
                originY_list.append(originY)
                endX_list.append(endX)
                endY_list.append(endY)
        if len(crs_list) != 1:
            print('Different CRS found:')
            print(crs_list)
            raise ValueError('Different CRS found!')
        originX_most = min(originX_list)
        originY_most = max(originY_list)
        endX_most = max(endX_list)
        endY_most = min(endY_list)
        return originX_most,originY_most,endX_most,endY_most,PixelWidth

    def get_global_profile(self):
        # originX_most, originY_most, endX_most, endY_most, PixelWidth = self.originX,self.originY,self.endX,self.endY,self.pixelsize
        profile_list = []
        for f in self.block_flist:
            with rasterio.open(f) as src:
                profile = src.profile
                profile_list.append(profile)
        global_profile = DIC_and_DF().merge_profiles(profile_list)

        return global_profile

    def gen_global_null_array(self,layer_num=1,dtype=np.float32,nodata=None):
        profile = self.get_global_profile()
        nodata = profile['nodata'] if nodata is None else nodata
        null_array = np.ones((profile['height'],profile['width']),dtype=dtype) * nodata

        if layer_num > 1:
            array_list = []
            for num in range(layer_num):
                array_list.append(null_array)
            null_array_3d = np.stack(array_list,axis=0)
            return null_array_3d
        else:
            return null_array

        pass

    def transform_block_to_spatial_dict(self,outdir,njobs=8):
        outdir = Path(outdir)
        Tools().mkdir(outdir,force=True)
        pix_key_dict = self.get_pix_key()
        metadata_dict = {}
        params_list = []
        for fpath in pix_key_dict:
            fname = Path(fpath).name
            bands_description = RasterIO_Func().read_tif_band_names(fpath)
            profile = rasterio.open(fpath).profile
            profile['bands_description'] = bands_description
            metadata_dict[fname+'.npy'] = profile
            params = (outdir,fname,fpath,pix_key_dict)
            params_list.append(params)

        outf_metadata = outdir / 'metadata.dict'
        outf_metadata = str(outf_metadata)
        Tools().save_dict_to_binary(metadata_dict,outf_metadata)
        MULTIPROCESS(self.kernel_transform_block_to_spatial_dict,params_list).run(process=njobs)
        pass

    def transform_spatial_dict_to_block(self,fdir_dict,outdir_block):
        fdir_dict = Path(fdir_dict)
        outdir_block = Path(outdir_block)
        Tools().mkdir(outdir_block,force=True)
        profile_dict = Tools().load_dict_from_binary(fdir_dict / 'metadata.dict.pkl')
        height_offset = 0
        for f in tqdm(profile_dict,desc='transform_spatial_dict_to_block'):
            # exit()
            profile = profile_dict[f]
            height = profile['height']
            width = profile['width']
            nodata = profile['nodata']
            dtype = profile['dtype']
            bands_description = profile['bands_description']
            # profile['count'] = len(bands_list)
            spatial_dict = Tools().load_npy(fdir_dict / f)

            null_array3d = np.ones((len(bands_description),height,width),dtype=dtype)*nodata
            for key in spatial_dict:
                vals = spatial_dict[key]
                r,c = key
                _r = r - height_offset
                null_array3d[:,_r,c] = vals
            height_offset += height
            outf = outdir_block / f.replace('.npy','')
            RasterIO_Func().write_tif_multi_bands(null_array3d, outf, profile,bands_description)

        pass

    def transform_block_to_tif_list(self,outdir,njob=8):

        # get global profile
        profile_list = []
        band_list = []
        for path in self.block_flist:
            with rasterio.open(path) as src:
                profile = src.profile
                profile_list.append(profile)
                band_list = src.descriptions
        global_profile = DIC_and_DF().merge_profiles(profile_list)
        count = global_profile['count']
        global_profile['count'] = 1
        params_list = []
        for indx in range(1,count+1):
            params = indx, band_list, global_profile, outdir
            params_list.append(params)
            # self.kernel_transform_block_to_tif_list(params)
        MULTIPROCESS(self.kernel_transform_block_to_tif_list,params_list).run(process=njob)
        pass

    def kernel_transform_block_to_tif_list(self,params):
        indx, band_list, global_profile, outdir = params

        patch_list = []
        for path in sorted(self.block_flist):
            with rasterio.open(path) as src:
                patch = src.read(
                    indexes=indx
                )
                patch_list.append(patch)
        patch_concat = np.concatenate(patch_list, axis=0)
        band_name = band_list[indx - 1]
        if band_name.endswith('.tif'):
            outf = Path(outdir) / f'{band_name}'
        else:
            outf = Path(outdir) / f'{band_name}.tif'
        RasterIO_Func().write_tif(patch_concat, outf, global_profile)


    def reduce(self,outf=None,method=np.mean,njob=8):
        global_profile = self.get_global_profile()
        global_profile['count'] = 1
        global_profile['dtype'] = np.float32
        # pprint(global_profile)
        # exit()
        params_list = []
        for f in self.block_flist:
            params = f,method
            params_list.append(params)
        if len(params_list) < njob:
            njob = len(params_list)
        result_arrays = MULTIPROCESS(self.kernel_reduce,params_list).run(process=njob)
        result_array = np.concatenate(result_arrays,axis=0)
        if outf != None:
            RasterIO_Func().write_tif(result_array,outf,global_profile)
        return result_array,global_profile

    def kernel_reduce(self,params):
        f,method = params
        data, profile = RasterIO_Func().read_tif(f)
        nodata = profile['nodata']
        data_reduce = method(data, axis=0, where=data != nodata)
        return data_reduce


    def get_digit_str(self,total_len,idx):
        digit = math.log(total_len, 10) + 1
        digit = int(digit)
        digit_str = f'{idx:0{digit}d}'
        return digit_str


    def kernel_transform_block_to_spatial_dict(self,params):
        outdir,fname,fpath,pix_key_dict = params
        outf = outdir / f'{fname}.npy'
        if isfile(outf):
            return
        data3d, profile = RasterIO_Func().read_tif(fpath)
        nodata = profile['nodata']
        # print(nodata)
        # exit()
        key_array = pix_key_dict[fpath]
        row, col = data3d.shape[1], data3d.shape[2]
        spatial_dict = {}
        for r in range(row):
            for c in range(col):
                global_r = key_array[0][r][c]
                global_c = key_array[1][r][c]
                vals = data3d[:, r, c]
                # print(vals,nodata)
                if Tools().is_all_nan(vals, nodata):
                    continue

                if Tools().is_all_nan(vals):
                    continue
                spatial_dict[(int(global_r), int(global_c))] = vals
        Tools().save_npy(spatial_dict, outf)
        pass


    def reset_block_height(self,chunk_h, out_dir, progress_bar=False,desc='reset_block_height'):
        global_profile = self.get_global_profile()
        width = global_profile['width']
        height = global_profile['height']
        global_transform = global_profile['transform']

        count = global_profile['count']
        block_count = height // chunk_h + 1

        buffer = None
        idx = 0
        bands_description = RasterIO_Func().read_tif_band_names(self.block_flist[0])
        if progress_bar:
            self.block_flist = tqdm(self.block_flist,desc=desc)
        for f in self.block_flist:
            data,profile_old = RasterIO_Func().read_tif(f)

            if buffer is None:
                buffer = data
            else:
                buffer = np.concatenate([buffer, data], axis=1)

            while buffer.shape[1] >= chunk_h:
                chunk = buffer[:, :chunk_h, :]
                fname = self.get_digit_str(block_count, idx)
                out_path = os.path.join(out_dir, f"{fname}.tif")


                window = Window(col_off=0, row_off=chunk_h * idx, width=width, height=chunk_h)
                new_transform = windows.transform(window, global_transform)

                profile_new = global_profile.copy()
                profile_new['height'] = chunk_h
                profile_new['transform'] = new_transform
                if not isfile(out_path):
                    RasterIO_Func().write_tif_multi_bands(chunk, out_path, profile_new, bands_description)

                idx += 1
                buffer = buffer[:, chunk_h:, :]

        if buffer is not None and buffer.shape[1] > 0:
            fname = self.get_digit_str(block_count, idx)
            out_path = os.path.join(out_dir, f"{fname}.tif")

            window = Window(col_off=0, row_off=chunk_h * idx, width=width, height=buffer.shape[1])
            new_transform = windows.transform(window, global_transform)

            profile_new = global_profile.copy()
            profile_new['height'] = buffer.shape[1]
            profile_new['transform'] = new_transform
            if not isfile(out_path):
                RasterIO_Func().write_tif_multi_bands(buffer, out_path, profile_new, bands_description)


class DIC_and_DF:

    def __init__(self):
        pass

    def load_spatial_dict_dir(self,fdir, progress_bar=False, desc='loading dict', **kwargs):
        fdir = Path(fdir)
        metadata_dict = Tools().load_dict_from_binary(fdir / 'metadata.dict.pkl')
        fname_list = list(metadata_dict.keys())
        fname_list.sort()
        if not progress_bar:
            for fname in fname_list:
                fpath = fdir / fname
                # spatial_dict = Tools().load_npy(fpath)
                spatial_dict_f = fpath
                profile = metadata_dict[fname]
                yield spatial_dict_f, profile, fname
        else:
            for fname in tqdm(fname_list, desc=desc, **kwargs):
                fpath = fdir / fname
                # spatial_dict = Tools().load_npy(fpath)
                spatial_dict_f = fpath
                profile = metadata_dict[fname]
                yield spatial_dict_f, profile, fname

    def load_spatial_dataframe_dir(self,fdir, progress_bar=False, desc='loading df', **kwargs):
        fdir = Path(fdir)
        metadata_dict = Tools().load_dict_from_binary(fdir / 'metadata.dict.pkl')
        fname_list = list(metadata_dict.keys())
        fname_list.sort()
        if not progress_bar:
            for fname in fname_list:
                fpath = fdir / fname
                df = Tools().load_df(fpath)
                profile = metadata_dict[fname]
                yield df, profile, fname
        else:
            for fname in tqdm(fname_list, desc=desc, **kwargs):
                fpath = fdir / fname
                df = Tools().load_df(fpath)
                profile = metadata_dict[fname]
                yield df, profile, fname

    def load_spatial_dataframe_dir_profile(self,fdir):
        fdir = Path(fdir)
        profile_dict = Tools().load_dict_from_binary(fdir / 'metadata.dict.pkl')

        profile_list = []
        for profile in profile_dict:
            profile_list.append(profile_dict[profile])

        profile_merge = self.merge_profiles(profile_list)

        return profile_merge

    def merge_profiles(self,profile_list, count=None, dtype=None, nodata=None):

        originX_list = []
        originY_list = []
        endX_list = []
        endY_list = []
        PixelWidth = 0
        PixelHeight = 0
        profile = {}
        for profile in profile_list:
            # pprint(profile)
            # exit()
            ImageHeight = profile['height']
            ImageWidth = profile['width']
            PixelWidth = profile['transform'][0]
            PixelHeight = profile['transform'][4]

            originX = profile['transform'][2]
            originY = profile['transform'][5]
            endX = originX + ImageWidth * PixelWidth
            endY = originY + ImageHeight * PixelHeight

            originX_list.append(originX)
            originY_list.append(originY)
            endX_list.append(endX)
            endY_list.append(endY)
        # if 'bands_description' in profile:
        #     del profile['bands_description']
        # pprint(profile)
        originX_most = min(originX_list)
        originY_most = max(originY_list)
        endX_most = max(endX_list)
        endY_most = min(endY_list)
        ImageHeight_all = int((endY_most - originY_most) / PixelHeight)
        PixelWidth_all = int((endX_most - originX_most) / PixelWidth)

        profile['height'] = ImageHeight_all
        profile['width'] = PixelWidth_all
        if nodata != None:
            profile['nodata'] = nodata
        if count != None:
            profile['count'] = count
        if dtype != None:
            profile['dtype'] = dtype
        transform = Affine(PixelWidth, 0, originX_most, 0, PixelHeight, originY_most)

        profile['transform'] = transform
        # print('---')
        return profile

    def spatial_dataframe_dir_to_tif(self,df_loader, profile_dict, col_name, outf, method, njob: int = 7):
        # df_loader = load_spatial_dataframe(dataframe_dir,progress_bar=progress_bar)
        # profile_dict = load_spatial_dataframe_profile(dataframe_dir)

        profile_list = []
        for profile in profile_dict:
            profile_list.append(profile_dict[profile])

        profile_merge = DIC_and_DF().merge_profiles(profile_list)

        if njob == 1:
            result_array_list = []
            for df, profile, fname in df_loader:
                params = (df, col_name, method, profile, profile_merge)
                result_array = self.kernel_spatial_dataframe_dir_to_tif(params)
                result_array_list.append(result_array)
            result_array = np.concatenate(result_array_list, axis=0)
            RasterIO_Func().write_tif(result_array, outf, profile_merge)

        else:
            params_list = []
            for df, profile, fname in df_loader:
                params = (df, col_name, method, profile, profile_merge)
                params_list.append(params)
            result_array_list = MULTIPROCESS(self.kernel_spatial_dataframe_dir_to_tif, params_list).run(process=njob)
            result_array = np.concatenate(result_array_list, axis=0)
            RasterIO_Func().write_tif(result_array, outf, profile_merge)

    def spatial_dataframe_dir_to_arr(self,dataframe_dir, col_name, method, progress_bar=True, njob: int = 7):
        df_loader = self.load_spatial_dataframe_dir(dataframe_dir, progress_bar=progress_bar)
        profile_merge = self.load_spatial_dataframe_dir_profile(dataframe_dir)
        # exit()
        if njob == 1:
            result_array_list = []
            for df, profile, fname in df_loader:
                params = (df, col_name, method, profile, profile_merge)
                result_array = self.kernel_spatial_dataframe_dir_to_tif(params)
                result_array_list.append(result_array)
            result_array = np.concatenate(result_array_list, axis=0)
            return result_array

        else:
            params_list = []
            for df, profile, fname in df_loader:
                params = (df, col_name, method, profile, profile_merge)
                params_list.append(params)
            result_array_list = MULTIPROCESS(self.kernel_spatial_dataframe_dir_to_tif, params_list).run(process=njob)
            result_array = np.concatenate(result_array_list, axis=0)
            return result_array

    def kernel_spatial_dataframe_dir_to_tif(self,params):
        df, col_name, method, profile, profile_merge = params
        row = profile['height']
        col = profile['width']
        origin_Y = profile['transform'][5]
        origin_Y_merge = profile_merge['transform'][5]
        pix_height = profile['transform'][4]
        row_offset = int((origin_Y_merge - origin_Y) / pix_height)
        result_array = np.ones((row, col)) * np.nan
        df_group_dict = Tools().df_groupby(df, 'pix')
        for pix in df_group_dict:
            r, c = pix
            r_new = r + row_offset
            df_i = df_group_dict[pix]
            vals = df_i[col_name].values
            if len(vals) == 0:
                continue
            if Tools().is_all_nan(vals):
                continue
            vals_mean = method(vals)
            result_array[r_new, c] = vals_mean
        return result_array

    def add_tif_to_df(self,dff_dir, tif_fpath, col_name, njob=8, progress_bar=True):
        df_loader = self.load_spatial_dataframe_dir(dff_dir, progress_bar=progress_bar)
        data2d, data_profile = RasterIO_Func().read_tif(tif_fpath)
        val_nodata = data_profile['nodata']

        if njob == 1:
            for df, profile, fname in df_loader:
                params = df, data2d, val_nodata, col_name, dff_dir, fname
                self.kernel_add_tif_to_df(params)
        else:
            params_list = []
            for df, profile, fname in df_loader:
                params = df, data2d, val_nodata, col_name, dff_dir, fname
                params_list.append(params)
            if len(params_list) < njob:
                njob = len(params_list)
            MULTIPROCESS(self.kernel_add_tif_to_df, params_list).run(process=njob)

    def kernel_add_tif_to_df(self,params):
        df, data2d, val_nodata, col_name, dff_dir, fname = params
        dff_dir = Path(dff_dir)
        outf = dff_dir / fname
        # val_list = []
        pix_list = df['pix'].values
        rows, cols = zip(*pix_list)
        val_list = data2d[rows, cols]
        val_list = np.array(val_list)
        val_list[val_list == val_nodata] = np.nan
        df[col_name] = val_list
        Tools().save_df(df, outf)
        Tools().df_to_excel(df, outf)

    def spatial_dataframe_to_tif(self,df, profile, profile_merge, outpath, col_name, method):
        row = profile['height']
        col = profile['width']
        origin_Y = profile['transform'][5]
        origin_Y_merge = profile_merge['transform'][5]
        pix_height = profile['transform'][4]
        row_offset = int((origin_Y_merge - origin_Y) / pix_height)
        profile['count'] = 1
        result_array = np.ones((row, col)) * profile['nodata']
        df_group_dict = Tools().df_groupby(df, 'pix')
        for pix in df_group_dict:
            r, c = pix
            r_new = r + row_offset
            df_i = df_group_dict[pix]
            vals = df_i[col_name].values
            if len(vals) == 0:
                continue
            if Tools().is_all_nan(vals):
                continue
            vals_mean = method(vals)
            result_array[r_new, c] = vals_mean
        RasterIO_Func().write_tif(result_array, outpath, profile)

    def spatial_dataframe_to_arr(self,df, profile, profile_merge, col_name, method):
        row = profile['height']
        col = profile['width']
        origin_Y = profile['transform'][5]
        origin_Y_merge = profile_merge['transform'][5]
        pix_height = profile['transform'][4]
        row_offset = int((origin_Y_merge - origin_Y) / pix_height)
        profile['count'] = 1
        nodata = profile['nodata']
        result_array = np.ones((row, col)) * nodata
        df_group_dict = Tools().df_groupby(df, 'pix')
        for pix in df_group_dict:
            r, c = pix
            r_new = r + row_offset
            df_i = df_group_dict[pix]
            vals = df_i[col_name].values
            if len(vals) == 0:
                continue
            if Tools().is_all_nan(vals):
                continue
            vals = vals[vals != nodata]
            vals_mean = method(vals)
            result_array[r_new, c] = vals_mean
        return result_array
    
    def copy_metadata_dict(self,fdir,outdir):
        fpath = join(fdir,'metadata.dict.pkl')
        if not isfile(fpath):
            raise FileNotFoundError(f'{fpath} not exist')
        outpath = join(outdir,'metadata.dict.pkl')
        shutil.copyfile(fpath,outpath)
        
    def spatial_dict_dir_to_block(self,fdir, outdir, njobs=8):

        spatial_dict_loader = self.load_spatial_dict_dir(fdir)
        profile_list = []
        for spatial_dict_f, profile, fname in spatial_dict_loader:
            profile_list.append(profile)
        global_profile = self.merge_profiles(profile_list)
        # pprint(global_profile)
        # exit()
        global_PixelWidth = global_profile['transform'][0]
        global_PixelHeight = global_profile['transform'][4]

        global_originX = global_profile['transform'][2]
        global_originY = global_profile['transform'][5]

        spatial_dict_loader = self.load_spatial_dict_dir(fdir)
        params_list = []
        for spatial_dict_f, profile, fname in spatial_dict_loader:
            outf = join(outdir, fname)
            if not fname.endswith('.tif'):
                outf = outf + '.tif'

            height = profile['height']
            width = profile['width']
            nodata = profile['nodata']
            dtype = profile['dtype']
            count = profile['count']
            bands_description = profile['bands_description']
            bands_description = list(bands_description)
            # print(bands_description)
            # exit()
            if isfile(outf):
                continue
            params = (count, height, width, dtype, nodata, spatial_dict_f, profile, global_originX, global_originY,
                        outf, bands_description)
            params_list.append(params)
        MULTIPROCESS(self.kernel_spatial_dict_dir_to_block,params_list).run(njobs)
        
    def kernel_spatial_dict_dir_to_block(self,params):
        (count, height, width, dtype, nodata, spatial_dict_f, profile, global_originX, global_originY,
         outf, bands_description) = params
        null_array3d = np.ones((count, height, width), dtype=dtype) * nodata
        spatial_dict = Tools().load_npy(spatial_dict_f)

        PixelWidth = profile['transform'][0]
        PixelHeight = profile['transform'][4]

        originX = profile['transform'][2]
        originY = profile['transform'][5]

        offset_x = int((originX - global_originX) / PixelWidth)
        offset_y = int((global_originY - originY) / abs(PixelHeight))

        for key in spatial_dict:
            vals = spatial_dict[key]
            r, c = key
            _r = r - offset_y
            null_array3d[:, _r, c] = vals
        RasterIO_Func().write_tif_multi_bands(null_array3d, outf, profile, bands_description)
        
    def spatial_dict_to_tif(self,spatial_dict,profile,outf,bands_description=None,nodata=np.nan):
        dimension = -99
        dtype = np.float32
        vals_init = [np.nan]
        for pix in spatial_dict:
            vals_init = spatial_dict[pix]
            r, c = pix
            dimension = len(np.shape(vals_init))
            break


        height = profile['height']
        width = profile['width']
        profile['nodata'] = nodata
        profile['dtype'] = dtype

        if len(spatial_dict) == 0:
            void_array = np.ones((height, width), dtype=dtype) * nodata
            RasterIO_Func().write_tif(void_array,outf,profile)
            return

        if dimension == 0:
            void_array = np.ones((height, width), dtype=dtype) * nodata
        elif dimension == 1:
            count = len(vals_init)
            void_array = np.ones((count, height, width), dtype=dtype) * nodata
        else:
            raise ValueError(f'dimension {dimension} in spatial dict must be a number or a list of number')

        for pix in spatial_dict:
            vals = spatial_dict[pix]
            r, c = pix
            if dimension == 0:
                void_array[r, c] = vals
            elif dimension == 1:
                void_array[:, r, c] = vals
            else:
                raise ValueError('dimension in spatial dict must be a number or a list of number')
        if dimension == 0:
            RasterIO_Func().write_tif(void_array,outf,profile)
        elif dimension == 1:
            count = len(vals_init)
            profile['count'] = count
            RasterIO_Func().write_tif_multi_bands(void_array, outf, profile, bands_description)
            
    def df_to_spatial_dic(self, df, col_name, reduce_method=None):
        return Tools().df_to_spatial_dic(df, col_name, reduce_method=reduce_method)
    
    
    
def sleep(t=1):
    time.sleep(t)


def pause():
    # ANSI colors: https://gist.github.com/rene-d/9e584a7dd2935d0f461904b9f2950007
    input('\33[7m' + "PRESS ENTER TO CONTINUE." + '\33[0m')

def join(*args):
    args_new = []
    for path in args:
        path_new = path.replace('\\','/')
        args_new.append(path_new)
    return os.path.join(*args_new)


def run_ly_tools():
    raise UserWarning('Do not run this script')


if __name__ == '__main__':
    run_ly_tools()