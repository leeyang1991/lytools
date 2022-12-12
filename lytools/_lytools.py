# coding=utf-8
import sys

version = sys.version_info.major
assert version == 3, 'Python Version Error'

import numpy as np

from tqdm import tqdm

from scipy import stats
from scipy.stats import gaussian_kde as kde
from scipy import interpolate
from scipy import signal

import pandas as pd

import seaborn as sns

from netCDF4 import Dataset

import multiprocessing
from multiprocessing.pool import ThreadPool as TPool

import os
from os.path import *
import copyreg
import types
import math
import copy
import random
import requests
import pickle
import time
import datetime
import itertools
import subprocess

from operator import itemgetter
from itertools import groupby

from osgeo import osr
from osgeo import ogr
from osgeo import gdal

import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt

import hashlib
from calendar import monthrange

import zipfile


class Tools:
    '''
    小工具
    '''

    def __init__(self):
        pass

    def mk_dir(self, dir, force=False):
        # print('will deprecated in the future version\nuse mkdir instead')
        if not os.path.isdir(dir):
            if force == True:
                os.makedirs(dir)
            else:
                os.mkdir(dir)

    def mkdir(self, dir, force=False):

        if not os.path.isdir(dir):
            if force == True:
                os.makedirs(dir)
            else:
                os.mkdir(dir)

    def mk_class_dir(self, class_name, result_root_this_script):
        this_class_arr = join(result_root_this_script, f'arr/{class_name}/')
        this_class_tif = join(result_root_this_script, f'tif/{class_name}/')
        this_class_png = join(result_root_this_script, f'png/{class_name}/')
        self.mkdir(this_class_arr, force=True)
        self.mkdir(this_class_tif, force=True)
        self.mkdir(this_class_png, force=True)

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
        pass

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
        pass

    def load_npy(self, f):
        try:
            return dict(np.load(f, allow_pickle=True).item())
        except Exception as e:
            return dict(np.load(f, allow_pickle=True, encoding='latin1').item())
        except:
            return dict(np.load(f).item())

    def save_distributed_perpix_dic(self, dic, outdir, n=10000):
        '''
        :param dic:
        :param outdir:
        :param n: save to each file every n sample
        :return:
        '''
        flag = 0
        temp_dic = {}
        for key in tqdm(dic, 'saving...'):
            flag += 1
            arr = dic[key]
            arr = np.array(arr)
            temp_dic[key] = arr
            if flag % n == 0:
                np.save(outdir + '/per_pix_dic_%03d' % (flag / n), temp_dic)
                temp_dic = {}
        np.save(outdir + '/per_pix_dic_%03d' % 0, temp_dic)

        pass

    def load_df(self, f):
        df = pd.read_pickle(f)
        df = pd.DataFrame(df)
        return df
        pass

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
        print('\33[7m' + "getting address from BaiDu." + '\33[0m')
        ak = "mziulWyNDGkBdDnFxWDTvELlMSun8Obt"  # 参照自己的应用
        url = 'http://api.map.baidu.com/reverse_geocoding/v3/?ak=mziulWyNDGkBdDnFxWDTvELlMSun8Obt&output=json&coordtype=wgs84ll&location=%s,%s' % (
            lat, lon)
        content = requests.get(url).text
        dic = eval(content)
        add = dic['result']['formatted_address']
        if len(add) == 0:
            return 'None'
        return add

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

        pass

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
        if True in np.isnan(vals):
            return vals
        return signal.detrend(vals) + np.mean(vals)

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
                    val = arr[r,c]
                    val_list.append(val)
                val_list = np.array(val_list)
                if self.is_all_nan(val_list):
                    trend_i.append(np.nan)
                    p_i.append(np.nan)
                    continue
                a, b, R, p = self.nan_line_fit(np.arange(len(val_list)),val_list)
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
        pass

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

    def pick_max_n_index(self,vals,n):
        vals = np.array(vals)
        vals[np.isnan(vals)] = -np.inf
        argsort = np.argsort(vals)
        max_n_index = argsort[::-1][:n]
        max_n_val = self.pick_vals_from_1darray(vals,max_n_index)
        return max_n_index,max_n_val

    def point_to_shp(self, inputlist, outSHPfn):
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
            'float':ogr.OFTReal,
            'int':ogr.OFTInteger,
            'str':ogr.OFTString
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
                    raise UserWarning(f'The length of column name "{col}" is too long, length must be less than 10\nplease rename the column')
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
                spatialRef.ImportFromEPSG(4326)
                spatialRef.MorphToESRI()
                file = open(outSHPfn[:-4] + '.prj', 'w')
                file.write(spatialRef.ExportToWkt())
                file.close()

                outLayer.CreateFeature(outFeature)
                outFeature.Destroy()
            outFeature = None

    def show_df_all_columns(self):
        pd.set_option('display.max_columns', None)
        pass

    def print_head_n(self, df, n=10, pause_flag=0):
        self.show_df_all_columns()
        print(df.head(n))
        print('Dataframe length:',len(df))
        print('Dataframe columns length:',len(df.columns))
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
        plt.figure()
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

        pass

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
            a, b, r, p = K.linefit(val1_list_new,val2_list_new)

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
        for key in dic:
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
        df = pd.DataFrame()
        key_list = []
        for key in tqdm(dic):
            dic_i = dic[key]
            key_list.append(key)
            # print(dic_i)
            new_dic = {k: [] for k in dic_i}
            for k in dic_i:
                new_dic[k].append(dic_i[k])
            df_i = pd.DataFrame.from_dict(data=new_dic)
            df = df.append(df_i)
        # print(len(data[0]))
        # df = pd.DataFrame(data=data, columns=columns, index=index)
        df[key_col_str] = key_list
        columns = df.columns.tolist()
        columns.remove(key_col_str)
        columns.insert(0, key_col_str)
        df = df[columns]
        return df

    def df_to_spatial_dic(self, df, col_name):
        pix_list = df['pix']
        val_list = df[col_name]
        set_pix_list = set(pix_list)
        if len(val_list) == len(set_pix_list):
            spatial_dic = dict(zip(pix_list, val_list))
            return spatial_dic
        else:
            raise UserWarning(f'"pix" is not unique')

    def is_unique_key_in_df(self,df,unique_key):
        len_df = len(df)
        unique_key_list = self.get_df_unique_val_list(df,unique_key)
        len_unique_key = len(unique_key_list)
        if len_df == len_unique_key:
            return True
        else:
            return False

    def add_dic_to_df(self, df, dic, unique_key):
        if not self.is_unique_key_in_df(df,unique_key):
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

    def join_df_list(self,df,df_list,key):
        # key must be unique
        if len(df) == 0:
            df = df_list[0]
            df_list = df_list[1:]
        if not self.is_unique_key_in_df(df,key):
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
        pass

    def df_to_dic_non_unique_key(self,df,non_unique_colname,unique_colname):
        df_to_dict = {}
        col_name_list = self.get_df_unique_val_list(df,non_unique_colname)
        for key in tqdm(col_name_list,desc='df_to_dic_non_unique_key'):
            df_col_name = df[df[non_unique_colname]==key]
            df_year_dict = self.df_to_dic(df_col_name,unique_colname)
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

    def add_lon_lat_to_df(self,df,D=None):
        lon_list = []
        lat_list = []
        if D is None:
            D = DIC_and_TIF()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            lon,lat = D.pix_to_lon_lat(pix)
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

    def monthly_to_annual_with_datetime_obj(self,vals,date_range,grow_season:list, method='mean'):
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
        annual_dict = {y:[] for y in year_list}
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

    def month_index_to_date_obj(self,month_index,init_date_obj):
        year = init_date_obj.year
        month = init_date_obj.month
        end_month = month + month_index
        end_year = year + int(end_month/12)
        end_month = end_month%12
        if end_month == 0:
            end_month = 12
            end_year -= 1
        end_date_obj = datetime.datetime(end_year,end_month,1)
        return end_date_obj

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

    def is_all_nan(self,vals):
        if type(vals) == float:
            return True
        vals = np.array(vals)
        isnan_list = np.isnan(vals)
        isnan_list_set = set(isnan_list)
        isnan_list_set = list(isnan_list_set)
        if len(isnan_list_set) == 1:
            if isnan_list_set[0] == True:
                return True
            else:
                return False
            pass
        else:
            return False
        pass

    def reverse_dic(self,dic):
        items = dic.items()
        df = pd.DataFrame.from_dict(data=items)
        unique_vals = self.get_df_unique_val_list(df,1)
        dic_reverse = {}
        for v in unique_vals:
            dic_reverse[v] = []
        for key in dic:
            val = dic[key]
            dic_reverse[val].append(key)
        return dic_reverse

    def combine_df_columns(self,df,combine_col,new_col_name,method='mean'):
        combined_vals_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            vals_list = []
            for cols in combine_col:
                vals = row[cols]
                if type(vals) == float:
                    continue
                vals = np.array(vals)
                vals_list.append(vals)
            vals_list = np.array(vals_list)
            vals_list_mean = np.nanmean(vals_list,axis=0)
            combined_vals_list.append(vals_list_mean)
        df[new_col_name] = combined_vals_list
        return df

    def get_vals_std_up_down(self,vals):
        vals = np.array(vals)
        std = np.nanstd(vals)
        mean = np.nanmean(vals)
        up = mean + std
        down = mean - std
        return up,down

    def vals_to_time_sereis_annual(self,vals,yearlist:list=None,start_year:int=None):
        if yearlist == start_year == None:
            raise UserWarning('You must input at least one parameter of "year list" or "start year"')
        if start_year != None:
            yearlist = list(range(start_year, start_year + len(vals)))
            xval_ts = pd.Series(vals, index=yearlist)
            return xval_ts
        if yearlist != None:
            xval_ts = pd.Series(vals, index=yearlist)
            return xval_ts

    def hex_color_to_rgb(self,hex_color):
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

    def cross_list(self,*args,is_unique=False):
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

    def cross_select_dataframe(self,df,*args,is_unique=False):
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
            cross_list = self.cross_list(*unique_value_list,is_unique=is_unique)
            cross_df_dict = {}
            for x in cross_list:
                df_copy = copy.copy(df)
                for xi in range(len(x)):
                    df_copy = df_copy[df_copy[args[xi]] == x[xi]]
                cross_df_dict[x] = df_copy
            return cross_df_dict


    def resample_nan(self,array,target_res,original_res,nan_value=-999999):
        array = array.astype(np.float32)
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
        pass

    def cmap_blend(self,color_list,as_cmap=True,n_colors=6):
        # color_list = ['r', 'g', 'b']
        cmap = sns.blend_palette(color_list, as_cmap=as_cmap, n_colors=n_colors)
        return cmap

    def cmap_diverging(self,start_color_hue,end_color_hue,saturation=100,lightness=40):
        cmap = sns.diverging_palette(0, 120, s=saturation, l=40, as_cmap=True)
        return cmap

    def get_max_key_from_dict(self,input_dict):
        max_key = None
        max_value = -np.inf
        for key in input_dict:
            value = input_dict[key]
            if value > max_value:
                max_key = key
                max_value = value
        return max_key

    def days_number_of_year(self,year):
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

    def count_days_of_two_dates(self,date1,date2):
        date1 = datetime.datetime.strptime(date1,'%Y-%m-%d')
        date2 = datetime.datetime.strptime(date2,'%Y-%m-%d')
        delta = date2 - date1
        return delta.days

    def nc_to_tif(self,fname,var_name,outdir):
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
            lon_list = []
            lat_list = []
            value_list = []
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    lon_i = xx[i][j]
                    if lon_i > 180:
                        lon_i -= 360
                    lat_i = yy[i][j]
                    value_i = arr[i][j]
                    lon_list.append(lon_i)
                    lat_list.append(lat_i)
                    value_list.append(value_i)
            DIC_and_TIF().lon_lat_val_to_tif(lon_list, lat_list, value_list, outpath)

    def uncertainty_err(self,vals):
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up, bottom = stats.t.interval(0.95, len(vals) - 1, loc=mean, scale=std / np.sqrt(len(vals)))
        err = mean - bottom
        return err, up, bottom

    def df_bin(self,df,col,bins):
        df_copy = df.copy()
        df_copy[f'{col}_bins'] = pd.cut(df[col],bins=bins)
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
        return df_group,bins_list_str

    def ANOVA_test(self,*args,method:str):
        if method == 'f_oneway':
            return stats.f_oneway(*args)
        elif method == 'ks':
            if len(args) == 2:
                return stats.ks_2samp(*args)
            else:
                raise ValueError('KS test args length must be 2')

    def drop_df_index(self,df):
        df = df.reset_index(drop=True)
        return df

    def date_to_DOY(self,date_list):
        '''
        :param date_list: list of datetime objects
        :return: list of DOY
        '''
        start_year = date_list[0].year
        start_date = datetime.datetime(start_year, 1, 1)
        date_delta = date_list - start_date + datetime.timedelta(days=1)
        DOY = [date.days for date in date_delta]
        return DOY

    def gen_colors(self,color_list_number,palette='Spectral'):
        color_list = sns.color_palette(palette, color_list_number)
        return color_list

    def del_columns(self,df,columns:list):
        df = df.drop(columns=columns,axis=1)
        return df

    def bootstrap_data(self,data:pd.DataFrame,n:int,ratio:float):
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
        :param inlist:
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

        pass

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
                    a,_,_,_ = Tools().nan_line_fit(x,y)
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

    def pix_dic_to_tif_every_time_stamp(self, spatial_dic, outdir, filename_list:list=None):
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

    def spatial_tif_to_lon_lat_dic(self, temp_dir):
        # outf = self.this_class_arr + '{}_pix_to_lon_lat_dic.npy'.format(prefix)
        this_class_dir = os.path.join(temp_dir, 'DIC_and_TIF')
        Tools().mkdir(this_class_dir, force=True)
        outf = os.path.join(this_class_dir, 'spatial_tif_to_lon_lat_dic')
        if os.path.isfile(outf):
            print(f'loading {outf}')
            dic = Tools().load_npy(outf)
            print('done')
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
                    # print(tuple([lon, lat]))
            print('saving')
            np.save(outf, pix_to_lon_lat_dic)
            return pix_to_lon_lat_dic

    def spatial_tif_to_dic(self, tif):

        arr = ToRaster().raster2array(tif)[0]
        arr = np.array(arr, dtype=float)
        arr = Tools().mask_999999_arr(arr, warning=False)
        dic = self.spatial_arr_to_dic(arr)
        return dic

        pass

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

    def plot_back_ground_arr(self, rasterized_world_tif,ax=None, **kwargs):
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


    def plot_back_ground_arr_north_sphere(self, rasterized_world_tif,ax=None,**kwargs):
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
            plt.imshow(back_ground[:int(len(arr) / 2)], 'gray', vmin=0, vmax=1.4, zorder=-1,**kwargs)
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
                    temp.append(None)
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

    def unify_raster1(self, in_tif, out_tif, res, srcSRS='EPSG:4326', dstSRS='EPSG:4326'): # todo: need to be tested
        row = len(self.arr_template)
        col = len(self.arr_template[0])
        sY = self.originY
        sX = self.originX
        eY = sY + row * self.pixelHeight
        eX = sX + col * self.pixelWidth
        extent = [sX, eY, eX, sY]
        dataset = gdal.Open(in_tif)
        gdal.Warp(out_tif, dataset, srcSRS=srcSRS, dstSRS=dstSRS, outputBounds=extent)

    def resample_reproj(self, in_tif, out_tif, res, srcSRS='EPSG:4326', dstSRS='EPSG:4326'):
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
                            pixel_size=None, text_list=None, colorlist=None, isshow=False,**kwargs):
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

    def plot_df_spatial_pix(self,df,global_land_tif):
        pix_list = df['pix'].tolist()
        pix_list = list(set(pix_list))
        spatial_dict = {pix:1 for pix in pix_list}
        arr = self.pix_dic_to_spatial_arr(spatial_dict)
        self.plot_back_ground_arr(global_land_tif)
        plt.imshow(arr)


    def rad(self,d):
        return d * math.pi / 180

    def GetDistance(self,lng1, lat1, lng2, lat2):
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
        for pix in tqdm(pix_list,desc='calculate_pixel_area'):
            lon,lat = self.pix_to_lon_lat(pix)
            upper_left_lon = lon - pixel_size/2
            upper_left_lat = lat + pixel_size/2
            upper_right_lon = lon + pixel_size/2
            upper_right_lat = lat + pixel_size/2
            lower_left_lon = lon - pixel_size/2
            lower_left_lat = lat - pixel_size/2
            lower_right_lon = lon + pixel_size/2
            lower_right_lat = lat - pixel_size/2
            upper_left_to_upper_right = self.GetDistance(upper_left_lon,upper_left_lat,upper_right_lon,upper_right_lat)
            upper_left_to_lower_left = self.GetDistance(upper_left_lon,upper_left_lat,lower_left_lon,lower_left_lat)
            area = upper_left_to_upper_right * upper_left_to_lower_left
            area_dict[pix] = area
        return area_dict


class MULTIPROCESS:
    '''
    可对类内的函数进行多进程并行
    由于GIL，多线程无法跑满CPU，对于不占用CPU的计算函数可用多线程
    并行计算加入进度条
    '''

    def __init__(self, func, params):
        self.func = func
        self.params = params
        copyreg.pickle(types.MethodType, self._pickle_method)
        pass

    def _pickle_method(self, m):
        if m.__self__ is None:
            return getattr, (m.__self__.__class__, m.__func__.__name__)
        else:
            return getattr, (m.__self__, m.__func__.__name__)

    def run(self, process=4, process_or_thread='p', **kwargs):
        '''
        # 并行计算加进度条
        :param func: input a kenel_function
        :param params: para1,para2,para3... = params
        :param process: number of cpu
        :param thread_or_process: multi-thread or multi-process,'p' or 't'
        :param kwargs: tqdm kwargs
        :return:
        '''

        if process > 0:
            if process_or_thread == 'p':
                pool = multiprocessing.Pool(process)
            elif process_or_thread == 't':
                pool = TPool(process)
            else:
                raise IOError('process_or_thread key error, input keyword such as "p" or "t"')

            results = list(tqdm(pool.imap(self.func, self.params), total=len(self.params), **kwargs))
            pool.close()
            pool.join()
            return results
        else:
            if process_or_thread == 'p':
                pool = multiprocessing.Pool()
            elif process_or_thread == 't':
                pool = TPool()
            else:
                raise IOError('process_or_thread key error, input keyword such as "p" or "t"')

            results = list(tqdm(pool.imap(self.func, self.params), total=len(self.params), **kwargs))
            pool.close()
            pool.join()
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
            plt.title(title)
            plt.scatter(val1, val2, c=colors, linewidths=0, s=s, **kwargs)

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

    def plot_scatter_hex(self,x,y,kind="hex", color="#4CB391", xlim=None, ylim=None, gridsize=80):
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
        # 不可并行，内存不足
        Tools().mkdir(outdir)
        # 将空间图转换为数组
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
                        array = np.array(array, dtype=np.float)
                        # print np.min(array)
                        # print type(array)
                        # plt.imshow(array)
                        # plt.show()
                        all_array.append(array)

        row = len(all_array[0])
        col = len(all_array[0][0])

        void_dic = {}
        void_dic_list = []
        for r in range(row):
            for c in range(col):
                void_dic[(r, c)] = []
                void_dic_list.append((r, c))

        # print(len(void_dic))
        # exit()
        params = []
        for r in tqdm(list(range(row))):
            for c in range(col):
                for arr in all_array:
                    val = arr[r][c]
                    void_dic[(r, c)].append(val)

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
                np.save(outdir + '/per_pix_dic_%03d' % (flag / n), temp_dic)
                temp_dic = {}
        np.save(outdir + '/per_pix_dic_%03d' % 0, temp_dic)

    def data_transform_with_date_list(self, fdir, outdir, date_list, n=10000):
        n = int(n)
        # 不可并行，内存不足
        Tools().mkdir(outdir)
        outdir = outdir + '/'
        # 将空间图转换为数组
        template_f = os.path.join(fdir, Tools().listdir(fdir)[0])
        template_arr = ToRaster().raster2array(template_f)[0]
        void_arr = np.ones_like(template_arr) * np.nan
        all_array = []
        invalid_f_num = 0
        for d in tqdm(date_list, 'loading...'):
            f = os.path.join(fdir, d)
            if os.path.isfile(f):
                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
                array = np.array(array, dtype=np.float)
                all_array.append(array)
            else:
                all_array.append(void_arr)
                invalid_f_num += 1
        print('\n', 'invalid_f_num:', invalid_f_num)
        # exit()

        row = len(all_array[0])
        col = len(all_array[0][0])

        void_dic = {}
        void_dic_list = []
        for r in range(row):
            for c in range(col):
                void_dic[(r, c)] = []
                void_dic_list.append((r, c))

        # print(len(void_dic))
        # exit()
        params = []
        for r in tqdm(range(row)):
            for c in range(col):
                for arr in all_array:
                    val = arr[r][c]
                    void_dic[(r, c)].append(val)

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

    def climotology_mean_std(self,vals):
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
            result_dict[m] = {'mean':mean,'std':std}
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

    def cal_relative_change(self,vals):
        relative_change_list = []
        mean = np.nanmean(vals)
        for v in vals:
            relative_change = (v-mean) / v
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
                val = np.array(val, dtype=np.float)
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
            dic = Tools().load_npy(join(fdir,f))
            dic_detrend = Tools().detrend_dic(dic)
            outf = join(outdir,f)
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
            vals = np.array(vals)
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

    def plot_hist_smooth(self, arr, interpolate_window=5,range=None, **kwargs):
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

    def multi_step_pdf(self,df, pdf_colname ,bin_colname, bins, pdf_bin_n=40,pdf_range=None, discard_limit_n=10,
                       blend_color_list=('#ABD3E0','#EFDDE2','#DC9AA2')):
        flag = 0
        df_group,bins_list_str = Tools().df_bin(df,bin_colname,bins)
        color_len = 0
        for name,df_group_i in df_group:
            vals = df_group_i[pdf_colname].tolist()
            vals = np.array(vals)
            if len(vals) <= discard_limit_n:
                continue
            color_len += 1
        # exit()
        color_list = Tools().cmap_blend(blend_color_list,as_cmap=False,n_colors=color_len)
        alpha_list = np.linspace(0.5,1,color_len)
        fig, ax = plt.subplots(1)
        delta_y = 0.1
        label_list = []
        y_tick_list = []
        vals_mean_list = []
        for name,df_group_i in df_group:
            label = str(name.left) + '-' + str(name.right)
            vals = df_group_i[pdf_colname].tolist()
            vals = np.array(vals)
            print(bin_colname,label,len(vals))
            if len(vals) <= discard_limit_n:
                continue
            vals_mean = np.nanmedian(vals)
            vals_mean_list.append(vals_mean)
            x, y = Plot().plot_hist_smooth(vals, bins=pdf_bin_n, alpha=0., range=pdf_range, color=color_list[flag],
                                         linewidth=2)
            # ax.plot(x, y, label=label, color=color_list[flag])
            ax.plot(x, y+delta_y*flag, color='k',zorder=100)
            y_tick_list.append(delta_y*flag)
            ax.fill(x, y+delta_y*flag, color=color_list[flag], label=label,zorder=-flag)
            label_list.append(label)
            flag += 1
        # plt.legend()
        plt.xlabel(pdf_colname)
        plt.ylabel(bin_colname)
        plt.yticks(y_tick_list,label_list)
        y_tick_list = np.array(y_tick_list)
        plt.scatter(vals_mean_list,y_tick_list+0.05,color='k',zorder=100)
        # plt.tight_layout()
        # plt.show()


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
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
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
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        # Close Geotiff
        outband.FlushCache()
        del outRaster

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

    def resample_reproj(self, in_tif, out_tif, res, srcSRS='EPSG:4326', dstSRS='EPSG:4326'):
        dataset = gdal.Open(in_tif)
        gdal.Warp(out_tif, dataset, xRes=res, yRes=res, srcSRS=srcSRS, dstSRS=dstSRS)


class HANTS:

    def __init__(self):
        '''
        HANTS algorithm for time series smoothing
        '''
        pass

    def __left_consecutive_index(self,values_list,invalid_value):
        left_consecutive_non_valid_index = []
        for i in range(len(values_list)):
            if values_list[i] == invalid_value:
                left_consecutive_non_valid_index.append(i)
            else:
                break
        return left_consecutive_non_valid_index

    def __right_consecutive_index(self,values_list,invalid_value):
        right_consecutive_non_valid_index = []
        for i in range(len(values_list) - 1, -1, -1):
            if values_list[i] == invalid_value:
                right_consecutive_non_valid_index.append(i)
            else:
                break
        return right_consecutive_non_valid_index

    def hants_interpolate(self, values_list, dates_list, valid_range,nan_value=np.nan):
        '''
        :param values_list: 1D, list of values, multi years
        :param dates_list:  1D, list of dates, corresponding to values_list
        :param valid_range: tuple, (low,high), valid range of values
        :param nan_value: float, nan value
        :return: Dict, {year:DOY_values}
        '''
        values_list = np.array(values_list)
        values_list[np.isnan(values_list)] = valid_range[0]
        year_list = []
        for date in dates_list:
            year = date.year
            if year not in year_list:
                year_list.append(year)
        values_dict = dict(zip(dates_list, values_list))
        annual_date_dict = {}
        for date in values_dict:
            year = date.year
            if year not in annual_date_dict:
                annual_date_dict[year] = []
            annual_date_dict[year].append(date)
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
        std_i = np.nanmean(_values_list_1)
        std = 2. * std_i # larger than twice the standard deviation of the input data is rejected
        interpolated_values_list = []
        for i,values in enumerate(_values_list):
            dates = _dates_list[i]
            xnew, ynew = self.__interp_values_to_DOY(values, dates)
            # print(xnew)
            # print(ynew)
            # print('---')
            # plt.plot(xnew, ynew, 'o')
            # plt.show()
            interpolated_values_list.append(ynew)


        interpolated_values_list = np.array(interpolated_values_list)
        results = HANTS().__hants(sample_count=365, inputs=interpolated_values_list, low=valid_range[0], high=valid_range[1],
                                fit_error_tolerance=std)
        results_new = []
        for i in range(len(results)):
            results_i = results[i]
            left_consecutive_non_valid_index = self.__left_consecutive_index(interpolated_values_list[i],valid_range[0])
            # print(left_consecutive_non_valid_index)
            right_consecutive_non_valid_index = self.__right_consecutive_index(interpolated_values_list[i],valid_range[0])
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
            # plt.plot(results_i_new)
            # plt.plot(interpolated_values_list[i])
            # plt.show()
        # plt.imshow(interpolated_values_list, aspect='auto')
        # plt.colorbar()
        #
        # plt.figure()
        # plt.imshow(results_new, aspect='auto')
        # plt.colorbar()
        # plt.show()
        results_dict = dict(zip(year_list, results_new))
        return results_dict

    def __date_list_to_DOY(self,date_list):
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

    def __makediag3d(self,M):
        b = np.zeros((M.shape[0], M.shape[1] * M.shape[1]))
        b[:, ::M.shape[1] + 1] = M
        return b.reshape((M.shape[0], M.shape[1], M.shape[1]))

    def __get_starter_matrix(self,base_period_len, sample_count, frequencies_considered_count):
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

    def __hants(self,sample_count, inputs,
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
            zr = np.linalg.solve(A, za)
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

    def __init__(self,df,variable_list,start_year,end_year):
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
        return df,year_list

    def dataframe_per_value(self):
        variable_list = self.variable_list
        df_,year_list = self.init_void_dataframe()
        pix_list = Tools().get_df_unique_val_list(df_, 'pix')
        nan_list = [np.nan] * len(year_list)
        all_data = {}
        for col in variable_list:
            spatial_dict = Tools().df_to_spatial_dic(self.df_in,col)
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
                for i,v in enumerate(vals):
                    val_list_all.append(v)
            df_[var_i] = val_list_all
        df = df_.dropna(subset=variable_list,how='all')
        self.df = df

def sleep(t=1):
    time.sleep(t)


def pause():
    # ANSI colors: https://gist.github.com/rene-d/9e584a7dd2935d0f461904b9f2950007
    input('\33[7m' + "PRESS ENTER TO CONTINUE." + '\33[0m')


def run_ly_tools():
    raise UserWarning('Do not run this script')
    pass


if __name__ == '__main__':
    run_ly_tools()