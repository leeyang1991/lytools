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


class Tools:
    '''
    小工具
    '''

    def __init__(self):
        pass

    def mk_dir(self, dir, force=False):

        if not os.path.isdir(dir):
            if force == True:
                os.makedirs(dir)
            else:
                os.mkdir(dir)

    def path_join(self,*args):
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

    def mask_999999_arr(self, arr):
        arr[arr < -9999] = np.nan

    def lonlat_to_address(self, lon, lat):
        ak = "mziulWyNDGkBdDnFxWDTvELlMSun8Obt"  # 参照自己的应用
        url = 'http://api.map.baidu.com/reverse_geocoding/v3/?ak=mziulWyNDGkBdDnFxWDTvELlMSun8Obt&output=json&coordtype=wgs84ll&location=%s,%s' % (
            lat, lon)
        content = requests.get(url).text
        dic = eval(content)
        # for key in dic['result']:
        add = dic['result']['formatted_address']
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

    def detrend_dic(self, dic):
        dic_new = {}
        for key in dic:
            vals = dic[key]
            if len(vals) == 0:
                dic_new[key] = []
                continue
            try:
                vals_new = signal.detrend(vals)
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

    def point_to_shp(self, inputlist, outSHPfn):
        '''

        :param inputlist:

        # input list format
        # [[lon,lat,val],
        #      ...,
        # [lon,lat,val]]

        :param outSHPfn:
        :return:
        '''

        if len(inputlist) > 0:
            outSHPfn = outSHPfn + '.shp'
            fieldType = ogr.OFTReal
            # Create the output shapefile
            shpDriver = ogr.GetDriverByName("ESRI Shapefile")
            if os.path.exists(outSHPfn):
                shpDriver.DeleteDataSource(outSHPfn)
            outDataSource = shpDriver.CreateDataSource(outSHPfn)
            outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPoint)
            idField1 = ogr.FieldDefn('val', fieldType)
            outLayer.CreateField(idField1)
            for i in range(len(inputlist)):
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(inputlist[i][0], inputlist[i][1])
                featureDefn = outLayer.GetLayerDefn()
                outFeature = ogr.Feature(featureDefn)
                outFeature.SetGeometry(point)
                outFeature.SetField('val', inputlist[i][2])
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
        if pause_flag == 1:
            pause()

    def remove_np_nan(self, arr, is_relplace=False):
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

    def nan_correlation(self, val1_list, val2_list):
        # pearson correlation of val1 and val2, which contain Nan

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
            r, p = stats.pearsonr(val1_list_new, val2_list_new)

        return r, p

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
        os.system('open {}'.format(fpath))

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

    def normalize(self,vals,norm_max=1.,norm_min=-1.,up_limit=None,bottom_limit=None):
        vals_max = np.nanmax(vals)
        vals_min = np.nanmin(vals)
        norm_list = []
        for v in vals:
            percentile = (v-vals_min)/(vals_max-vals_min)
            norm = percentile * (norm_max - norm_min) + norm_min
            norm_list.append(norm)
        norm_list = np.array(norm_list)
        if up_limit and bottom_limit:
            norm_list[norm_list>up_limit] = np.nan
            norm_list[norm_list<bottom_limit] = np.nan
        return norm_list


    def number_of_days_in_month(self,year=2019, month=2):
        return monthrange(year, month)[1]


    def dic_to_df(self,dic,key_col_str='__key__'):
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
        all_cols = []
        for key in dic:
            vals = dic[key]
            for col in vals:
                all_cols.append(col)
        all_cols = list(set(all_cols))
        all_cols.sort()
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
        df = pd.DataFrame(data=data, columns=columns[0],index=index)
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

    def spatial_dics_to_df(self,spatial_dic_all):
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
        df = df.dropna(how='all',subset=var_name_list)
        return df

    def add_spatial_dic_to_df(self,df,dic,key_name):
        val_list = []
        for i,row in df.iterrows():
            pix = row['pix']
            if not pix in dic:
                val = None
            else:
                val = dic[pix]
            val_list.append(val)
        df[key_name] = val_list
        return df

    def df_to_dic(self,df,key_str='__key__'):
        '''
        :param df: Dataframe
        :param key_str: Unique column name
        :return:
        '''
        columns = df.columns
        dic = {}
        for i, row in df.iterrows():
            key = row[key_str]
            dic_i = {}
            for col in columns:
                val = row[col]
                dic_i[col] = val
            dic[key] = dic_i
        return dic
        pass

    def shasum(self,fpath, isprint=True):
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

    def convert_val_to_time_series_obj(self,data,time_stamp_list,name='value'):
        index = pd.DatetimeIndex(time_stamp_list)
        time_series = pd.Series(data=data, index=index, name=name)
        return time_series

    def drop_n_std(self,vals,n=1):
        vals = np.array(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up = mean + n * std
        down = mean - n * std
        vals[vals>up] = np.nan
        vals[vals<down] = np.nan
        return vals

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
        return y[(window_len // 2 - 1):-(window_len // 2)]

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

    def hist_plot_smooth(self, arr, interpolate_window=5, **kwargs):
        weights = np.ones_like(arr) / float(len(arr))
        n1, x1, patch = plt.hist(arr, weights=weights, **kwargs)
        density1 = stats.gaussian_kde(arr)
        y1 = density1(x1)
        coe = max(n1) / max(y1)
        y1 = y1 * coe
        x1, y1 = self.smooth_interpolate(x1, y1, interpolate_window)
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

    def spatial_tif_to_lon_lat_dic(self, temp_dir):
        # outf = self.this_class_arr + '{}_pix_to_lon_lat_dic.npy'.format(prefix)
        this_class_dir = os.path.join(temp_dir, 'DIC_and_TIF')
        Tools().mk_dir(this_class_dir, force=True)
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
        Tools().mask_999999_arr(arr)
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

    def plot_back_ground_arr(self, rasterized_world_tif):
        arr = ToRaster().raster2array(rasterized_world_tif)[0]
        ndv = ToRaster().get_ndv(rasterized_world_tif)
        back_ground = []
        for i in range(len(arr)):
            temp = []
            for j in range(len(arr[0])):
                val = arr[i][j]
                if val == ndv:
                    temp.append(np.nan)
                else:
                    temp.append(1)
            back_ground.append(temp)
        back_ground = np.array(back_ground)
        plt.imshow(back_ground, 'gray', vmin=0, vmax=1.4, zorder=-1)

        # return back_ground

        pass

    def plot_back_ground_arr_north_sphere(self, rasterized_world_tif):

        arr = ToRaster().raster2array(rasterized_world_tif)[0]
        back_ground = []
        for i in range(len(arr)):
            temp = []
            for j in range(len(arr[0])):
                val = arr[i][j]
                if val < -90000:
                    temp.append(np.nan)
                else:
                    temp.append(1)
            back_ground.append(temp)
        back_ground = np.array(back_ground)
        plt.imshow(back_ground[:int(len(arr) / 2)], 'gray', vmin=0, vmax=1.4, zorder=-1)

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

    def unify_raster(self, in_tif, out_tif, ndv=-999999.):
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

    def gen_ones_background_tif(self,outtif):
        # outtif = T.path_join(this_root,'conf','ones_background.tif')
        arr = self.void_spatial_dic_ones()
        self.pix_dic_to_tif(arr,outtif)
        pass

    def gen_land_background_tif(self,shp,outtif,pix_size=0.5):
        # outtif = T.path_join(this_root,'conf','land.tif')
        # shp = T.path_join(this_root,'shp','world.shp')
        ToRaster().shp_to_raster(shp,outtif,pix_size)
        self.unify_raster(outtif,outtif)
        pass

    def pix_to_lon_lat(self,pix):
        r,c = pix
        lat = self.originY + (self.pixelHeight * r)
        lon = self.originX + (self.pixelWidth * c)
        return lon,lat

    def plot_sites_location(self,lon_list,lat_list,background_tif=None,inshp=None,out_background_tif=None,pixel_size=None,text_list=None,colorlist=None,isshow=True):
        pix_list = self.lon_lat_to_pix(lon_list,lat_list,isInt=False)
        lon_list = []
        lat_list = []
        for lon,lat in pix_list:
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
                print(out_background_tif,'available')
                self.plot_back_ground_arr(out_background_tif)
            else:
                print(out_background_tif,'generating...')
                background_tif = ToRaster().shp_to_raster(inshp,out_background_tif,pixel_size=pixel_size,ndv=255)
                ToRaster().unify_raster(background_tif,background_tif,GDT_Byte=True)
                print('done')
                self.plot_back_ground_arr(background_tif)

        if colorlist:
            plt.scatter(lat_list,lon_list,c=colorlist,cmap='jet')
            plt.colorbar()
        else:
            plt.scatter(lat_list, lon_list)
            if not text_list == None:
                for i in range(len(lon_list)):
                    plt.text(lat_list[i], lon_list[i],text_list[i])
        if isshow:
            plt.show()


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
        return a, b, r

    def plot_fit_line(self, a, b, r, X, ax=None, title='', is_label=True, is_formula=True, line_color='k', **argvs):
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
                label = 'y={:0.2f}x+{:0.2f}\nr={:0.2f}'.format(a, b, r)
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
            a, b, r = self.linefit(val1, val2)
            if is_plot_1_1_line:
                plt.plot([np.min([val1, val2]), np.max([val1, val2])], [np.min([val1, val2]), np.max([val1, val2])],
                         '--', c='k')
            self.plot_fit_line(a, b, r, val1, line_color=fit_line_c)
            # plt.legend()
            return a, b, r

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

    def data_transform(self, fdir, outdir):
        # 不可并行，内存不足
        Tools().mk_dir(outdir)
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
                        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(Tools().path_join(fdir, f))
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
            if flag % 10000 == 0:
                # print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + '/per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + '/per_pix_dic_%03d' % 0, temp_dic)

    def data_transform_with_date_list(self, fdir, outdir, date_list):
        # 不可并行，内存不足
        Tools().mk_dir(outdir)
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
            if flag % 10000 == 0:
                # print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
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
            Tools().mask_999999_arr(vals)
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

    def cal_anomaly(self, fdir, save_dir):
        # fdir = this_root + 'NDVI/per_pix/'
        # save_dir = this_root + 'NDVI/per_pix_anomaly/'
        Tools().mk_dir(save_dir)
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

    def clean_per_pix(self, fdir, outdir):
        Tools().mk_dir(outdir)
        for f in tqdm(Tools().listdir(fdir)):
            dic = Tools().load_npy(fdir + '/' + f)
            clean_dic = {}
            for pix in dic:
                val = dic[pix]
                val = np.array(val, dtype=np.float)
                val[val < -9999] = np.nan
                new_val = Tools().interp_nan(val, kind='linear')
                if len(new_val) == 1:
                    continue
                # plt.plot(val)
                # plt.show()
                clean_dic[pix] = new_val
            np.save(outdir + '/' + f, clean_dic)
        pass

    def detrend(self, fdir, outdir):
        Tools().mk_dir(outdir)
        for f in tqdm(Tools().listdir(fdir), desc='detrend...'):
            dic = Tools().load_npy(fdir + f)
            dic_detrend = Tools().detrend_dic(dic)
            outf = outdir + f
            Tools().save_npy(dic_detrend, outf)
        pass

    def compose_tif_list(self,flist,outf,less_than=-9999):
        # less_than -9999, mask as np.nan
        tif_template = flist[0]
        void_dic = DIC_and_TIF(tif_template=tif_template).void_spatial_dic()
        for f in tqdm(flist,desc='transforming...'):
            if not f.endswith('.tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
            for r in range(len(array)):
                for c in range(len(array[0])):
                    pix = (r,c)
                    val = array[r][c]
                    void_dic[pix].append(val)
        spatial_dic = {}
        for pix in tqdm(void_dic,desc='calculating mean...'):
            vals = void_dic[pix]
            vals = np.array(vals)
            vals[vals<less_than] = np.nan
            mean = np.nanmean(vals)
            spatial_dic[pix] = mean
        DIC_and_TIF(tif_template=tif_template).pix_dic_to_tif(spatial_dic,outf)

    def time_series_dic_to_tif(self,spatial_dic,tif_template,outf_list):
        for i in tqdm(range(len(outf_list))):
            outf = outf_list[i]
            spatial_dic_i = {}
            for pix in spatial_dic:
                vals = spatial_dic[pix]
                val = vals[i]
                spatial_dic_i[pix] = val
            arr = DIC_and_TIF(tif_template=tif_template).pix_dic_to_spatial_arr(spatial_dic_i)
            DIC_and_TIF(tif_template=tif_template).arr_to_tif(arr,outf)


class Plot_line:
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

    def clip_array(self,in_raster,out_raster,in_shp):
        in_array, originX, originY, pixelWidth, pixelHeight = self.raster2array(in_raster)
        input_shp = ogr.Open(in_shp)
        shp_layer = input_shp.GetLayer()
        xmin, xmax, ymin, ymax = shp_layer.GetExtent()
        in_shp_encode = in_shp.encode('utf-8')
        originX_str = str(originX)
        originY_str = str(originY)
        pixelWidth_str = str(pixelWidth)
        pixelHeight_str = str(pixelHeight)
        originX_str = originX_str.encode('utf-8')
        originY_str = originY_str.encode('utf-8')
        pixelWidth_str = pixelWidth_str.encode('utf-8')
        pixelHeight_str = pixelHeight_str.encode('utf-8')
        m1 = hashlib.md5(originX_str + originY_str+pixelWidth_str+ pixelHeight_str+ in_shp_encode)
        md5filename = m1.hexdigest() + '.tif'
        temp_dir = 'temporary_directory/'
        Tools().mk_dir(temp_dir)
        temp_out_raster = temp_dir+md5filename
        if not os.path.isfile(temp_out_raster):
            self.shp_to_raster(in_shp, temp_out_raster, pixelWidth,in_raster_template=in_raster,)
        rastered_mask_array = self.raster2array(temp_out_raster)[0]
        in_mask_arr = np.array(rastered_mask_array)
        in_mask_arr[in_mask_arr < -9999] = False
        in_mask_arr = np.array(in_mask_arr, dtype=bool)
        in_array[~in_mask_arr] = np.nan
        lon_list = [xmin, xmax]
        lat_list = [ymin, ymax]
        pix_list = DIC_and_TIF(tif_template=in_raster).lon_lat_to_pix(lon_list,lat_list)
        pix1,pix2 = pix_list
        in_array = in_array[pix2[0]:pix1[0]]
        in_array = in_array.T
        in_array = in_array[pix1[1]:pix2[1]]
        in_array = in_array.T
        longitude_start, latitude_start = xmin,ymax
        self.array2raster(out_raster,longitude_start, latitude_start, pixelWidth, pixelHeight,in_array)


    def mask_array(self,in_raster,out_raster,in_mask_raster):
        in_arr,originX, originY, pixelWidth, pixelHeight = self.raster2array(in_raster)
        in_mask_arr,originX, originY, pixelWidth, pixelHeight = self.raster2array(in_mask_raster)
        in_arr = np.array(in_arr,dtype=float)
        in_mask_arr = np.array(in_mask_arr)
        in_mask_arr[in_mask_arr<-9999]=False
        in_mask_arr = np.array(in_mask_arr,dtype=bool)
        in_arr[~in_mask_arr] = np.nan
        longitude_start, latitude_start, pixelWidth, pixelHeight = originX, originY, pixelWidth, pixelHeight
        self.array2raster(out_raster,longitude_start, latitude_start, pixelWidth, pixelHeight,in_arr)

    def resample_reproj(self, in_tif, out_tif, res, srcSRS='EPSG:4326', dstSRS='EPSG:4326'):
        dataset = gdal.Open(in_tif)
        gdal.Warp(out_tif, dataset, xRes=res, yRes=res, srcSRS=srcSRS, dstSRS=dstSRS)

    def unify_raster(self, in_tif, out_tif, ndv=-999999,GDT_Byte=False):
        '''
        Unify raster to the extend of global (-180 180 90 -90)
        '''
        if GDT_Byte:
            insert_value = 255
        else:
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
        if GDT_Byte:
            self.array2raster_GDT_Byte(newRasterfn, -180, 90, pixelWidth, pixelHeight, array_unify_left_right)
        else:
            self.array2raster(newRasterfn, -180, 90, pixelWidth, pixelHeight, array_unify_left_right, ndv=ndv)

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