import pandas as pd
import pandas_datareader.data as pdr
import datetime
import numpy as np

import tensorflow as tf

import sys

def LoadData(tarm):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=tarm)
    
    #print(start)
    #print(end)
    
    pd_CNY = pdr.DataReader('CNY=X', 'yahoo', start, end)
    pd_JPY = pdr.DataReader('JPY=X', 'yahoo', start, end)
    pd_GBP = pdr.DataReader('GBP=X', 'yahoo', start, end)
    pd_EUR = pdr.DataReader('EUR=X', 'yahoo', start, end)
    
    pd_SP500 = pdr.DataReader('^GSPC', 'yahoo', start, end)
    pd_SSE = pdr.DataReader('000001.SS', 'yahoo', start, end)
    pd_N225 = pdr.DataReader('^N225', 'yahoo', start, end)
    pd_GDAXI = pdr.DataReader('^GDAXI', 'yahoo', start, end)
    pd_FTSE = pdr.DataReader('^FTSE', 'yahoo', start, end)
    
    pd_data = pd.concat([pd_CNY, pd_JPY, pd_GBP, pd_EUR, pd_SP500, pd_SSE, pd_N225, pd_GDAXI, pd_FTSE], axis=1, keys = ['CNY','JPY','GBP','EUR','pd_SP500','SEE','N225','GDAXI','FTSE'])
    pd_data.to_csv('StockDataRaw.csv')
    
    print(pd_data)
    
    return pd_data
    
def MakePredictionData_x(pd_data):
    #print(pd_data)
    pd_data_diff = pd_data.diff(periods=1)
    #print(pd_data_diff)
    pd_data_diff_dn = pd_data_diff.dropna()
    #print(pd_data_diff_dn)
    pd_data_diff_dn_norm = pd_data_diff_dn.apply(lambda x: (x/x.std()), axis=0).fillna(0)
    #print(pd_data_diff_dn_norm)
    select = [0,1,2,3,6,7,8,9,12,13,14,15,18,19,20,21,24,25,26,27,30,31,32,33,36,37,38,39,42,43,44,45,48,49,50,51]
    pd_data_diff_dn_norm_selct = pd_data_diff_dn_norm[select]
    #print(pd_data_diff_dn_norm_selct)
    np_data_x = pd_data_diff_dn_norm_selct.values[:-2,:]
    #print(np_training_data_x)
    
    np.savetxt("PredictionData_x.csv", np_data_x, delimiter=",")
    
    return np_data_x

pd_load_data = LoadData(3000)
#print(pd_load_data)
#np_data_x = MakePredictionData_x(pd_load_data)
#print(np_data_x)