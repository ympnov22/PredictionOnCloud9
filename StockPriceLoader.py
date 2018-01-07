import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
import tensorflow as tf
import datetime

import sys

def LoadData(tarm):
    end = datetime.date.today()
    end = end + datetime.timedelta(days=3)
    start = end - datetime.timedelta(days=tarm)

    print("start date :" + str(start))
    print("end date : " + str(end))    
    
    pd_CNY = pdr.DataReader('CNY=X', 'yahoo', start, end)
    pd_CNY = pd_CNY[['Open','High','Low','Close']]
    
    pd_JPY = pdr.DataReader('JPY=X', 'yahoo', start, end)
    pd_JPY = pd_JPY[['Open','High','Low','Close']]
    
    pd_GBP = pdr.DataReader('GBP=X', 'yahoo', start, end)
    pd_GBP = pd_GBP[['Open','High','Low','Close']]
    
    pd_EUR = pdr.DataReader('EUR=X', 'yahoo', start, end)
    pd_EUR = pd_EUR[['Open','High','Low','Close']]
    
    pd_SP500 = pdr.DataReader('^GSPC', 'yahoo', start, end)
    pd_SP500 = pd_SP500[['Open','High','Low','Close']]
    
    pd_SSE = pdr.DataReader('000001.SS', 'yahoo', start, end)
    pd_SSE = pd_SSE[['Open','High','Low','Close']]
    
    pd_N225 = pdr.DataReader('^N225', 'yahoo', start, end)
    pd_N225 = pd_N225[['Open','High','Low','Close']]
    
    pd_GDAXI = pdr.DataReader('^GDAXI', 'yahoo', start, end)
    pd_GDAXI = pd_GDAXI[['Open','High','Low','Close']]
    
    pd_FTSE = pdr.DataReader('^FTSE', 'yahoo', start, end)
    pd_FTSE = pd_FTSE[['Open','High','Low','Close']]
    
    pd_data = pd.concat([pd_CNY, pd_JPY, pd_GBP, pd_EUR, pd_SP500, pd_SSE, pd_N225, pd_GDAXI, pd_FTSE], axis=1, keys = ['CNY','JPY','GBP','EUR','SP500','SEE','N225','GDAXI','FTSE'])
    pd_data.to_csv('StockDataRaw.csv')
    
    #print(pd_data['JPY'][['Open','Close']])
    print(pd_data.tail(10))
    return pd_data

args = sys.argv
pd_load_data = LoadData(10000)
#print(pd_load_data)