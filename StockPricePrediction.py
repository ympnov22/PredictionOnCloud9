import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
import tensorflow as tf
import datetime

import sys

def LoadData(tarm):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=tarm)
    
    #print(start)
    #print(end)
    
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
    #pd_data.to_csv('StockDataRaw.csv')
    
    #print(pd_data['JPY'][['Open','Close']])
    print(pd_data.tail(10))
    return pd_data
    
def MakePredictionData_x(pd_data):
    pd_data = pd.read_csv('StockDataRaw.csv' ,index_col = 0, header = 1)
    print(pd_data)
    
    pd_data_diff = pd_data.diff(periods=1)
    #print(pd_data_diff)
    pd_data_diff_dn = pd_data_diff.dropna()
    #print(pd_data_diff_dn)
    pd_data_diff_dn_norm = pd_data_diff_dn.apply(lambda x: (x/x.std()), axis=0).fillna(0)
    #print(pd_data_diff_dn_norm)
    np_data_x = pd_data_diff_dn_norm.values[:,:]
    #print(np_data_x)
    
    #np.savetxt("PredictionData_x.csv", np_data_x, delimiter=",")
    
    return np_data_x


def Prediction(input_num,hidden_1_num,hidden_2_num,output_num,np_data_x):
    INPUT = input_num
    HIDDEN_1 = hidden_1_num
    HIDDEN_2 = hidden_2_num
    OUTPUT = output_num

    x = tf.placeholder(tf.float32, [None, INPUT])
    w1 = tf.Variable(tf.random_normal([INPUT, HIDDEN_1]))
    b1 = tf.Variable(tf.zeros([HIDDEN_1]))
    w2 = tf.Variable(tf.random_normal([HIDDEN_1, HIDDEN_2]))
    b2 = tf.Variable(tf.zeros([HIDDEN_2]))
    wy = tf.Variable(tf.random_normal([HIDDEN_2, OUTPUT]))
    by = tf.Variable(tf.zeros([OUTPUT]))
    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    y = tf.matmul(h2, wy) + by

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state('./')

    if(ckpt):
        last_model = ckpt.model_checkpoint_path
        print("load " + last_model)
        saver.restore(sess, last_model)

    else: 
        print("no variables")
        exit()

    print("predicting...")  
    result_y = sess.run(y, feed_dict={x: np_data_x})
    #print(result_y[-1])

    np.savetxt("PredictionResult.csv", result_y, delimiter=",")
    
    return result_y

args = sys.argv
pd_load_data = pd.DataFrame(index=[1,1], columns=[1,1])
#pd_load_data = LoadData(10000)
#print(pd_load_data)
np_data_x = MakePredictionData_x(pd_load_data)
#print(np_data_x.shape)
#Result_y = Prediction(36,int(args[1]),int(args[2]),2,np_data_x)
Result_y = Prediction(36,200,200,2,np_data_x)

print(Result_y[-1])