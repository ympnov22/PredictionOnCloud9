import pandas as pd
import numpy as np
import tensorflow as tf

import sys

def MakePredictionData_x():
    pd_data = pd.read_csv('StockDataRaw.csv' ,index_col = 0, header = 1)
    #print(pd_data)
    select = [0,1,2,3,6,7,8,9,12,13,14,15,18,19,20,21,24,25,26,27,30,31,32,33,36,37,38,39,42,43,44,45,48,49,50,51]
    pd_data_select = pd_data[select]
    #print(pd_data_select)
    pd_data_select_diff = pd_data_select.diff(periods=1)
    #print(pd_data_select_diff)
    pd_data_select_diff_dn = pd_data_select_diff.dropna()
    #print(pd_data_select_diff_dn)
    pd_data_select_diff_dn_norm = pd_data_select_diff_dn.apply(lambda x: (x/x.std()), axis=0).fillna(0)
    #print(pd_data_select_diff_dn_norm)
    
    np_data_x = pd_data_select_diff_dn_norm.values[-10:,:]
    
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
np_genom = np.loadtxt("genom.csv",delimiter=",")
print(int(np_genom[0]))
print(int(np_genom[1]))

np_data_x = MakePredictionData_x()
Result_y = Prediction(36,int(np_genom[0]),int(np_genom[1]),2,np_data_x)

print(Result_y)