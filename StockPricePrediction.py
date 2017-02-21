import pandas as pd
import numpy as np
import tensorflow as tf

def MakePredictionData_x():
    pd_data = pd.read_csv('PredictionDataRaw.csv' ,index_col = 1, header = 2)
    #print(pd_data)
    pd_data_diff = pd_data.diff(periods=1)
    #print(pd_data_diff)
    pd_data_diff_dn = pd_data_diff.dropna()
    #print(pd_data_diff_dn)
    pd_data_diff_dn_norm = pd_data_diff_dn.apply(lambda x: (x/x.std()), axis=0).fillna(0)
    #print(pd_data_diff_dn_norm)
    
    np_data_x = pd_data_diff_dn_norm.values[-10:,:]
    
    np.savetxt("PredictionData_x.csv", np_data_x, delimiter=",")
    
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
    print(result_y[-1])

    np.savetxt("PredictionResult.csv", result_y, delimiter=",")

np_data_x = MakePredictionData_x()
Prediction(36,50,50,2,np_data_x)