import pandas as pd
import pandas_datareader.data as pdr
import datetime
import numpy as np

import tensorflow as tf

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
    
    #print(pd_data)
    
    return pd_data
    
def MakeTrainingData_x(pd_data):
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
    
    np.savetxt("TrainingData_x.csv", np_data_x, delimiter=",")
    
    return np_data_x

def MakeTrainingData_y(pd_data):
    #print(pd_data)
    pd_data_diff = pd_data.diff(periods=1)
    #print(pd_data_diff)
    pd_data_diff_dn = pd_data_diff.dropna()
    #print(pd_data_diff_dn)
    pd_data_diff_dn_norm = pd_data_diff_dn.apply(lambda x: (x/x.std()), axis=0).fillna(0)
    #print(pd_data_diff_dn_norm)
    pd_data_diff_dn_norm_con = pd.concat([pd_data_diff_dn_norm.ix[:,36],pd_data_diff_dn_norm.ix[:,36]] , axis=1)
    #print(pd_data_diff_dn_norm_con)
    np_data_y = pd_data_diff_dn_norm_con.values[2:,:]
    
    np.savetxt("TrainingData_y.csv", np_data_y, delimiter=",")
    
    return np_data_y

def Training(input_num,hidden_1_num,hidden_2_num,output_num,test_size,np_data_x,np_data_y,training_times):
    INPUT = input_num
    #print(INPUT)
    HIDDEN_1 = hidden_1_num
    #print(HIDDEN_1)
    HIDDEN_2 = hidden_2_num
    #print(HIDDEN_2)
    OUTPUT = output_num
    #print(OUTPUT)
    TEST_SIZE = test_size
    #print(TEST_SIZE)
    TRAINING_TIMES = training_times
    #print(np_data_x)
    #print(np_data_y)
    
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    
    training_loss = []
    testing_loss =[]
    acuracy = []
    profit = []
    
    for i in range(len(np_data_x)):
        if(np.random.rand() > TEST_SIZE):
            train_x.append(np_data_x[i])
            train_y.append(np_data_y[i])
        else:
            test_x.append(np_data_x[i])
            test_y.append(np_data_y[i])
            
    #print(train_x)
    #print(train_y)
    #print(test_x)
    #print(test_y)
            
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
    y_ = tf.placeholder(tf.float32, [None, OUTPUT])
    
    loss = tf.reduce_mean(tf.square(y - y_))
    optimizer = tf.train.GradientDescentOptimizer(0.0005)
    train = optimizer.minimize(loss)
    
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    sess = tf.Session()
    
    ckpt = tf.train.get_checkpoint_state('./')
    if(ckpt):
        last_model = ckpt.model_checkpoint_path
        print("load " + last_model)
        saver.restore(sess, last_model)
    else: 
        print("inti variables")
        sess.run(init)
        
    for i in range(TRAINING_TIMES):
        sess.run(train, feed_dict={x: train_x, y_: train_y})
        
        if(i%1000 == 0):
            print("training..." + "step:" + str(i))
            print(sess.run(loss, feed_dict={x: train_x, y_: train_y}))
            training_loss.append(sess.run(loss, feed_dict={x: train_x, y_: train_y}))
            
            print("testing...")
            print(sess.run(loss, feed_dict={x: test_x, y_: test_y}))
            testing_loss.append(sess.run(loss, feed_dict={x: test_x, y_: test_y}))
            
            output = sess.run(y, feed_dict={x: test_x})
        
            temp_a = 0
            temp_p = 0
        
            for j in range(len(output)):
                if(output[j][1]*test_y[j][1]>0):
                    temp_a = temp_a + 1
                    temp_p += abs(test_y[j][1])
                else:
                    temp_p -= abs(test_y[j][1])
                    
            temp_a = float(temp_a) / len(output)
                    
            print('acuracy is ')
            print(temp_a)

            print('profit is ')
            print(temp_p)

            acuracy.append(temp_a)
            profit.append(temp_p)
            
            train_x = []
            train_y = []
            test_x = []
            test_y = []
            
            for i in range(len(np_data_x)):
                if(np.random.rand() > TEST_SIZE):
                    train_x.append(np_data_x[i])
                    train_y.append(np_data_y[i])
                else:
                    test_x.append(np_data_x[i])
                    test_y.append(np_data_y[i])
    
    result_y = sess.run(y, feed_dict={x: test_x, y_: test_y})

    np.savetxt("test_x.csv", test_x, delimiter=",")
    np.savetxt("test_y.csv", test_y, delimiter=",")
    np.savetxt("test_y_.csv", result_y, delimiter=",")

    np.savetxt("training_loss.csv", training_loss, delimiter=",")
    np.savetxt("testing_loss.csv", testing_loss, delimiter=",")

    np.savetxt("acuracy.csv", acuracy, delimiter=",")
    np.savetxt("profit.csv", profit, delimiter=",")

    saver.save(sess, "./model.ckpt")
    sess.close()
    
pd_load_data = LoadData(3000)
#print(pd_data)
np_data_x = MakeTrainingData_x(pd_load_data)
#print(np_data_x)
np_data_y = MakeTrainingData_y(pd_load_data)
#print(np_data_y)
Training(36,50,30,2,0.1,np_data_x,np_data_y,5000)
