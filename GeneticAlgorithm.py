import os
import random
import numpy as np

GENOM_NUM = 10
GENOM_SIZE =2 
DATA_SIZE = 10
SELECT_NUM = 7

np_profit = []

for i in range(GENOM_NUM):
    np_temp = np.loadtxt("./G{}/profit.csv".format(i),delimiter=",")
    #print(np_temp)
    np_profit = np.concatenate((np_profit,np_temp),axis = 0)

np_profit_reshape = np.reshape(np_profit,(DATA_SIZE,GENOM_NUM))
#print(np_profit_reshape)

np_profit_reshape_mean = np_profit_reshape.mean(axis = 0)
#print(np_profit_reshape_mean)

sorted_index = np.argsort(np_profit_reshape_mean)[::-1]
#print(sorted_index)

selection = sorted_index[:SELECT_NUM]
#print(selection)

for i in range(GENOM_NUM):
    temp_1 = np.random.randint(0,SELECT_NUM)
    temp_2 = np.random.randint(0,SELECT_NUM)
    #print(selection[temp_1],selection[temp_2])
    np_temp_genom_1 = np.loadtxt("./G{}/genom.csv".format(selection[temp_1]),delimiter=",")
    np_temp_genom_2 = np.loadtxt("./G{}/genom.csv".format(selection[temp_2]),delimiter=",")
    #print(np_temp_genom_1)
    #print(np_temp_genom_2)
    
    np_genom = np.zeros(GENOM_SIZE)
    for j in range(GENOM_SIZE):
        if(np.random.randint(0,2) == 0):
            np_genom[j] = np_temp_genom_1[j]
        else:
            np_genom[j] = np_temp_genom_2[j]
    print(np_genom)
    np.savetxt("G{}/genom.csv".format(i), np_genom, delimiter=",")
