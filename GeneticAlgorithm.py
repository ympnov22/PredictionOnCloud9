import os
import random
import numpy as np

GENOM_NUM = 30
GENOM_SIZE =2 
SELECT_NUM = 25
MEAN_NUM = 10

np_profit  = []

for i in range(GENOM_NUM):
    np_temp = np.loadtxt("./G{}/profit.csv".format(i),delimiter=",")
    np_profit = np.concatenate((np_profit,np_temp),axis = 0)
    #print(np_temp)
#print(np_profit)

np_profit_reshape = np.reshape(np_profit,(GENOM_NUM,-1))
#print(np_profit_reshape)
np_profit_reshape_cut = np_profit_reshape[:,-MEAN_NUM:]
#print(np_profit_reshape_cut)
np_profit_mean = np_profit_reshape_cut.mean(axis = 1)
#print(np_profit_mean)
sorted_index = np.argsort(np_profit_mean)[::-1]
#print(sorted_index)
selected = sorted_index[:SELECT_NUM]
#print(selected)

for i in range(GENOM_NUM):
    genom_1_index = np.random.randint(0,SELECT_NUM)
    genom_2_index = np.random.randint(0,SELECT_NUM)
    #print(genom_1_index,genom_2_index)
    #print(selected[genom_1_index],selected[genom_2_index])
    np_genom_1 = np.loadtxt("./G{}/genom.csv".format(selected[genom_1_index]),delimiter=",")
    np_genom_2 = np.loadtxt("./G{}/genom.csv".format(selected[genom_2_index]),delimiter=",")
    #print(np_genom_1)
    #print(np_genom_2)
    
    np_genom_next_generation = np.zeros(GENOM_SIZE)
    for j in range(GENOM_SIZE):
        if(np.random.randint(0,2) == 0):
            np_genom_next_generation[j] = np_genom_1[j]
        else:
            np_genom_next_generation[j] = np_genom_2[j]
    print(np_genom_next_generation)
    np.savetxt("G{}/genom.csv".format(i), np_genom_next_generation, delimiter=",")