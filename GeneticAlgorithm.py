import os
import random
import numpy as np

GENOM_NUM = 15
GENOM_SIZE =2 
SELECT_NUM = 10
MEAN_NUM = 10

def Selection():
    np_profit  = []
    np_selected = np.zeros(GENOM_NUM)

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
    np_sorted_index = np.argsort(np_profit_mean)[::-1]
    print(np_sorted_index)
    
    #np_selected = np_sorted_index[:SELECT_NUM]
    print(np_selected)
    return np_selected
    
def Crossing(Np_selected):
    np_selected = Np_selected
    np_genom_next_generation = np.zeros((GENOM_NUM,GENOM_SIZE))
    
    for i in range(GENOM_NUM):
        genom_1_index = np.random.randint(0,SELECT_NUM)
        genom_2_index = np.random.randint(0,SELECT_NUM)
        #print(genom_1_index,genom_2_index)
        #print(np_selected[genom_1_index],np_selected[genom_2_index])
        np_genom_1 = np.loadtxt("./G{}/genom.csv".format(np_selected[genom_1_index]),delimiter=",")
        np_genom_2 = np.loadtxt("./G{}/genom.csv".format(np_selected[genom_2_index]),delimiter=",")
        #print(np_genom_1)
        #print(np_genom_2)
    
        
        for j in range(GENOM_SIZE):
            if(np.random.randint(0,2) == 0):
                np_genom_next_generation[i][j] = np_genom_1[j]
            else:
                np_genom_next_generation[i][j] = np_genom_2[j]
    #print(np_genom_next_generation)
    return np_genom_next_generation

def Mutation(Np_genom_next_generation):
    np_genom_next_generation = Np_genom_next_generation
    
    for i in range(GENOM_NUM):
        for j in range(GENOM_SIZE):
            if(np.random.randint(1,100) < 5):
                np_genom_next_generation[i][j] = np.random.randint(1,100)
    print(np_genom_next_generation)
    return np_genom_next_generation

def ReweiteGenom(Np_genom_next_generation):
    np_genom_next_generation = Np_genom_next_generation
    for i in range(GENOM_NUM):
        np.savetxt("G{}/genom.csv".format(i), np_genom_next_generation[i], delimiter=",")

Np_selected = Selection()
Np_genom_next_generation = Crossing(Np_selected)
Np_genom_next_generation = Mutation(Np_genom_next_generation)
ReweiteGenom(Np_genom_next_generation)
