import os
import random
import numpy as np

GENOM_NUM = 30
GENOM_SIZE = 2

for i in range(GENOM_NUM):
    os.mkdir("G{}".format(i))
    np_genom = np.random.randint(1,100,GENOM_SIZE)
    #print (np_genom)
    np.savetxt("G{}/genom.csv".format(i), np_genom, delimiter=",")