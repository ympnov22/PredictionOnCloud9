import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd_test_accu = pd.read_csv('testing_accuracy.csv',header = None)
pd_test_prof = pd.read_csv('testing_profit.csv',header = None)
pd_test_loss = pd.read_csv('testing_loss.csv',header = None)

pd_train_accu = pd.read_csv('training_accuracy.csv',header = None)
pd_train_prof = pd.read_csv('training_profit.csv',header = None)
pd_train_loss = pd.read_csv('training_loss.csv',header = None)

#print(pd_test_accu)
plt.subplot(3, 1, 1)
plt.plot(pd_test_loss,label= 'test_loss')
plt.plot(pd_train_loss,label= 'tarin_loss')
plt.ylim(0,4)
plt.xlim(1,)

plt.subplot(3, 1, 2)
plt.plot(pd_test_accu,label= 'test_accu')
plt.plot(pd_train_accu,label= 'train_accu')
plt.legend()
plt.xlim(1,)

plt.subplot(3, 1, 3)
plt.plot(pd_test_prof,label= 'test_prof')
plt.plot(pd_train_prof,label= 'train_prof')
plt.xlim(1,)



plt.savefig("graph.png")
