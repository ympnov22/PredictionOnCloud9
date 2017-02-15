import pandas as pd
import numpy as np
import datetime

pd_data = pd.read_csv( 'PredictionDataRaw.csv' , index_col = 0, header = 2)

#print(pd_data)

pd_data_diff = pd_data.diff(periods=1)

pd_data_diff_dn = pd_data_diff.dropna()

#print(pd_data_diff_dn)

pd_data_diff_dn_norm = pd_data_diff_dn.apply(lambda x: (x/x.std()), axis=0).fillna(0)

#pd_data_diff_dn_norm.to_csv('PredictionData.csv')
#pd_data_diff_dn.mean().to_csv('mean.csv')
#pd_data_diff_dn.std().to_csv('std.csv')

np.savetxt("PredictionData_x.csv", pd_data_diff_dn_norm.values[-10:,:], delimiter=",")