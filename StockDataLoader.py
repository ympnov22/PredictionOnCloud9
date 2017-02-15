import pandas as pd
import pandas_datareader.data as pdr
import datetime

start = datetime.datetime(2007, 1, 1)
end = datetime.date.today()

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

print(pd_data)