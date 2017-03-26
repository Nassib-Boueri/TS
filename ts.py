import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
import numpy as np
import matplotlib.pyplot as plt
from math import *

df = pd.read_csv('DJA.csv',parse_dates=[0])
df.index = df['Date']
del df['Date']

print (df)
print(df.describe())


df.plot(title = 'DJIA entre Mai 1885 et Mars 2017')
plt.show()

dflog =df['DJIA'].apply(lambda x: log(x))

dflog.plot(title = 'DJIA entre Mai 1885 et Mars 2017 (échelle logarithmique)') 
plt.show()


yearlyvar = pd.DataFrame()
yearlyvar = (df.resample('A').mean())
yearlyvar=(yearlyvar.pct_change()).dropna()
yearlyvar.plot(title = 'Mean Annual DJIA Variation')
plt.show()

autocorrelation_plot(yearlyvar)
plt.show()
