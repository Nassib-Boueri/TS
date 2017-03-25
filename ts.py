import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import *

df = pd.read_csv('DJA.csv',parse_dates=[0])
df.index = df['Date']
del df['Date']
print (df)
df.plot(title = 'DJIA entre Mai 1885 et Mars 2017')
plt.show()

dflog =df['DJIA'].apply(lambda x: log(x))

dflog.plot(title = 'DJIA entre Mai 1885 et Mars 2017 (échelle logarithmique)')
plt.figure(num=None, figsize=(5, 10), dpi=80, facecolor='y', edgecolor='k')  
plt.show()
