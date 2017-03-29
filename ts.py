import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
import pylab

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

print(yearlyvar.describe())
print(yearlyvar.skew())
print(yearlyvar.kurtosis()) #Attention, il s'agit de l'excess kurtosis (-3)



#La suite des pertes annuelles (en %)
X = -(yearlyvar.T).values[0]




#GARCH
monthly= (df.resample('M').mean())
monthlyvar = np.log(1 + monthly.pct_change()).dropna()
monthlyvar.plot(title="Monthly DJIA log variation")
plt.show()
autocorrelation_plot(monthlyvar)
plt.show()
autocorrelation_plot(monthlyvar**2)
plt.show()

from arch import arch_model
garch11 = arch_model(monthlyvar*100, p=1, q=1)
res = garch11.fit()
print(res.summary())
fig = res.hedgehog_plot(type='mean')
plt.show()


index= np.asarray(monthly.index)

temp=[]
vol=[]
for i in range(1571):
    res1 = garch11.fit(first_obs=1570-i, last_obs=1582, disp='off')
    print(res1.rsquared)
    temp.append(np.asarray(res1.params))
    vol.append(np.sqrt((np.asarray(res1.forecast(horizon=12).variance)[-1,:]).sum())/100)
    print(i)

temp = pd.DataFrame(np.asarray(temp))
temp.columns = ['mu','omega','alpha','beta']
del temp['mu']
del temp['omega']
temp = temp.set_index(np.flip(index[:-12],axis=0))

temp.plot()
plt.show()

vol = pd.DataFrame(np.asarray(vol))
vol.columns = ['volatilité prédite à 12 mois']
vol.plot()
plt.show()

VaR = 1 -np.exp(-1.65*np.asarray(vol))  
VaR = pd.DataFrame(np.asarray(VaR))
VaR.columns = ['VaR à 95 %']
VaR = VaR.set_index(np.flip(index[:-12],axis=0))
VaR.plot()
plt.show()


temp2=[]
vol2=[]
for i in range(1083):
    res2 = garch11.fit(first_obs=i, last_obs=500+i, disp='off')
    print(res2.rsquared)
    temp2.append(np.asarray(res2.params))
    vol2.append(np.sqrt((np.asarray(res2.forecast(horizon=12).variance)[500+i,:]).sum())/100)
    print(i)
    
temp2 = pd.DataFrame(np.asarray(temp2))
temp2.columns = ['mu','omega','alpha','beta']
del temp2['mu']
del temp2['omega']
temp2 = temp2.set_index(index[500:])
temp2.plot()
plt.show()

VaR2 = 1 -np.exp(-1.65*np.asarray(vol2))  
VaR2 = pd.DataFrame(np.asarray(VaR2))
VaR2.columns = ['VaR roulante à 95% à un an']
VaR2 = VaR2.set_index(index[500:])
VaR2.plot()
plt.show()









#Annexe A-1: Histogramme vs best normal fit
# Fit a normal distribution to the data:
muX, stdX = norm.fit(X)

# Plot the histogram.
plt.hist(X, bins='auto', normed=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, muX, stdX)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (muX, stdX)
plt.title(title)

plt.show()



#Annexe A-2 QQplot

test = (X-X.mean())/X.std()
sm.qqplot(test, line='45')
pylab.show()

