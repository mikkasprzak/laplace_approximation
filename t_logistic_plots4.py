import matplotlib.pyplot as plt
import pandas as pd
from ast import literal_eval
import numpy as np
import itertools


d=5 # dimension - change accordingly

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


''''
Producing the plot comparing our W1 bounds for different parameter sizes (and different condition numbers)
'''

df=pd.read_csv(r't_logistic_results_zero'+str(d)+'.csv')
df.drop(columns='Unnamed: 0', inplace=True)
df1=pd.read_csv(r't_logistic_results_medium'+str(d)+'.csv')
df1.drop(columns='Unnamed: 0', inplace=True)
df2=pd.read_csv(r't_logistic_results_large'+str(d)+'.csv')
df2.drop(columns='Unnamed: 0', inplace=True)

l=[0]*3
c=0
for j in [df,df1,df2]:
    r=[]
    for i in range(51):
        r.append(eval(j['1'][i])[0])
    l[c]=r
    c=c+1
plt.plot(list(df['0']),l[0], label='logistic parameter = (0,...,0)')
plt.plot(list(df1['0']),l[1], '--', label='logistic parameter = (0.5,...,0.5)')
plt.plot(list(df2['0']),l[2],':', label='logistic parameter = (1,...,1)')
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':16})
plt.savefig('w1_logistic_comparison_new.png',bbox_inches = 'tight', pad_inches = 0)


'''
Producing the plot comparing our W2 bounds for different parameter sizes (and different condition numbers)
'''
plt.clf()
df=pd.read_csv(r't_logistic_results_zero'+str(d)+'.csv')
df.drop(columns='Unnamed: 0', inplace=True)
df1=pd.read_csv(r't_logistic_results_medium'+str(d)+'.csv')
df1.drop(columns='Unnamed: 0', inplace=True)
df2=pd.read_csv(r't_logistic_results_large'+str(d)+'.csv')
df2.drop(columns='Unnamed: 0', inplace=True)

l=[0]*3
c=0
for j in [df,df1,df2]:
    r=[]
    for i in range(51):
        r.append(eval(j['1'][i])[2])
    l[c]=r
    c=c+1

plt.plot(list(df['0']),l[0], label='logistic parameter = (0,...,0)')
plt.plot(list(df1['0']),l[1], '--', label='logistic parameter = (0.5,...,0.5)')
plt.plot(list(df2['0']),l[2],':', label='logistic parameter = (1,...,1)')
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':16})
plt.savefig('w2_logistic_comparison_new.png',bbox_inches = 'tight', pad_inches = 0)

