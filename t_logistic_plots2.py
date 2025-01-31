import matplotlib.pyplot as plt
import pandas as pd
from ast import literal_eval
import numpy as np
import itertools



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

'''
Producing the plot comparing our W1 bounds for various dimensions
'''
plt.clf()
df=pd.read_csv(r't_logistic_results_zero'+str(1)+'.csv')
df.drop(columns='Unnamed: 0', inplace=True)
df1=pd.read_csv(r't_logistic_results_zero'+str(3)+'.csv')
df1.drop(columns='Unnamed: 0', inplace=True)
df2=pd.read_csv(r't_logistic_results_zero'+str(5)+'.csv')
df2.drop(columns='Unnamed: 0', inplace=True)
df3=pd.read_csv(r't_logistic_results_zero'+str(7)+'.csv')
df3.drop(columns='Unnamed: 0', inplace=True)
df4=pd.read_csv(r't_logistic_results_zero'+str(9)+'.csv')
df4.drop(columns='Unnamed: 0', inplace=True)


l=[0]*5
c=0
for j in [df,df1,df2,df3,df4]:
    r=[]
    for i in range(51):
        r.append(eval(j['1'][i])[0])
    l[c]=r
    c=c+1
plt.plot(list(df['0']),l[0], label='d=1')
plt.plot(list(df1['0']),l[1], '--', label='d=3')
plt.plot(list(df2['0']),l[2],':', label=r'd=5')
plt.plot(list(df3['0']),l[3],'-', label=r'd=7')
plt.plot(list(df4['0']),l[4],'.', label=r'd=9')
# plt.plot(list(df['0']),inverses)
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':16})
plt.savefig('w1_logistic_zero_dimensions.png',bbox_inches = 'tight', pad_inches = 0)


'''
Producing the plot comparing our W2 bounds for various dimensions
'''
plt.clf()
df=pd.read_csv(r't_logistic_results_zero'+str(1)+'.csv')
df.drop(columns='Unnamed: 0', inplace=True)
df1=pd.read_csv(r't_logistic_results_zero'+str(3)+'.csv')
df1.drop(columns='Unnamed: 0', inplace=True)
df2=pd.read_csv(r't_logistic_results_zero'+str(5)+'.csv')
df2.drop(columns='Unnamed: 0', inplace=True)
df3=pd.read_csv(r't_logistic_results_zero'+str(7)+'.csv')
df3.drop(columns='Unnamed: 0', inplace=True)
df4=pd.read_csv(r't_logistic_results_zero'+str(9)+'.csv')
df4.drop(columns='Unnamed: 0', inplace=True)


l=[0]*5
c=0
for j in [df,df1,df2,df3,df4]:
    r=[]
    for i in range(51):
        r.append(eval(j['1'][i])[2])
    l[c]=r
    c=c+1
inverses=[]

for i in list(df['0']):
    inverses.append(400/i**(3/2))
plt.plot(list(df['0']),l[0], label='d=1')
plt.plot(list(df1['0']),l[1], '--', label='d=3')
plt.plot(list(df2['0']),l[2],':', label=r'd=5')
plt.plot(list(df3['0']),l[3],'-', label=r'd=7')
plt.plot(list(df4['0']),l[4],'.', label=r'd=9')
# plt.plot(list(df['0']),inverses)
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':16})
plt.savefig('w2_logistic_zero_dimensions.png',bbox_inches = 'tight', pad_inches = 0)