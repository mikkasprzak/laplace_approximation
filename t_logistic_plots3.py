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
Minimum number of points required to compute our bound - zero parameter
'''
df1=pd.read_csv(r'lower_bounds_logistic_zero1.csv')
df1.drop(columns='Unnamed: 0', inplace=True)
df1=list(df1['0'])
df2=pd.read_csv(r'lower_bounds_logistic_zero2.csv')
df2.drop(columns='Unnamed: 0', inplace=True)
df2=list(df2['0'])
df3=pd.read_csv(r'lower_bounds_logistic_zero3.csv')
df3.drop(columns='Unnamed: 0', inplace=True)
df3=list(df3['0'])
df4=pd.read_csv(r'lower_bounds_logistic_zero4.csv')
df4.drop(columns='Unnamed: 0', inplace=True)
df4=list(df4['0'])
df5=pd.read_csv(r'lower_bounds_logistic_zero5.csv')
df5.drop(columns='Unnamed: 0', inplace=True)
df5=list(df5['0'])
df6=pd.read_csv(r'lower_bounds_logistic_zero6.csv')
df6.drop(columns='Unnamed: 0', inplace=True)
df6=list(df6['0'])
df7=pd.read_csv(r'lower_bounds_logistic_zero7.csv')
df7.drop(columns='Unnamed: 0', inplace=True)
df7=list(df7['0'])
df8=pd.read_csv(r'lower_bounds_logistic_zero8.csv')
df8.drop(columns='Unnamed: 0', inplace=True)
df8=list(df8['0'])
df9=pd.read_csv(r'lower_bounds_logistic_zero9.csv')
df9.drop(columns='Unnamed: 0', inplace=True)
df9=list(df9['0'])

val=[0]+df1+df2+df3+df4+df5+df6+df7+df8+df9

l=range(10)
plt.plot(l, val)
plt.ylabel('number of data points')
plt.xlabel('dimension')
# plt.legend(prop={'size':16})
plt.savefig('min_values_logistic_zero.png',bbox_inches = 'tight', pad_inches = 0)