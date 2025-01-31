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

'''
Producing the plot comparing our bound to the true difference of means - for large parameter and with error bands
'''

df=pd.read_csv(r't_logistic_results_large'+str(d)+'.csv')
df.drop(columns='Unnamed: 0', inplace=True)
r1=[]
r2=[]
for i in range(51):
    r1.append(eval(df['1'][i])[0])
    r2.append(eval(df['1'][i])[1])

df1=pd.read_csv(r'mean_covs_large'+str(d)+'.csv')
df1=np.array(df1)
means=[]
for i in range(51):
    means.append(df1[:,6*i])
means=np.array(means)

df2=pd.read_csv(r't_logistic_maps_large'+str(d)+'.csv')
df2.drop(columns='Unnamed: 0', inplace=True)
df2=np.array(df2)
diff=np.subtract(means, df2)

df3=pd.read_csv(r'mean_error'+str(d)+'.csv')
df3=np.array(df3)
errors=[]
for i in range(51):
    errors.append(df3[:5,i])

l=[]
u=[]
for i in range(51):
    lower_bound=[]
    upper_bound=[]
    for j in range(d):
        upper_bound.append(max(abs(diff[i][j] + errors[i][j]), abs(diff[i][j] - errors[i][j])))
        if abs(diff[i][j])<errors[i][j]/2:
            lower_bound.append(0)
        else:
            lower_bound.append(min(abs(diff[i][j]+errors[i][j]),abs(diff[i][j]-errors[i][j])))
    l.append(lower_bound)
    u.append(upper_bound)

diff=np.linalg.norm(diff, axis=1)
diff1=np.array(l)
diff1=np.linalg.norm(diff1, axis=1)
diff2=np.array(u)
diff2=np.linalg.norm(diff2, axis=1)


plt.plot(list(df['0']),r1, label='our bound on the \n 1-Wasserstein distance')
plt.plot(list(df['0']),r2, '-.', label='norm of the approximating mean')
plt.plot(list(df['0']),diff, ':',label='norm of the simulated \n true difference of means')
plt.plot(list(df['0']),diff1, '--', color='brown', label='standard error bands for the \n true difference of means')
plt.plot(list(df['0']),diff2, '--',color='brown')
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':13})
plt.savefig('w1_logistic_large.png',bbox_inches = 'tight', pad_inches = 0)


'''
Producing the plot comparing our bound to the true difference of covariances - for large parameter
'''

df=pd.read_csv(r't_logistic_results_large'+str(d)+'.csv')
df.drop(columns='Unnamed: 0', inplace=True)
r1=[]
r2=[]
for i in range(51):
    r1.append(eval(df['1'][i])[2])
    r2.append(eval(df['1'][i])[3])
df1=pd.read_csv(r'mean_covs_large'+str(d)+'.csv')
df1=np.array(df1)
covs=[]
for i in range(51):
    covs.append(df1[:,6*i+1:6*(i+1)])
covs=np.array(covs)
maps=[]
df2=pd.read_csv(r't_logistic_covs_large'+str(d)+'.csv')
df2.drop(columns='Unnamed: 0', inplace=True)
df2=np.array(df2)
approx_covs=[]
for i in range(51):
    approx_covs.append(df2[:,5*i:5*(i+1)])
approx_covs=np.array(approx_covs)
# print(approx_covs)
diff=np.subtract(covs, approx_covs)
# print(diff)
diff_norm=[]
for i in range(51):
    diff_norm.append(np.linalg.norm(diff[i], ord=2))


plt.plot(list(df['0']),r1, label='our bound on the \n difference of covariances')
plt.plot(list(df['0']),r2, '--', label='norm of the approximating covariance')
plt.plot(list(df['0']),diff_norm, ':', label='norm of the simulated \n true difference of covariances')
# plt.plot(list(df['0']),inverses, ':', label='true difference')
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':14})
plt.savefig('w2_logistic_large.png',bbox_inches = 'tight', pad_inches = 0)












