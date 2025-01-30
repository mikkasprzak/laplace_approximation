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

'''
Producing the plot of our W1 bound and the norm of the approximating mean - small parameter
'''
df=pd.read_csv(r't_logistic_results_zero'+str(d)+'.csv')
df.drop(columns='Unnamed: 0', inplace=True)
r1=[]
r2=[]
for i in range(51):
    r1.append(eval(df['1'][i])[0])
    r2.append(eval(df['1'][i])[1])

plt.clf()

plt.plot(list(df['0']),r1, label='our bound on the 1-Wasserstein distance')
plt.plot(list(df['0']),r2, '--', label='norm of the approximating mean')
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':12})
plt.savefig('w1_logistic_zero.png',bbox_inches = 'tight', pad_inches = 0)


'''
Producing the plot of our W2 bound and the norm of the approximating covariance - small parameter
'''
df=pd.read_csv(r't_logistic_results_zero'+str(d)+'.csv')
df.drop(columns='Unnamed: 0', inplace=True)
r1=[]
r2=[]
for i in range(51):
    r1.append(eval(df['1'][i])[2])
    r2.append(eval(df['1'][i])[3])

plt.clf()

plt.plot(list(df['0']),r1, label='our bound on the 2-norm of the \n difference of covariances')
plt.plot(list(df['0']),r2, '--', label='2-norm of the approximating covariance')
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':12})
plt.savefig('w2_logistic_zero.png',bbox_inches = 'tight', pad_inches = 0)


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
# inverses=[]
#
# for i in list(df['0']):
#     inverses.append(400/i**(3/2))
plt.plot(list(df['0']),l[0], label='logistic parameter = (0,...,0)')
plt.plot(list(df1['0']),l[1], '--', label='logistic parameter = (0.5,...,0.5)')
plt.plot(list(df2['0']),l[2],':', label='logistic parameter = (1,...,1)')
# plt.plot(list(df['0']),inverses)
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
# inverses=[]
#
# for i in list(df['0']):
#     inverses.append(400/i**(3/2))
plt.plot(list(df['0']),l[0], label='logistic parameter = (0,...,0)')
plt.plot(list(df1['0']),l[1], '--', label='logistic parameter = (0.5,...,0.5)')
plt.plot(list(df2['0']),l[2],':', label='logistic parameter = (1,...,1)')
# plt.plot(list(df['0']),inverses)
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':16})
plt.savefig('w2_logistic_comparison_new.png',bbox_inches = 'tight', pad_inches = 0)





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
# inverses=[]
#
# for i in list(df['0']):
#     inverses.append(400/i**(3/2))
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

d=5
df=pd.read_csv(r't_logistic_results_large'+str(d)+'.csv')
m=list(df['0'])
df = pd.DataFrame(m)
df.to_csv(r'logistic_indices5_large.csv',index=False)

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