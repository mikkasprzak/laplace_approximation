import pandas as pd
import numpy as np
from t_logistic_improved4 import mle_map_calculator, i_likelihood_posterior
import time
import math
from multiprocessing import Pool
import matplotlib.pyplot as plt


def get_condition_numb(df_x,df_y, nu , mu , sigma, d, n):
    print(d,n, " started")
    data_x = df_x.values[:n]
    data_y1 = df_y.values[:n]
    data_y=[]
    for i in data_y1:
        data_y.append(i[0])
    inv_sigma=np.linalg.inv(sigma)
    map=mle_map_calculator(data_x,data_y,nu,mu,inv_sigma,d)[1]
    cov=i_likelihood_posterior(map, data_x, data_y, nu, mu, inv_sigma, n)[1]
    cond=np.linalg.norm(cov,2)/np.linalg.norm(cov,-2)
    print(d,n, " done", )
    return cond

# def get_condition_numb_and_map(df_x,df_y, nu , mu , sigma, d, n):
#     # print(d,n, " started")
#     data_x = df_x.values[:n]
#     data_y1 = df_y.values[:n]
#     data_y=[]
#     for i in data_y1:
#         data_y.append(i[0])
#     inv_sigma=np.linalg.inv(sigma)
#     map=mle_map_calculator(data_x,data_y,nu,mu,inv_sigma,d)[1]
#     cov=i_likelihood_posterior(map, data_x, data_y, nu, mu, inv_sigma, n)[1]
#     cond=np.linalg.norm(cov,2)/np.linalg.norm(cov,-2)
#     print(n, " done", )
#     return [cond,np.linalg.norm(map)]

'''
Condition number calculation
'''

d=5 # change the d to take values 1,2,3,4,5
nu=4
mu=[0]*d
sigma=np.identity(d)

df_x=pd.read_csv(r'normal_data_small'+str(d)+'.csv')
df_y_large = pd.read_csv(r'logistic_data_large' + str(d) + '.csv')
df_y_zero = pd.read_csv(r'logistic_data_zero' + str(d) + '.csv')
df_y_medium = pd.read_csv(r'logistic_data_medium' + str(d) + '.csv')



l=np.linspace(3,6,10000)
l=[int(10**i) for i in l]

def res_pool_large(n):
    res=get_condition_numb(df_x,df_y_large,nu,mu,sigma,d,n)
    return res

def res_pool_zero(n):
    res=get_condition_numb(df_x,df_y_zero,nu,mu,sigma,d,n)
    return res

def res_pool_medium(n):
    res=get_condition_numb(df_x,df_y_medium,nu,mu,sigma,d,n)
    return res

if __name__ == '__main__':
    start = time.time()
    with Pool(16) as p:
        r_large=p.map(res_pool_large, l)
        r_zero = p.map(res_pool_zero, l)
        r_medium = p.map(res_pool_medium, l)
    df_l = pd.DataFrame(r_large)
    df_z = pd.DataFrame(r_zero)
    df_m = pd.DataFrame(r_medium)
    df_l.to_csv('t_logistic_condition_numbers_large' + str(d) + '.csv')
    df_m.to_csv('t_logistic_condition_numbers_medium' + str(d) + '.csv')
    df_z.to_csv('t_logistic_condition_numbers_zero' + str(d) + '.csv')
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 16
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    plt.plot(l, r_zero, label=r'logistic parameter = (0,...0)')
    plt.plot(l, r_medium, '--', label=r'logistic parameter = (0.5,...0.5)')
    plt.plot(l, r_large, ':', label=r'logistic parameter = (1,...1)')
    plt.xlabel('number of data points')
    plt.xscale('log', base=10)
    plt.legend(prop={'size': 16})
    plt.savefig('w1_logistic_condition.png', bbox_inches='tight', pad_inches=0)


'''
Condition number relative to MAP
'''

# d=5 # change the d to take values 1,2,3,4,5
# nu=4
# mu=[0]*d
# sigma=np.identity(d)
#
# df_x=pd.read_csv(r'normal_data_small'+str(d)+'.csv')
# df_y_large = pd.read_csv(r'logistic_data_large' + str(d) + '.csv')
# df_y_zero = pd.read_csv(r'logistic_data_zero' + str(d) + '.csv')
# df_y_medium = pd.read_csv(r'logistic_data_medium' + str(d) + '.csv')

# df_large=pd.read_csv(r't_logistic_results_large'+str(d)+'.csv')
# df_zero=pd.read_csv(r't_logistic_results_zero'+str(d)+'.csv')
# df_medium=pd.read_csv(r't_logistic_results_medium'+str(d)+'.csv')
#
# df_large.drop(columns='Unnamed: 0', inplace=True)
# df_zero.drop(columns='Unnamed: 0', inplace=True)
# df_medium.drop(columns='Unnamed: 0', inplace=True)
#
# r_large=[]
# r_zero=[]
# r_medium=[]
#
# for i in range(51):
#     r_large.append(eval(df_large['1'][i])[1])
#     r2.append(eval(df['1'][i])[1])

# l=np.linspace(3,6,50)
# l=[int(10**i) for i in l]
#
# def res_pool_large(n):
#     res=get_condition_numb_and_map(df_x,df_y_large,nu,mu,sigma,d,n)
#     return res[0]
#
# def res_pool_zero(n):
#     res=get_condition_numb_and_map(df_x,df_y_zero,nu,mu,sigma,d,n)
#     return res[0]
#
# def res_pool_medium(n):
#     res=get_condition_numb_and_map(df_x,df_y_medium,nu,mu,sigma,d,n)
#     return res[0]
#
# if __name__ == '__main__':
#     start = time.time()
#     with Pool(16) as p:
#         r_large = p.map(res_pool_large, l)
#         r_zero = p.map(res_pool_zero, l)
#         r_medium = p.map(res_pool_medium, l)
#     plt.clf()
#     plt.plot(l, r_zero, label=r'condition number/map - zero parameter')
#     plt.plot(l, r_medium, '--', label=r'condition number/map - medium parameter')
#     plt.plot(l, r_large, ':', label=r'condition number/map - large parameter')
#     plt.xlabel('number of data points')
#     plt.yscale('log', base=10)
#     plt.xscale('log', base=10)
#     plt.legend(prop={'size': 12})
#     plt.savefig('w1_logistic_condition_map.png', bbox_inches='tight', pad_inches=0)

# '''
# Zero parameter
# '''
# d=5 # change the d to take values 1,2,3,4,5
# nu=4
# mu=[0]*d
# sigma=np.identity(d)
#
# l=np.linspace(1000,10**6,51)
# l=[int(i) for i in l]
#
# def res_pool(n):
#     res=get_condition_numb(df_x,df_y,nu,mu,sigma,d,n)
#     return res
#
# if __name__ == '__main__':
#     start = time.time()
#     with Pool(16) as p:
#         r_zero=p.map(res_pool, l)
#     print(np.average(r_zero))
#
# '''
# Middle parameter
# '''
# d=5 # change the d to take values 1,2,3,4,5
# nu=4
# mu=[0]*d
# sigma=np.identity(d)
# df_x=pd.read_csv(r'normal_data_small'+str(d)+'.csv')
# df_y = pd.read_csv(r'logistic_data_medium' + str(d) + '.csv')
# df=pd.read_csv(r't_logistic_results_medium'+str(d)+'.csv')
# df.drop(columns='Unnamed: 0', inplace=True)
# l=np.linspace(1000,10**6,51)
# l=[int(i) for i in l]
#
# def res_pool(n):
#     res=get_condition_numb(df_x,df_y,nu,mu,sigma,d,n)
#     return res
#
# if __name__ == '__main__':
#     start = time.time()
#     with Pool(16) as p:
#         r_medium=p.map(res_pool, l)
#     print(np.average(r_medium))

