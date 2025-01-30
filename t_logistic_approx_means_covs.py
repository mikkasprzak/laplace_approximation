import pandas as pd
import numpy as np
from t_logistic_improved4 import mle_map_calculator, i_likelihood_posterior
import time
import math
from multiprocessing import Pool


def get_map_cov(nu , mu , sigma, d, n):
    print(d,n, " started")
    df_x=pd.read_csv(r'normal_data_small'+str(d)+'.csv')
    df_y = pd.read_csv(r'logistic_data_large' + str(d) + '.csv')
    data_x = df_x.values[:n]
    data_y1 = df_y.values[:n]
    data_y=[]
    for i in data_y1:
        data_y.append(i[0])
    inv_sigma=np.linalg.inv(sigma)
    map=mle_map_calculator(data_x,data_y,nu,mu,inv_sigma,d)[1]
    cov=i_likelihood_posterior(map, data_x, data_y, nu, mu, inv_sigma, n)[1]
    cov=np.linalg.inv(cov)/n
    print(d,n, " done", )
    return [map,cov]



d=5 # change the d to take values 1,2,3,4,5
nu=4
mu=[0]*d
sigma=np.identity(d)
df=pd.read_csv(r't_logistic_results_large'+str(d)+'.csv')
df.drop(columns='Unnamed: 0', inplace=True)
l=list(df['0'])

def res_pool(n):
    res=get_map_cov(nu,mu,sigma,d,n)
    return res[0]

if __name__ == '__main__':
    start = time.time()
    with Pool(16) as p:
        r=p.map(res_pool, l)
    df=pd.DataFrame(r)
    df.to_csv('t_logistic_maps_large'+str(d)+'.csv')
    end = time.time()
    print((end - start) / 60, " minutes")

def res_pool_cov(n):
    res=get_map_cov(nu,mu,sigma,d,n)
    return res[1]

if __name__ == '__main__':
    start = time.time()
    with Pool(16) as p:
        r=p.map(res_pool_cov, l)
    df=np.hstack(r)
    df = pd.DataFrame(df)
    df.to_csv('t_logistic_covs_large'+str(d)+'.csv')
