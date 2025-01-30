from t_logistic_improved4 import m2_bar_fun, t_logistic_parameters
from main_multivariate import sample_size_check
import pandas as pd
import numpy as np
import time

def t_logistic_sample_size_check(n,d,nu,mu,sigma):
    df_x = pd.read_csv(r'normal_data_small' + str(d) + '.csv')
    df_y = pd.read_csv(r'logistic_data_zero' + str(d) + '.csv')
    data_x = df_x.values[:n]
    data_y1 = df_y.values[:n]
    data_y = []
    for i in data_y1:
        data_y.append(i[0])
    params = t_logistic_parameters(data_x,data_y,nu,mu,sigma,n,d)
    i_map = params.get("i_map")
    mle = params.get("mle")
    map = params.get("map")
    inv_sigma = np.linalg.inv(sigma)
    def m2_bar_fun1(x):
        return m2_bar_fun(x,data_x,data_y,n,inv_sigma,nu,mu,map,d)
    return sample_size_check(i_map, m2_bar_fun1,mle,map,n)


def results(d,start):
    n = start
    nu = 4
    mu = [0] * d
    sigma = np.identity(d)
    for i in range(100000):
        print("checking d,n: ", d, n)
        if t_logistic_sample_size_check(n,d,nu,mu,sigma):
            print(d,n, " successful")
            return n
        if d in [1,2]:
            n = n + 10
        if d in [3,4,5]:
            n=n+20
        if d>=6:
            n=n+50
        # if d>=5:
        #     n=n+500


n=10
start = time.time()
for d in range(1,10):
    r = []
    n=results(d,n)
    r.append(n)
    r=pd.DataFrame(r)
    r.to_csv('lower_bounds_logistic_zero'+str(d)+
                     '.csv')
end = time.time()