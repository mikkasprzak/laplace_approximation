import pandas as pd
import numpy as np
from t_logistic_improved4 import bound_w1_t_logistic, bound_w2_t_logistic
import time
import math
from multiprocessing import Pool

'''
Generic: returning the final results in a format ready for writing into a file:
'''
def results_to_file(nu , mu , sigma, d, n):
    print(d,n, " started")
    df_x=pd.read_csv(r'normal_data_small'+str(d)+'.csv')
    df_y = pd.read_csv(r'logistic_data_large' + str(d) + '.csv')
    data_x = df_x.values[:n]
    data_y1 = df_y.values[:n]
    data_y=[]
    for i in data_y1:
        data_y.append(i[0])
    bound_and_map=bound_w1_t_logistic(data_x,data_y,nu,mu,sigma,n,d)
    bound_and_var=bound_w2_t_logistic(data_x, data_y, nu, mu, sigma, n, d)
    boundw1=bound_and_map[0]
    map_norm=np.linalg.norm(bound_and_map[1])
    boundw2 = bound_and_var[0]+boundw1**2
    var=bound_and_var[1]
    print(d,n, " done", boundw1)
    return [boundw1, map_norm,boundw2, var]




df5=pd.read_csv(r'lower_bounds_logistic_large5.csv')
df5.drop(columns='Unnamed: 0', inplace=True)
df5=list(df5['0'])
val=df5



d=5 # change the d to take values 1,2,3,4,5
nu=4
mu=[0]*d
sigma=np.identity(d)
l=[0]*51
first=math.log10(val[0]+10)
for i in range(51):
    l[i]=int(10**(first+(6-first)*i/51))+10

def res_pool(n):
    sample=n+1
    for i in range(20000):
        try:
            res=results_to_file(nu,mu,sigma,d,n+i)
        except:
            print("Problem with n:", n+i)
        else:
            sample=n+i
            break
    return [sample, res]

if __name__ == '__main__':
    start = time.time()
    with Pool(16) as p:
        r=p.map(res_pool, l)
    df=pd.DataFrame(r)
    df.to_csv('t_logistic_results_large'+str(d)+'.csv')
    end = time.time()
    print((end - start) / 60, " minutes")