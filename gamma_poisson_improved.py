from main_univariate_improved import w1_sample_lower_bound as w1_sample_lower_bound
from main_univariate_improved import w1_bound as w1_bound_improved
from main_univariate_improved import w2_sample_lower_bound as w2_sample_lower_bound
from main_univariate_improved import w2_bound as w2_bound_improved
from main_univariate_improved import third_constant
import numpy as np
import scipy.special
from scipy.optimize import minimize_scalar
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from multiprocessing import Pool


def delta_optimizer(delta_fun,m1_hat_fun,m2_fun,i_mle,mle,size):
    def en3(c1):
        m2=m2_fun(c1)
        m1=m1_hat_fun(c1)
        delta=delta_fun(c1)
        return third_constant(delta,size,i_mle,m1,m2)
    res = minimize_scalar(fun=en3, bounds=(np.sqrt(1 / i_mle / size)/mle, 1), method='bounded')
    delta=delta_fun(res.x)
    m1_hat=m1_hat_fun(res.x)
    m2=m2_fun(res.x)
    return [delta,m1_hat,m2]


def poisson_gamma_parameters(alpha,beta,data,size):
    mle=np.sum(data)/size
    map=(size*mle+alpha-1)/(size+beta)
    i_mle = 1 / mle
    m2_bar = 250 * (size + beta) ** 3 / (64 * size * (size * mle + alpha - 1) ** 2)
    i_map = (size + beta) ** 2 / (size * (size * mle + alpha - 1))
    d = 1
    dif = abs(mle - map)
    integral = 1 / beta * np.sqrt(scipy.special.gamma(alpha + 2) / scipy.special.gamma(alpha))+mle
    integral_sq=2*(1 / beta * np.sqrt(scipy.special.gamma(alpha + 2) / scipy.special.gamma(alpha)))**2+2*mle**2
    """
    Optimising the delta parameter
    """
    def delta_fun(c1):
        return mle*c1
    def m1_hat_fun(c1):
        return scipy.special.gamma(alpha)/np.power(beta,alpha)*(1+c1)**(1-alpha)*mle**(1-alpha)*np.exp((1+c1)*beta*mle)
    def m2_fun(c1):
        return 2/mle**2/(1-c1)**3
    r=delta_optimizer(delta_fun,m1_hat_fun,m2_fun,i_mle,mle,size)
    delta=r[0]
    m1_hat=r[1]
    m2=r[2]
    """"
    The delta bar parameter
    """
    def delta_bar_fun(c):
        return c * map
    def kappa_bar_fun(c):
       delta_bar = delta_bar_fun(c)
       theta1 = mle + delta_bar - dif
       theta2 = mle - delta_bar + dif
       return -np.maximum((mle - theta1) + mle * (np.log(theta1 / mle)),
                            (mle - theta2) + mle * (np.log(theta2 / mle)))
    return {"size":size, "d":d, "delta":delta, "delta_bar":delta_bar_fun, "m1_hat":m1_hat, "m2":m2, "m2_bar":m2_bar, "i_mle":i_mle, "i_map":i_map, "mle":mle, "map":map, "kappa_bar":kappa_bar_fun,  "integral":integral, "integral_sq":integral_sq}

def bound_w1_gamma_poisson(alpha,beta,data,n):
    params = poisson_gamma_parameters(alpha, beta, data, n)
    delta = params.get("delta")
    delta_bar_fun = params.get("delta_bar")
    i_map = params.get("i_map")
    i_mle = params.get("i_mle")
    m2 = params.get("m2")
    m2_bar = params.get("m2_bar")
    m1_hat = params.get("m1_hat")
    kappa_bar_fun = params.get("kappa_bar")
    integral = params.get("integral")
    mle = params.get("mle")
    map = params.get("map")

    """
    Optimizing the bound with respect to delta bar
    """

    def w1(c):
        delta_bar = delta_bar_fun(c)
        kappa_bar = kappa_bar_fun(c)
        return w1_bound_improved(n, delta, delta_bar, i_map, i_mle, kappa_bar, integral, m2, m2_bar, m1_hat) / np.sqrt(n)
    res = minimize_scalar(fun=w1,
                          bounds=(max(abs(mle - map), np.sqrt(1 / i_map / n)) / map, i_map / m2_bar / map),
                          method='bounded')
    bound_map_logsobolev = res.fun
    return bound_map_logsobolev


def bound_w2_gamma_poisson(alpha,beta,data,n):
    params = poisson_gamma_parameters(alpha, beta, data, n)
    delta = params.get("delta")
    delta_bar_fun = params.get("delta_bar")
    i_map = params.get("i_map")
    i_mle = params.get("i_mle")
    m2 = params.get("m2")
    m2_bar = params.get("m2_bar")
    m1_hat = params.get("m1_hat")
    kappa_bar_fun = params.get("kappa_bar")
    integral_sq = params.get("integral_sq")
    mle = params.get("mle")
    map = params.get("map")

    """
    Optimizing the bound with respect to delta bar
    """

    def w2(c):
        delta_bar = delta_bar_fun(c)
        kappa_bar = kappa_bar_fun(c)
        return w2_bound_improved(n,delta, delta_bar, i_map, i_mle, kappa_bar, integral_sq, m2,m2_bar,m1_hat) / n
    res = minimize_scalar(fun=w2,
                          bounds=(max(abs(mle - map), np.sqrt(1 / i_map / n)) / map, i_map / m2_bar / map),
                          method='bounded')
    bound_map_logsobolev = res.fun
    return bound_map_logsobolev



def results_to_file(alpha,beta,n):
    df = pd.read_csv(r'exponential_data.csv')
    data1 = list(df['0'])
    data = data1[:n]
    boundw1=bound_w1_gamma_poisson(alpha, beta, data, n)
    boundw2=bound_w2_gamma_poisson(alpha,beta,data,n)+boundw1**2
    mle = np.sum(data) / n
    map = (n * mle + alpha - 1) / (n + beta)
    i_map = (n+ beta) ** 2 / (n* (n * mle + alpha - 1))
    mean = n / (n + beta) * mle + (beta / (n + beta)) * (alpha / beta)
    var = (alpha + n * mle) / (n + beta) ** 2
    print(n, " done")
    return [boundw1, mean, abs(map - mean), boundw2, var, abs(var - 1 / i_map / n)]

def res_pool(n):
    return results_to_file(0.1,3,n)
l=[0]*51
for i in range(51):
    l[i]=int(10**(2+3*i/50))
if __name__ == '__main__':
    with Pool(7) as p:
        r = p.map(res_pool, l)
    df = pd.DataFrame(r)
    df.to_csv('gamma_poisson_results1.csv')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

df=pd.read_csv(r'gamma_poisson_results1.csv')
df.drop(columns='Unnamed: 0', inplace=True)

'''
Producing first plot:
'''
plt.plot(l, df['0'], label=r'our bound on the 1-Wasserstein distance')
plt.plot(l, df['1'], '--', label='true mean')
plt.plot(l, df['2'], ':', label='true difference of means')
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':12})
plt.savefig('w1_gamma.png',bbox_inches = 'tight', pad_inches = 0)

plt.clf()

'''
Producing second plot
'''

plt.plot(l, df['3'], label=r'our bound on the difference of variances')
plt.plot(l, df['4'], '--', label='true variance')
plt.plot(l, df['5'], ':', label='true difference of variances')
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':12})
plt.savefig('w2_gamma.png',bbox_inches = 'tight', pad_inches = 0)











