from main_univariate_improved import w1_sample_lower_bound as w1_sample_lower_bound
from main_univariate_improved import w1_bound as w1_bound_improved
from main_univariate_improved import w2_bound as w2_bound_improved
from main_univariate_improved import third_constant
import numpy as np
import scipy.special
from scipy.optimize import minimize_scalar
from scipy.optimize import fsolve as fsolve
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.stats import invgamma
from multiprocessing import Pool

def delta_optimizer(m1_hat_fun,m2_fun,i_mle,mle,n):
    def en3(delta):
        m2=m2_fun(delta)
        m1=m1_hat_fun(delta)
        return third_constant(delta,n,i_mle,m1,m2)
    res = minimize_scalar(fun=en3, bounds=(np.sqrt(1 / i_mle / n), mle), method='bounded')
    delta=res.x
    m1_hat=m1_hat_fun(delta)
    m2=m2_fun(delta)
    return [delta,m1_hat,m2]


def weibull_inverse_gamma_parameters(k,alpha,beta,data,n):
    mlek=0
    for i in data:
        mlek+=np.power(i,k)
    mle=mlek/n
    map=(mlek+beta)/(n+alpha+1)
    i_mle = 1 / mle ** 2
    i_map = -1 / map ** 2 + 2 * mle / map ** 3 - (alpha + 1) / n / map ** 2 + 2 * beta / map ** 3 / n
    d = 1
    dif = abs(mle - map)
    """
    The delta parameter
    """
    def m1_hat_fun(delta):
        def pi_inv(theta):
            return scipy.special.gamma(alpha)/beta**alpha*theta**(alpha+1)*np.exp(beta/theta)
        return max(pi_inv(mle-delta),pi_inv(mle+delta))
    def m2_fun(delta):
        return (6*mlek-2*n*(mle-delta))/(mle-delta)**4

    r = delta_optimizer(m1_hat_fun, m2_fun, i_mle, mle, n)
    delta = r[0]
    m1_hat = r[1]
    m2 = r[2]
    """
    The delta bar parameter
    """
    def m2_bar_fun(delta_bar):
        def l_n_bar_third(theta):
            return -2/theta**3 + 6*mle/theta**4-2*(alpha+1)/n/theta**3+6*beta/n/theta**4
        return l_n_bar_third(map-delta_bar)

    def kappa_bar_fun(delta_bar):
        def l_n(theta):
            return - n*np.log(theta)-mlek/theta
        term1 = (l_n(mle-delta_bar+dif)-l_n(mle))/n
        term2 = (l_n(mle+delta_bar-dif)-l_n(mle))/n
        return -np.maximum(term1,term2)

    integral=beta/np.sqrt((alpha-1)*(alpha-2))+mle
    integral_sq=2*(beta/np.sqrt((alpha-1)*(alpha-2)))**2+2*mle**2

    return {"mlek":mlek, "size":n, "d":d, "delta":delta, "m1_hat":m1_hat, "m2":m2, "m2_bar":m2_bar_fun, "i_mle":i_mle, "i_map":i_map, "mle":mle, "map":map, "kappa_bar":kappa_bar_fun, "integral":integral, "integral_sq":integral_sq}


def upper_bound_for_delta_bar(alpha,n,map,i_map):
    def f(x):
        return x+x**4*i_map/(2+(2*alpha+2)/n)/(3*map-x)-map
    def fprime(x):
        return 1+3*i_map*n*x**3*(4*map-x)/2/(alpha+n+1)/(x-3*map)**2
    def fprime2(x):
        return 3*i_map*n*x**2*(18*map**2-8*map*x+x**2)/(alpha+n+1)/(3*map-x)**3
    u_res=root_scalar(f=f, x0=map/2, fprime=fprime, fprime2=fprime2, method='halley', xtol=0.000000000001)
    u=u_res.root
    return map-u

def bound_w1_weibull_inverse_gamma(k,alpha,beta,data,n):
    params = weibull_inverse_gamma_parameters(k, alpha, beta, data, n)
    delta = params.get("delta")
    i_map = params.get("i_map")
    i_mle = params.get("i_mle")
    m2 = params.get("m2")
    m2_bar_fun = params.get("m2_bar")
    m1_hat = params.get("m1_hat")
    kappa_bar_fun = params.get("kappa_bar")
    integral = params.get("integral")
    integral_sq = integral ** 2
    mle = params.get("mle")
    map = params.get("map")

    """
    Optimizing the bound with respect to delta bar
    """

    def w1(delta_bar):
        kappa_bar = kappa_bar_fun(delta_bar)
        m2_bar = m2_bar_fun(delta_bar)
        return w1_bound_improved(n, delta, delta_bar, i_map, i_mle, kappa_bar, integral, m2, m2_bar, m1_hat) / np.sqrt(n)

    u=upper_bound_for_delta_bar(alpha,n,map,i_map)-0.000000000001
    res = minimize_scalar(fun=w1,
                          bounds=(max(abs(mle - map), np.sqrt(1 / i_map / n)) , min(map, u)),
                          method='bounded')
    delta_bar=res.x
    bound_map_logsobolev = w1(delta_bar)
    return bound_map_logsobolev

def bound_w2_weibull_inverse_gamma(k,alpha,beta,data,n):
    params = weibull_inverse_gamma_parameters(k, alpha, beta, data, n)
    delta = params.get("delta")
    i_map = params.get("i_map")
    i_mle = params.get("i_mle")
    m2 = params.get("m2")
    m2_bar_fun = params.get("m2_bar")
    m1_hat = params.get("m1_hat")
    kappa_bar_fun = params.get("kappa_bar")
    integral_sq = params.get("integral_sq")
    mle = params.get("mle")
    map = params.get("map")

    """
    Optimizing the bound with respect to delta bar
    """

    def w2(delta_bar):
        kappa_bar = kappa_bar_fun(delta_bar)
        m2_bar = m2_bar_fun(delta_bar)
        return w2_bound_improved(n,delta, delta_bar, i_map, i_mle, kappa_bar, integral_sq, m2,m2_bar,m1_hat) / n

    u=upper_bound_for_delta_bar(alpha,n,map,i_map)
    res = minimize_scalar(fun=w2,
                          bounds=(max(abs(mle - map), np.sqrt(1 / i_map / n)), min(map, u)),
                          method='bounded')
    delta_bar=res.x
    bound_map_logsobolev = w2(delta_bar)
    return bound_map_logsobolev

def results_to_file(k,alpha,beta,n):
    df = pd.read_csv(r'weibull_data.csv')
    data1 = list(df['0'])
    data = data1[:n]
    boundw1=bound_w1_weibull_inverse_gamma(k,alpha, beta, data, n)
    boundw2=bound_w2_weibull_inverse_gamma(k,alpha,beta,data,n)+boundw1**2
    mlek = sum(np.power(data, k))
    mle = mlek / n
    map = (mlek + beta) / (n + alpha + 1)
    i_map = -1 / map ** 2 + 2 * mle / map ** 3 - (alpha + 1) / n / map ** 2 + 2 * beta / map ** 3 / n
    mean = (mlek + beta) / (alpha + n - 1)
    var = (mlek + beta) ** 2 / (alpha + n - 1) ** 2 / (alpha + n - 2)
    print(n, " done")
    return [boundw1, mean, abs(map - mean), boundw2, var, abs(var - 1 / i_map / n)]


def res_pool(n):
    return results_to_file(1/2,3,10,n)
l=[0]*51
for i in range(51):
    l[i]=int(10**(2+3*i/50))
if __name__ == '__main__':
    with Pool(7) as p:
        r = p.map(res_pool, l)
    df = pd.DataFrame(r)
    df.to_csv('weibull_invgamma_results1.csv')

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

df=pd.read_csv(r'weibull_invgamma_results1.csv')
df.drop(columns='Unnamed: 0', inplace=True)

plt.plot(l, df['0'], label=r'our bound on the 1-Wasserstein distance')
plt.plot(l, df['1'], '--', label='true mean')
plt.plot(l, df['2'], ':', label='true difference of means')
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':12})
plt.savefig('w1_weibull.png',bbox_inches = 'tight', pad_inches = 0)

plt.clf()


plt.plot(l, df['3'], label=r'our bound on the difference of variances')
plt.plot(l, df['4'], '--', label='true variance')
plt.plot(l, df['5'], ':', label='true difference of variances')
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':12})
plt.savefig('w2_weibull.png',bbox_inches = 'tight', pad_inches = 0)





