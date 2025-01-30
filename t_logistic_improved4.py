from numpy.f2py.rules import options

from main_multivariate import w1_bound
from main_multivariate import w2_bound
import scipy.special
import numpy as np
from main_multivariate import third_constant
import scipy.optimize
from scipy.optimize import minimize_scalar
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from multiprocessing import Pool
import math
import time
from scipy.optimize import root_scalar
import jax
import jax.numpy as jnp
from scipy.special import expit
from jax.nn import sigmoid

'''
Model specific: Calculation of the logistic log-likelihood
'''
def log_likelihood(theta,data_x,data_y):
    res = np.sum(np.log(expit(np.multiply(np.dot(data_x, theta), data_y))))
    return res

'''
Model specific: Calculation of Student's t log-prior
'''
def log_prior(theta,nu,mu,inv_sigma,d):
    theta=np.array(theta)
    mu=np.array(mu)
    return -(nu + d) / 2 * np.log(1 + 1 / nu * np.dot(np.transpose(theta- mu),
                                              np.dot(inv_sigma, theta - mu)))

'''
Model specific: bound on the inverse prior as a function of radius delta
'''
def m1_hat_fun(delta,sigma_evalues,mu,nu,map,d):
    term1 = scipy.special.gamma(nu / 2) * nu ** (d / 2) * np.pi ** (d / 2) * np.prod(sigma_evalues) ** (
                1 / 2) / scipy.special.gamma((nu + d) / 2)
    term2 = (1 + 1 / nu / sigma_evalues[0] * (2 * delta ** 2 + 2 * np.linalg.norm(mu - map))) ** ((nu + d) / 2)
    return term1 * term2

'''
Model specific: Calculation of the bound on the norm of the third derivative of log-likelihood, as a function of radius delta
'''

def bound_fun(theta,data_x,data_y,n):
    exponent = np.inner(theta, data_x) * data_y
    e = np.exp(exponent)
    # arr=np.linalg.norm(data_x,axis=1)
    # print(arr.shape)
    res=sum(np.linalg.norm(data_x,axis=1) ** 3 / n * e * abs(e - 1) / (1 + e) ** 3)
    return res

# Old slower calculation:
# def bound_fun(theta,data_x,data_y,n):
#     res = 0
#     for i in range(n):
#         exponent = np.inner(theta, data_x[i]) * data_y[i]
#         e = np.exp(exponent)
#         res = res + np.linalg.norm(data_x[i]) ** 3 / n * e * abs(e - 1) / (1 + e) ** 3
#         try:
#             if res < 0:
#                 print("Alert")
#         except:
#             print("Bound fun:", res)
#     return res

def m2_fun(delta,data_x,data_y,n,mle,d):
    def fun(x):
        return -bound_fun(x,data_x,data_y,n)
    l = []
    for i in range(d):
        l = l + [(mle[i] - delta, mle[i] + delta)]
    l = tuple(l)
    r = scipy.optimize.shgo(func=fun, bounds=l)
    return -r.fun

'''
Model specific: Calculation of the bound on the norm of the third derivative of log-posterior, as a function of radius delta bar
'''
def third_der_like(delta_bar,data_x,data_y,n,map,d):
    def fun(x):
        return -bound_fun(x,data_x,data_y,n)
    l = []
    for i in range(d):
        l = l + [(map[i] - delta_bar, map[i] + delta_bar)]
    l = tuple(l)
    r = scipy.optimize.shgo(func=fun, bounds=l)
    # print("done with delta bar:", delta_bar)
    return -r.fun



def m2_bar_fun(delta_bar,data_x,data_y,n,inv_sigma,nu,mu,map,d):
    matrix=inv_sigma+np.diag(np.diag(inv_sigma))
    op_norm=np.linalg.norm(matrix,2)
    term1=3*(nu+d)/nu**2*op_norm**2*(delta_bar+np.linalg.norm(map-mu))**2
    term2=2*(nu+d)/nu**3*op_norm**3*(delta_bar+np.linalg.norm(map-mu))**3
    res=(term1+term2)/n+third_der_like(delta_bar,data_x,data_y,n,map,d)
    # print("Delta bar:", delta_bar, "M2_bar:", res)
    return res


'''
Model specific: calculation of kappa bar
'''
def kappa_bar_fun(delta_bar,data_x,data_y,mle,map,min_evalue_mle,n,d):
    term1=third_der_like(delta_bar,data_x,data_y,n,map,d)*(delta_bar-np.linalg.norm(mle-map))**3
    term2=min_evalue_mle*(delta_bar-np.linalg.norm(mle-map))**2
    return term2-term1


'''
Model specific: Fisher information matrices (using JAX)
'''
def i_likelihood_posterior(theta, data_x, data_y, nu, mu, inv_sigma, n):
    # Convert inputs to JAX arrays
    data_x = jnp.array(data_x)
    data_y = jnp.array(data_y)
    theta = jnp.array(theta)
    mu = jnp.array(mu)
    inv_sigma = jnp.array(inv_sigma)

    # Dimension of data_x
    d = data_x.shape[1]  # Assuming data_x is concrete

    # Define g for i_likelihood
    g = lambda x: -jnp.sum(jnp.log(sigmoid(jnp.multiply(jnp.dot(data_x, x), data_y)))) / n
    i_likelihood = jax.hessian(g)

    # Define h for i_prior
    h = lambda x: (nu + d) / 2 * jnp.log(
        1 + 1 / nu * jnp.dot(jnp.transpose(x - mu), jnp.dot(inv_sigma, x - mu))
    ) / n
    i_prior = jax.hessian(h)

    # Compute results
    likelihood_value = i_likelihood(theta)
    prior_value = i_prior(theta)

    return [likelihood_value, likelihood_value + prior_value]

# Old hard-coded Hessians
# def i_likelihood_posterior(theta,data_x,data_y,nu,mu,inv_sigma,n):
#     i_likelihood = np.zeros((d, d))
#     for i in range(n):
#         i_likelihood = i_likelihood + np.outer(data_x[i], data_x[i]) * np.exp(np.inner(data_x[i], theta) * data_y[i]) / (
#                     1 + np.exp(np.inner(data_x[i], theta) * data_y[i])) ** 2
#     i_likelihood = i_likelihood / n
#
#     term1 = 0
#     for j in range(d):
#         for i in range(d):
#             term1 = term1 + (theta[i] - mu[i]) * (theta[j] - mu[j]) * inv_sigma[i, j]
#     term1 = term1 / nu + 1
#     term2 = [0] * d
#     for j in range(d):
#         term2[j] = inv_sigma[j, j] * (theta[j] - mu[j])
#     term3 = [0] * d
#     for j in range(d):
#         for l in range(d):
#             term3[j] = term3[j] + inv_sigma[l, j] * (theta[l] - mu[l])
#     res = np.zeros((d, d))
#     for i in range(d):
#         for j in range(d):
#             res[i, j] = -1 / nu ** 2 * (term2[j] + term3[j]) * (term2[i] + term3[i]) / term1 ** 2
#             if i == j:
#                 res[i, j] = res[i, j] + 2 / nu * inv_sigma[i, j] / term1
#             else:
#                 res[i, j] = res[i, j] + 1 / nu * inv_sigma[i, j] / term1
#             res[i, j] = res[i, j] * (nu + d) / 2
#     i_prior=res/n
#     # print(theta,': ',i_likelihood+i_prior)
#     return [i_likelihood,i_likelihood+i_prior]


''''
Generic: Calculation of MLE and MAP
'''
def mle_map_calculator(data_x,data_y,nu,mu,inv_sigma,d):
    def minus_log_likelihood(theta):
        return - log_likelihood(theta,data_x,data_y)
    def minus_log_posterior(theta):
        return minus_log_likelihood(theta)-log_prior(theta,nu,mu,inv_sigma,d)
    x0=[1]*d
    r=scipy.optimize.minimize(fun=minus_log_likelihood, x0=x0)
    mle=r.x
    x_0=mle
    s=scipy.optimize.minimize(fun=minus_log_posterior, x0=x0)
    map=s.x
    return [mle,map]




'''
Generic: a function that will optimise the choice of the radius delta
'''
def delta_optimizer(data_x,data_y,sigma_evalues,mu,nu,map,mle,d,i_mle,n):
    evalues=np.linalg.eigh(i_mle)[0]
    inv_evalues=1/evalues
    def en3(delta):
        m2 = m2_fun(delta,data_x,data_y,n,mle,d)
        m1 = m1_hat_fun(delta,sigma_evalues,mu,nu,map,d)
        return third_constant(d,delta,n,i_mle,m1,m2)
    res = minimize_scalar(fun=en3, bounds=(np.sqrt(sum(inv_evalues)/n), np.sqrt(sum(inv_evalues)/n)+1), method='bounded')
    delta=res.x
    m1_hat=m1_hat_fun(delta,sigma_evalues,mu,nu,map,d)
    return [delta,m1_hat]

'''
Generic: takes in the data and the parameters of the prior and calculates various statistics of the posterior
'''
def t_logistic_parameters(data_x,data_y,nu,mu,sigma,n,d):
    sigma_evalues = np.linalg.eigh(sigma)[0]
    inv_sigma = np.linalg.inv(sigma)
    mle_map=mle_map_calculator(data_x, data_y, nu, mu, inv_sigma,d)
    mle=mle_map[0]
    map=mle_map[1]
    i_mle=i_likelihood_posterior(mle,data_x,data_y,nu,mu,inv_sigma,n)[0]
    i_map=i_likelihood_posterior(map,data_x,data_y,nu,mu,inv_sigma,n)[1]
    i_mle_evalues = np.linalg.eigh(i_mle)[0]
    i_map_evalues = np.linalg.eigh(i_map)[0]
    min_evalue_mle = i_mle_evalues[0]
    min_evalue_map = i_map_evalues[0]
    i_map_inv_trace = sum(1/i_map_evalues)

    """
    The optimal delta parameter and M1_hat and M_2 evaluated at this optimal delta parameter
    """
    delta_opt = delta_optimizer(data_x,data_y,sigma_evalues,mu,nu,map,mle,d,i_mle,n)
    delta = delta_opt[0]
    m1_hat = delta_opt[1]
    m2 = m2_fun(delta,data_x,data_y,n,mle,d)

    """
    Prior integral bounds
    """
    integral=np.sqrt(nu/(nu-2)*sum(sigma_evalues))+np.linalg.norm(mle)
    integral_sq = 2*nu / (nu - 2) * sum(sigma_evalues) + 2*np.linalg.norm(mle)**2

    return {"size": n, "d": d, "delta": delta, "m1_hat": m1_hat, "m2": m2, "i_mle": i_mle, "i_map": i_map, "mle": mle, "map": map, "integral": integral, "integral_sq": integral_sq, "min_evalue_mle":min_evalue_mle,"min_evalue_map":min_evalue_map, "i_map_inv_trace":i_map_inv_trace}

'''
Generic: calculation of the W1 bound
'''
def bound_w1_t_logistic(data_x,data_y,nu,mu,sigma,n,d):
    params = t_logistic_parameters(data_x,data_y,nu,mu,sigma,n,d)
    i_map = params.get("i_map")
    # print("Hessian norm:",np.linalg.norm(i_map))
    i_mle = params.get("i_mle")
    integral = params.get("integral")
    m2 = params.get("m2")
    m1_hat = params.get("m1_hat")
    delta = params.get("delta")
    map=params.get("map")
    mle=params.get("mle")
    min_evalue_mle = params.get("min_evalue_mle")
    min_evalue_map = params.get("min_evalue_map")
    i_map_inv_trace= params.get("i_map_inv_trace")
    dif=np.linalg.norm(map-mle)
    inv_sigma = np.linalg.inv(sigma)
    def w1(delta_bar):
        kappa_bar = kappa_bar_fun(delta_bar,data_x,data_y,mle,map,min_evalue_mle,n,d)
        m2_bar = m2_bar_fun(delta_bar,data_x,data_y,n,inv_sigma,nu,mu,map,d)
        # print('n=',n,'m2_bar=',m2_bar)
        return w1_bound(n,d,delta,delta_bar,i_map, i_mle, kappa_bar, integral, m2,m2_bar,m1_hat)/np.sqrt(n)
    lower_bound=max(dif, np.sqrt(i_map_inv_trace/n))
    # if lower_bound>1:
    #     print("too small n")
    #     raise
    def upper_bound(x):
        return min_evalue_map / m2_bar_fun(min(x,1),data_x,data_y,n,inv_sigma,nu,mu,map,d)
    arg=scipy.optimize.fixed_point(func=upper_bound, x0=1)
    upper_bound = min_evalue_map / m2_bar_fun(min(arg,1),data_x,data_y,n,inv_sigma,nu,mu,map,d)# looking for an upper bound for delta_bar
    if upper_bound < lower_bound:
        raise
    res = minimize_scalar(fun=w1, bounds=(lower_bound, min(upper_bound, min(arg,1))), method='bounded')
    return [res.fun, map]

'''
Generic: calculation of the covariance bound
'''
def bound_w2_t_logistic(data_x,data_y,nu,mu,sigma,n,d):
    params = t_logistic_parameters(data_x,data_y,nu,mu,sigma,n,d)
    i_map = params.get("i_map")
    i_mle = params.get("i_mle")
    integral_sq = params.get("integral_sq")
    m2 = params.get("m2")
    m1_hat = params.get("m1_hat")
    delta = params.get("delta")
    map=params.get("map")
    mle=params.get("mle")
    min_evalue_mle = params.get("min_evalue_mle")
    min_evalue_map = params.get("min_evalue_map")
    i_map_inv_trace= params.get("i_map_inv_trace")
    dif=np.linalg.norm(map-mle)
    inv_sigma = np.linalg.inv(sigma)

    def w2(delta_bar):
        kappa_bar = kappa_bar_fun(delta_bar,data_x,data_y,mle,map,min_evalue_mle,n,d)
        m2_bar = m2_bar_fun(delta_bar,data_x,data_y,n,inv_sigma,nu,mu,map,d)
        return w2_bound(n,d,delta, delta_bar, i_map, i_mle, kappa_bar, integral_sq, m2,m2_bar,m1_hat)/n
    lower_bound=max(dif, np.sqrt(i_map_inv_trace/n))
    if lower_bound>1:
        print("too small n")
        raise
    def upper_bound(x):
        return min_evalue_map / m2_bar_fun(min(x,1),data_x,data_y,n,inv_sigma,nu,mu,map,d)
    arg=scipy.optimize.fixed_point(func=upper_bound, x0=1)
    upper_bound = min_evalue_map / m2_bar_fun(min(arg,1),data_x,data_y,n,inv_sigma,nu,mu,map,d)
    res = minimize_scalar(fun=w2, bounds=(lower_bound, min(upper_bound, min(arg,1))), method='bounded')

    return [res.fun, float(1/min_evalue_map)/n]





#
#
# def results_pool(d,start):
#     n = start
#     nu = 4
#     mu = [0] * d
#     sigma = np.identity(d)
#     for i in range(100000):
#         print("checking d,n: ", d, n)
#         if t_logistic_sample_size_check(n,d,nu,mu,sigma):
#             print(d,n, " successful")
#             return n
#         if d in [1,2]:
#             n = n + 10
#         if d==3:
#             n=n+50
#         if d==4:
#             n=n+100
#         if d==5:
#             n=n+500
#
#
# n=10
# start = time.time()
# for d in [1,2,3,4,5]:
#     r = []
#     n=results_pool(d,n)
#     r.append(n)
#     r=pd.DataFrame(r)
#     r.to_csv('lower_bounds_logistic'+str(d)+
#                      '.csv')





#
# df1=pd.read_csv(r'lower_bounds_logistic1.csv')
# df1.drop(columns='Unnamed: 0', inplace=True)
# df1=list(df1['0'])
# df2=pd.read_csv(r'lower_bounds_logistic2.csv')
# df2.drop(columns='Unnamed: 0', inplace=True)
# df2=list(df2['0'])
# df3=pd.read_csv(r'lower_bounds_logistic3.csv')
# df3.drop(columns='Unnamed: 0', inplace=True)
# df3=list(df3['0'])
# df4=pd.read_csv(r'lower_bounds_logistic4.csv')
# df4.drop(columns='Unnamed: 0', inplace=True)
# df4=list(df4['0'])
# df5=pd.read_csv(r'lower_bounds_logistic5.csv')
# df5.drop(columns='Unnamed: 0', inplace=True)
# df5=list(df5['0'])
# val=[0]+df1+df2+df3+df4+df5
#
#
#
#
#
#
#
#
# d=5 # change the d to take values 1,3,5,7,9
# nu=4
# mu=[0]*d
# sigma=np.identity(d)
# l=[0]*51
# first=math.log10(val[d])
# for i in range(51):
#     l[i]=int(10**(first+(6-first)*i/51))+10

# def res_pool(n):
#     sample=n+1
#     res=results_to_file(nu,mu,sigma,d,n+1)
    # for i in range(20000):
    #     try:
    #         res=results_to_file(nu,mu,sigma,d,n+i)
    #     except:
    #         print("Problem with n:", n+i)
    #     else:
    #         sample=n+1
    #         break
#     return [sample, res]
# start = time.time()
# res_pool(1000)
# end = time.time()
# print((end - start) / 60, " minutes")
# if __name__ == '__main__':
#     start = time.time()
#     with Pool(16) as p:
#         # r=p.map(res_pool, l)
#         r=p.map(res_pool, [1000])
#     df=pd.DataFrame(r)
#     df.to_csv('t_logistic_results_new'+str(d)+'.csv')
#     end = time.time()
#     print((end - start) / 60, " minutes")



# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12
#
# plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=MEDIUM_SIZE)
#
#
# df1=pd.read_csv(r'new_t_logistic_results1.csv')
# df1.drop(columns='Unnamed: 0', inplace=True)
# df3=pd.read_csv(r'new_t_logistic_results3.csv')
# df3.drop(columns='Unnamed: 0', inplace=True)
# df5=pd.read_csv(r'new_t_logistic_results5.csv')
# df5.drop(columns='Unnamed: 0', inplace=True)
# df7=pd.read_csv(r'new_t_logistic_results7.csv')
# df7.drop(columns='Unnamed: 0', inplace=True)
# df9=pd.read_csv(r'new_t_logistic_results9.csv')
# df9.drop(columns='Unnamed: 0', inplace=True)
#
#
# l=[0]*5
# c=0
# for j in [df1,df3,df5,df7,df9]:
#     r=[]
#     for i in range(51):
#         r.append(literal_eval(j['1'][i])[0])
#     l[c]=r
#     c=c+1
# # print(l[0])
#
# plt.plot(list(df1['0']), l[0],  label=r'd=1')
# plt.plot(list(df3['0']), l[1], '--', label=r'd=3')
# plt.plot(list(df5['0']),l[2],':', label=r'd=5')
# plt.plot(list(df7['0']), l[3],'-.', label=r'd=7')
# plt.plot(list(df9['0']), l[4],'.', label=r'd=9')
# # plt.plot(l, df['1'], '--', label='true mean')
# # plt.plot(l, df['2'], ':', label='true difference of means')
# plt.xlabel('number of data points')
# plt.yscale('log',base=10)
# plt.xscale('log',base=10)
# plt.legend(prop={'size':12})
# # plt.show()
# plt.savefig('w1_logistic_comparison.png',bbox_inches = 'tight', pad_inches = 0)
#
# l=[0]*5
# c=0
# for j in [df1,df3,df5,df7,df9]:
#     r=[]
#     for i in range(51):
#         r.append(literal_eval(j['1'][i])[2])
#     l[c]=r
#     c=c+1
# plt.clf()
# plt.plot(list(df1['0']), l[0],  label=r'd=1')
# plt.plot(list(df3['0']), l[1], '--', label=r'd=3')
# plt.plot(list(df5['0']),l[2],':', label=r'd=5')
# plt.plot(list(df7['0']), l[3],'-.', label=r'd=7')
# plt.plot(list(df9['0']), l[4],'.', label=r'd=9')
# plt.xlabel('number of data points')
# plt.yscale('log',base=10)
# plt.xscale('log',base=10)
# plt.legend(prop={'size':12})
# plt.savefig('w2_logistic_comparison.png',bbox_inches = 'tight', pad_inches = 0)


