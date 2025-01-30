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






