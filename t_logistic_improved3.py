from main_multivariate import w1_bound
from main_multivariate import w2_bound
import scipy.special
import numpy as np
from main_multivariate import third_constant
import scipy.optimize
from scipy.optimize import minimize_scalar
import pandas as pd
import matplotlib.pyplot as plt
from main_multivariate import sample_size_check
from ast import literal_eval
from multiprocessing import Pool
import math
import stan as stan
import time
from scipy.optimize import root_scalar

def mle_calculator(data_x,data_y,n,d):
    def minus_log_likelihood(theta):
        res=0
        for i in range(n):
            res=res+np.log(1+np.exp(-np.inner(theta,data_x[i])*data_y[i]))
        return res
    def jac(theta):
        res=[0]*d
        for i in range(d):
            for j in range(n):
                res[i]=res[i]-data_y[j]*data_x[j][i]/(1+np.exp(np.inner(theta,data_x[j])*data_y[j]))
        return np.array(res)

    x0=[1]*d
    r=scipy.optimize.minimize(fun=minus_log_likelihood, x0=x0, jac=jac)
    return r.x

def map_calculator(data_x,data_y,nu,mu,inv_sigma,n,d,mle):
    def minus_log_posterior(theta):
        res=(nu+d)/2*np.log(1+1/nu*np.dot(np.transpose(np.array(theta)-np.array(mu)),np.dot(inv_sigma,np.array(theta)-np.array(mu))))
        for i in range(n):
            res=res+np.log(1+np.exp(-np.inner(theta,data_x[i])*data_y[i]))
        return res
    x0=mle
    r=scipy.optimize.minimize(fun=minus_log_posterior, x0=x0)
    return r.x


def delta_optimizer(m1_hat_fun,m2_fun,i_mle,d,n):
    evalues=np.linalg.eigh(i_mle)[0]
    #print(evalues)
    inv_evalues=1/evalues
    def en3(delta):
        m2 = m2_fun(delta)
        m1 = m1_hat_fun(delta)
        return third_constant(d,delta,n,i_mle,m1,m2)
    res = minimize_scalar(fun=en3, bounds=(np.sqrt(sum(inv_evalues)/n), np.sqrt(sum(inv_evalues)/n)+1), method='bounded')
    delta=res.x
    m1_hat=m1_hat_fun(res.x)
    return [delta,m1_hat]


def t_logistic_parameters(data_x,data_y,nu,mu,sigma,n,d):
    mle=mle_calculator(data_x,data_y,n,d)
    # print("MLE:", mle)
    map=map_calculator(data_x,data_y,nu,mu,sigma,n,d,mle)
    # print("MAP: ", map)
    inv_sigma = np.linalg.inv(sigma)
    sigma_evalues = np.linalg.eigh(sigma)[0]
    """
    I map and I mle
    """
    def i_likelihood(theta):
        res=np.zeros((d,d))
        for i in range(n):
            res=res+np.outer(data_x[i],data_x[i])*np.exp(np.inner(data_x[i],theta)*data_y[i])/(1+np.exp(np.inner(data_x[i],theta)*data_y[i]))**2
        res=res/n
        return res
    def i_posterior(theta):
        term1=0
        for j in range(d):
            for i in range(d):
                term1=term1+(theta[i]-mu[i])*(theta[j]-mu[j])*inv_sigma[i,j]
        term1=term1/nu+1
        term2=[0]*d
        for j in range(d):
            term2[j]=inv_sigma[j,j]*(theta[j]-mu[j])
        term3=[0]*d
        for j in range(d):
            for l in range(d):
                term3[j]=term3[j]+inv_sigma[l,j]*(theta[l]-mu[l])
        res=np.zeros((d,d))
        for i in range(d):
            for j in range(d):
                res[i,j]=-1/nu**2 * (term2[j]+term3[j])*(term2[i]+term3[i])/term1**2
                if i==j:
                    res[i,j]=res[i,j]+2/nu*inv_sigma[i,j]/term1
                else:
                    res[i,j]=res[i,j]+1/nu*inv_sigma[i,j]/term1
                res[i,j]=res[i,j]*(nu+d)/2
        return res/n + i_likelihood(theta)
    i_mle=i_likelihood(mle)
    i_map=i_posterior(map)
    i_mle_evalues = np.linalg.eigh(i_mle)[0]
    i_map_evalues = np.linalg.eigh(i_map)[0]
    min_evalue_mle = i_mle_evalues[0]
    min_evalue_map = i_map_evalues[0]
    i_map_inv_trace = sum(1/i_map_evalues)
    """
    The delta parameter
    """
    def m1_hat_fun(delta):
        term1=scipy.special.gamma(nu/2)*nu**(d/2)*np.pi**(d/2)*np.prod(sigma_evalues)**(1/2)/scipy.special.gamma((nu+d)/2)
        term2=(1+1/nu/sigma_evalues[0]*(2*delta**2+2*np.linalg.norm(mu-map)))**((nu+d)/2)
        return term1*term2

    def bound_fun(theta):
        res = 0
        for i in range(d):
            exponent=np.inner(theta, data_x[i]) * data_y[i]
            e = np.exp(exponent)
            res = res + np.linalg.norm(data_x[i]) ** 3 / n * e * abs(e - 1) / (1 + e) ** 3
            # print("Bound fun:",res)
            try:
                if res<0:
                    print("Alert")
            except:
                print("Bound fun:", res)
        return res



    def m2_fun(delta):
        def fun(x):
            return -bound_fun(x)
        l = []
        for i in range(d):
            l = l + [(mle[i] - delta, mle[i] + delta)]
        l = tuple(l)
        r = scipy.optimize.shgo(func=fun, bounds=l)
        return -r.fun
    delta_opt = delta_optimizer(m1_hat_fun,m2_fun,i_mle,d,n)
    delta = delta_opt[0]
    m1_hat = delta_opt[1]
    m2=m2_fun(delta)

    """
    The delta bar parameter
    """
    def third_der_like(delta_bar):
        def fun(x):
            return -bound_fun(x)
        l = []
        for i in range(d):
            l = l + [(map[i] - delta_bar, map[i] + delta_bar)]
        l = tuple(l)
        r=scipy.optimize.shgo(func=fun, bounds=l)
        # print("done with delta bar:", delta_bar)
        return -r.fun
    def m2_bar_fun(delta_bar):
        matrix=inv_sigma+np.diag(np.diag(inv_sigma))
        op_norm=np.linalg.norm(matrix,2)
        term1=3*(nu+d)/nu**2*op_norm**2*(delta_bar+np.linalg.norm(map-mu))**2
        term2=2*(nu+d)/nu**3*op_norm**3*(delta_bar+np.linalg.norm(map-mu))**3
        res=(term1+term2)/n+third_der_like(delta_bar)
        # print("Delta bar:", delta_bar, "M2_bar:", res)
        return res
    """
    The kappa bar parameter
    """
    def kappa_bar_fun(delta_bar):
        term1=third_der_like(delta_bar)*(delta_bar-np.linalg.norm(mle-map))**3
        term2=min_evalue_mle*(delta_bar-np.linalg.norm(mle-map))**2
        return term2-term1
    """
    Prior integral bounds
    """
    integral=np.sqrt(nu/(nu-2)*sum(sigma_evalues))+np.linalg.norm(mle)
    integral_sq = 2*nu / (nu - 2) * sum(sigma_evalues) + 2*np.linalg.norm(mle)**2
    return {"size": n, "d": d, "delta": delta, "m1_hat": m1_hat, "m2": m2,
            "m2_bar": m2_bar_fun, "i_mle": i_mle, "i_map": i_map, "mle": mle, "map": map, "kappa_bar": kappa_bar_fun,
            "integral": integral, "integral_sq": integral_sq, "min_evalue_mle":min_evalue_mle,"min_evalue_map":min_evalue_map, "i_map_inv_trace":i_map_inv_trace}


def bound_w1_t_logistic(data_x,data_y,nu,mu,sigma,n,d):
    params = t_logistic_parameters(data_x,data_y,nu,mu,sigma,n,d)
    i_map = params.get("i_map")
    i_mle = params.get("i_mle")
    integral = params.get("integral")
    m2 = params.get("m2")
    m1_hat = params.get("m1_hat")
    delta = params.get("delta")
    kappa_bar_fun=params.get("kappa_bar")
    m2_bar_fun=params.get("m2_bar")
    map=params.get("map")
    mle=params.get("mle")
    min_evalue_mle = params.get("min_evalue_mle")
    min_evalue_map = params.get("min_evalue_map")
    i_map_inv_trace= params.get("i_map_inv_trace")
    dif=np.linalg.norm(map-mle)
    inv_sigma = np.linalg.inv(sigma)
    matrix = inv_sigma + np.diag(np.diag(inv_sigma))
    def w1(delta_bar):
        kappa_bar = kappa_bar_fun(delta_bar)
        m2_bar = m2_bar_fun(delta_bar)
        return w1_bound(n,d,delta,delta_bar,i_map, i_mle, kappa_bar, integral, m2,m2_bar,m1_hat)/np.sqrt(n)
    lower_bound=max(dif, np.sqrt(i_map_inv_trace/n))
    if lower_bound>1:
        print("too small n")
        raise
    def upper_bound(x):
        return min_evalue_map / m2_bar_fun(min(x,1))
    arg=scipy.optimize.fixed_point(func=upper_bound, x0=1)
    upper_bound = min_evalue_map / m2_bar_fun(min(arg,1)) # looking for an upper bound for delta_bar
    res = minimize_scalar(fun=w1, bounds=(lower_bound, min(upper_bound, min(arg,1))), method='bounded')
    return [res.fun, map]


def bound_w2_t_logistic(data_x,data_y,nu,mu,sigma,n,d):
    params = t_logistic_parameters(data_x,data_y,nu,mu,sigma,n,d)
    i_map = params.get("i_map")
    i_mle = params.get("i_mle")
    integral_sq = params.get("integral_sq")
    m2 = params.get("m2")
    m1_hat = params.get("m1_hat")
    delta = params.get("delta")
    kappa_bar_fun=params.get("kappa_bar")
    m2_bar_fun=params.get("m2_bar")
    map=params.get("map")
    # print("Map norm:",np.linalg.norm(map))
    mle=params.get("mle")
    # print("MAP-MLE: ", np.linalg.norm(map - mle))
    min_evalue_mle = params.get("min_evalue_mle")
    min_evalue_map = params.get("min_evalue_map")
    i_map_inv_trace= params.get("i_map_inv_trace")
    dif=np.linalg.norm(map-mle)
    inv_sigma = np.linalg.inv(sigma)
    matrix = inv_sigma + np.diag(np.diag(inv_sigma))
    # op_norm = np.linalg.norm(matrix, 2)
    # term1 = 3 * (nu + d) / nu ** 2 * op_norm ** 2 * (1 + np.linalg.norm(map - mu)) ** 2
    # term2 = 2 * (nu + d) / nu ** 3 * op_norm ** 3 * (1 + np.linalg.norm(map - mu)) ** 3
    def w2(delta_bar):
        kappa_bar = kappa_bar_fun(delta_bar)
        m2_bar = m2_bar_fun(delta_bar)
        return w2_bound(n,d,delta, delta_bar, i_map, i_mle, kappa_bar, integral_sq, m2,m2_bar,m1_hat)/n
    lower_bound=max(dif, np.sqrt(i_map_inv_trace/n))
    if lower_bound>1:
        print("too small n")
        raise
    def upper_bound(x):
        return min_evalue_map / m2_bar_fun(min(x,1))
    arg=scipy.optimize.fixed_point(func=upper_bound, x0=1)
    upper_bound = min_evalue_map / m2_bar_fun(min(arg,1))
    res = minimize_scalar(fun=w2, bounds=(lower_bound, min(upper_bound, min(arg,1))), method='bounded')

    # delta_bar = res.x
    # print("Delta_bar: ", delta_bar)
    return [res.fun, float(1/min_evalue_map)/n]




def results(nu,mu,sigma,n,d,df_x,df_y):
    data_x = df_x[:n]
    data_y = df_y[:n]
    bound_w1=bound_w1_t_logistic(data_x,data_y,nu,mu,sigma,n,d)
    return bound_w1


def results_w2(nu,mu,sigma,n,d,df_x,df_y):
    data_x = df_x[:n]
    data_y = df_y[:n]
    bound_and_var=bound_w2_t_logistic(data_x, data_y, nu, mu, sigma, n, d)
    bound_w2=bound_and_var[0]
             # +bound_w1**2
    var=bound_and_var[1]/n
    return [bound_w2,var]


def results_to_file(nu , mu , sigma, d, n):
    df_x=pd.read_csv(r'normal_data'+str(d)+'.csv')
    df_y = pd.read_csv(r'logistic_data' + str(d) + '.csv')
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
    print(d,n, " done")
    return [boundw1, map_norm, boundw2, var]

def t_logistic_sample_size_check(n,d,nu,mu,sigma):
    df_x = pd.read_csv(r'normal_data' + str(d) + '.csv')
    df_y = pd.read_csv(r'logistic_data' + str(d) + '.csv')
    data_x = df_x.values[:n]
    data_y1 = df_y.values[:n]
    data_y = []
    for i in data_y1:
        data_y.append(i[0])
    params = t_logistic_parameters(data_x,data_y,nu,mu,sigma,n,d)
    i_map = params.get("i_map")
    mle = params.get("mle")
    map = params.get("map")
    m2_bar_fun = params.get("m2_bar")
    return sample_size_check(i_map, m2_bar_fun,mle,map,n)

'''
Making the first plot:
'''
d=5
nu=4
mu=[0]*d
sigma=np.identity(d)

def res_pool(n):
    return results_to_file(nu,mu,sigma,d,n)

l=[0]*51
for i in range(51):
    l[i]=int(10**(3+2*i/51))

if __name__ == '__main__':
    start = time.time()
    with Pool(7) as p:
        r=p.map(res_pool, l)
    df=pd.DataFrame(r)
    df.to_csv('t_logistic_results'+str(d)+'.csv')
    end = time.time()
    print((end - start) / 60, " minutes")
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
df=pd.read_csv(r't_logistic_results'+str(d)+'.csv')
df.drop(columns='Unnamed: 0', inplace=True)

plt.plot(l, df['0'], label=r'our bound on the 1-Wasserstein distance')
plt.plot(l, df['1'], '--', label='norm of the approximating mean')
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':12})
plt.savefig('w1_logistic.png',bbox_inches = 'tight', pad_inches = 0)

plt.clf()


plt.plot(l, df['2'], label='our bound on the 2-norm of the \n difference of covariances')
plt.plot(l, df['3'], '--', label='2-norm of the approximating covariance')
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':12})
plt.savefig('w2_logistic.png',bbox_inches = 'tight', pad_inches = 0)

'''
Establishing the minimal n and making the second plot:
'''

def results_pool(d,start):
    n = start
    nu = 4
    mu = [0] * d
    sigma = np.identity(d)
    for i in range(100000):
        print("checking d,n: ", d, n)
        if t_logistic_sample_size_check(n,d,nu,mu,sigma):
            print(d,n, " successful")
            return n
        n = n + 2

r=[]
n=1
start = time.time()
for d in [1,2,3,4,5]:
    n=results_pool(d,n)
    r.append(n)
r=pd.DataFrame(r)
r.to_csv('lower_bounds_logistic'
                     '.csv')
end = time.time()
print((end - start) / 60, " minutes")

def results_pool(d,start):
    n = start
    nu = 4
    mu = [0] * d
    sigma = np.identity(d)
    for i in range(100000):
        print("checking d,n: ", d, n)
        if t_logistic_sample_size_check(n,d,nu,mu,sigma):
            print(d,n, " successful")
            return n
        n = n + 10

r=[]
n=630
start = time.time()
for d in [6,7]:
    n=results_pool(d,n)
    r.append(n)
r=pd.DataFrame(r)
r.to_csv('lower_bounds_logistic1'
                     '.csv')
end = time.time()


def results_pool(d,start):
    n = start
    nu = 4
    mu = [0] * d
    sigma = np.identity(d)
    for i in range(100000):
        print("checking d,n: ", d, n)
        if t_logistic_sample_size_check(n,d,nu,mu,sigma):
            print(d,n, " successful")
            return n
        n = n + 10

r=[]
n=1600
start = time.time()
for d in [8,9]:
    n=results_pool(d,n)
    r.append(n)
r=pd.DataFrame(r)
r.to_csv('lower_bounds_logistic2'
                     '.csv')
end = time.time()

def results_pool(d,start):
    n = start
    nu = 4
    mu = [0] * d
    sigma = np.identity(d)
    for i in range(100000):
        print("checking d,n: ", d, n)
        if t_logistic_sample_size_check(n,d,nu,mu,sigma):
            print(d,n, " successful")
            return n
        n = n + 50

r=[]
n=2800
start = time.time()
for d in [10,11]:
    n=results_pool(d,n)
    r.append(n)
r=pd.DataFrame(r)
r.to_csv('lower_bounds_logistic3'
                     '.csv')
end = time.time()


df=pd.read_csv(r'lower_bounds_logistic.csv')
df.drop(columns='Unnamed: 0', inplace=True)
df=list(df['0'])
df1=pd.read_csv(r'lower_bounds_logistic1.csv')
df1.drop(columns='Unnamed: 0', inplace=True)
df1=list(df1['0'])
df2=pd.read_csv(r'lower_bounds_logistic2.csv')
df2.drop(columns='Unnamed: 0', inplace=True)
df2=list(df2['0'])
df3=pd.read_csv(r'lower_bounds_logistic3.csv')
df3.drop(columns='Unnamed: 0', inplace=True)
df3=list(df3['0'])
val=[0]+df+df1+df2+df3

l=range(12)
plt.plot(l, val, label=r'Minimum number of points required to compute our bound')
plt.ylabel('number of data points')
plt.xlabel('dimension')
# plt.legend(prop={'size':10})
plt.savefig('min_values_logistic.png',bbox_inches = 'tight', pad_inches = 0)


'''
Making the third plot:
'''
df=pd.read_csv(r'lower_bounds_logistic.csv')
df.drop(columns='Unnamed: 0', inplace=True)
df=list(df['0'])
df1=pd.read_csv(r'lower_bounds_logistic1.csv')
df1.drop(columns='Unnamed: 0', inplace=True)
df1=list(df1['0'])
df2=pd.read_csv(r'lower_bounds_logistic2.csv')
df2.drop(columns='Unnamed: 0', inplace=True)
df2=list(df2['0'])
df3=pd.read_csv(r'lower_bounds_logistic3.csv')
df3.drop(columns='Unnamed: 0', inplace=True)
df3=list(df3['0'])
val=[0]+df+df1+df2+df3



d=1 # change the d to take values 1,3,5,7,9
nu=4
mu=[0]*d
sigma=np.identity(d)
l=[0]*51
first=math.log10(val[d])
for i in range(51):
    l[i]=int(10**(first+(5-first)*i/51))+10

def res_pool(n):
    for i in range(20000):
        try:
            res=results_to_file(nu,mu,sigma,d,n+i)
        except:
            print("Problem with n:", n+i)
        else:
            sample=n+1
            break
    return [sample, res]

if __name__ == '__main__':
    start = time.time()
    with Pool(7) as p:
        r=p.map(res_pool, l)
    df=pd.DataFrame(r)
    df.to_csv('new_t_logistic_results'+str(d)+'.csv')
    end = time.time()
    print((end - start) / 60, " minutes")



SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)


df1=pd.read_csv(r'new_t_logistic_results1.csv')
df1.drop(columns='Unnamed: 0', inplace=True)
df3=pd.read_csv(r'new_t_logistic_results3.csv')
df3.drop(columns='Unnamed: 0', inplace=True)
df5=pd.read_csv(r'new_t_logistic_results5.csv')
df5.drop(columns='Unnamed: 0', inplace=True)
df7=pd.read_csv(r'new_t_logistic_results7.csv')
df7.drop(columns='Unnamed: 0', inplace=True)
df9=pd.read_csv(r'new_t_logistic_results9.csv')
df9.drop(columns='Unnamed: 0', inplace=True)


l=[0]*5
c=0
for j in [df1,df3,df5,df7,df9]:
    r=[]
    for i in range(51):
        r.append(literal_eval(j['1'][i])[0])
    l[c]=r
    c=c+1
# print(l[0])

plt.plot(list(df1['0']), l[0],  label=r'd=1')
plt.plot(list(df3['0']), l[1], '--', label=r'd=3')
plt.plot(list(df5['0']),l[2],':', label=r'd=5')
plt.plot(list(df7['0']), l[3],'-.', label=r'd=7')
plt.plot(list(df9['0']), l[4],'.', label=r'd=9')
# plt.plot(l, df['1'], '--', label='true mean')
# plt.plot(l, df['2'], ':', label='true difference of means')
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':12})
# plt.show()
plt.savefig('w1_logistic_comparison.png',bbox_inches = 'tight', pad_inches = 0)

l=[0]*5
c=0
for j in [df1,df3,df5,df7,df9]:
    r=[]
    for i in range(51):
        r.append(literal_eval(j['1'][i])[2])
    l[c]=r
    c=c+1
plt.clf()
plt.plot(list(df1['0']), l[0],  label=r'd=1')
plt.plot(list(df3['0']), l[1], '--', label=r'd=3')
plt.plot(list(df5['0']),l[2],':', label=r'd=5')
plt.plot(list(df7['0']), l[3],'-.', label=r'd=7')
plt.plot(list(df9['0']), l[4],'.', label=r'd=9')
plt.xlabel('number of data points')
plt.yscale('log',base=10)
plt.xscale('log',base=10)
plt.legend(prop={'size':12})
plt.savefig('w2_logistic_comparison.png',bbox_inches = 'tight', pad_inches = 0)


