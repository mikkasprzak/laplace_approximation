import numpy as np
import scipy.special
from scipy.optimize import minimize_scalar
"""
This file calulates the bounds from Theorems 3.2 and 3.3
"""

def third_constant(d,delta,n,i_mle,m1,m2):
    """
    This function calculates constants B_3 and C_3 from Theorems 3.2 and 3.3
    :param d: dimension
    :param delta: the value of delta
    :param n: sample size
    :param i_mle: the value of hat{J}_n(mle), i.e. minus Hessian of the log likelihood, evaluated at MLE, divided by n
    :param m1: value of hat{M}_1 from Assumption 2
    :param m2: value of M_2 from Assumption 1
    :return: the value of B_3 and C_3
    """
    matrix=i_mle + m2 * delta / 3*np.identity(d) # calculating hat{j}^p_n(mle, delta) defined at the top of section 2.5
    evalues=np.linalg.eigh(matrix)[0] #eigenvalues of matrix hat{j}^p_n(mle, delta)
    det=np.prod(evalues)
    inv_evalues=1/evalues
    trace = sum(inv_evalues)
    min_evalue= evalues[0] # minimal eigenvalue of matrix hat{j}^p_n(mle, delta)
    numerator = m1 * np.sqrt(det) # numerator of the constant
    denominator = (2 * np.pi )**(d/2) * (
                1 - np.exp(-(delta * np.sqrt(n) - np.sqrt(trace)) ** 2 / 2 * min_evalue)) # denominator of the costant
    return numerator / denominator

def sample_size_check(i_map, m2_bar_fun,mle,map,n):
    """
    This function checks whether the sample size is large enough for the Assumptions 4 and 5 to be satisfied
    :param i_map: the value of hat{J}_n(mle), i.e. minus Hessian of the log likelihood, evaluated at MLE, divided by n
    :param m2_bar_fun: a function that calculates bar(M)_2 from Assumption 3 and takes the value of bar{delta} as its argument
    :param mle: the value of the MLE
    :param map: the value of the MAP
    :param n: the sample size
    :return: TRUE if the sample size is large enough, FALSE otherwise
    """
    evalues_map=np.linalg.eigh(i_map)[0] # eigenvalues of i_map
    inv_evalues_map=1/evalues_map
    trace_map=sum(inv_evalues_map) # trace of the inverse of i_map
    min_evalue_map=evalues_map[0] # the minimal eigenvalue of i_map
    lower_bound_delta_bar=max(np.linalg.norm(mle-map), np.sqrt(trace_map/n)) # the lower bound on delta_bar from Assumption 4
    def upper_bound(x):
        """
        This function calculates an upper bound on bar{delta} from Assumption 5, having access to bar{M}_2
        :param x: the value of bar{delta} at which the value of bar{M}_2 should be derived
        :return: an upper bound on bar{delta} from Assumption 5
        """
        return min_evalue_map / m2_bar_fun(min(x,1))
    arg=scipy.optimize.fixed_point(func=upper_bound, x0=1) # finds the value of bar(delta) that is a fixed point of the function upper_bound subject to the constraint that bar(delta)<1
    upper_bound_delta_bar = min_evalue_map / m2_bar_fun(arg) # the upper bound on delta_bar from Assumption 5
    print("n:", n, "lower bound:", lower_bound_delta_bar, "upper_bound:", upper_bound_delta_bar)
    return lower_bound_delta_bar<upper_bound_delta_bar


def w1_bound(n,d,delta,delta_bar,i_map, i_mle, kappa_bar, integral, m2,m2_bar,m1_hat):
    """
    This function calculates the bound from Theorem 3.2
    :param n: sample size
    :param d: dimension
    :param delta: value of delta
    :param delta_bar: value of delta_bar
    :param i_map: the value of bar{J}_n(map), i.e. minus Hessian of the log posterior, evaluated at MAP, divided by n
    :param i_mle: the value of hat{J}_n(mle), i.e. minus Hessian of the log likelihood, evaluated at MLE, divided by n
    :param kappa_bar: value of bar{kappa} from Assumption 6
    :param integral: an upper bound on B_4 from Theorem 3.2
    :param m2: value of M_2 from Assumption 1
    :param m2_bar: value of bar{M}_2 from Assumption 3
    :param m1_hat: value of hat{M}_1 from Assumption 2
    :return: value of the bound from Theorem 3.2
    """
    evalues_imap = np.linalg.eigh(i_map)[0] #eigenvalues of bar{J}_n(map)
    inv_evalues_imap = 1 / evalues_imap
    trace_imap = sum(inv_evalues_imap)
    min_evalue_imap = evalues_imap[0] #minimal eigenvalue of bar{J}_n(map)
    def en1():
        """
        Calculating the value of B_1 from Theorem 3.2
        :return: the value of B_1 from Theorem 3.2
        """
        denominator=2*(min_evalue_imap/m2_bar-delta_bar)*np.sqrt(1-np.exp(-(delta_bar*np.sqrt(n)-np.sqrt(trace_imap))**2/2*min_evalue_imap))
        numerator=np.sqrt(3)*trace_imap
        return numerator/denominator
    def en2():
        """
        Calculating the value of B_2 from Theorem 3.3
        :return: the value of B_2 from Theorem 3.3
        """
        matrix1 = i_map - m2_bar*delta_bar/3*np.identity(d) # bar{J}_n^m(mle,delta_bar) as defined in Section 2.5
        matrix2 = i_map + m2_bar * delta_bar / 3 * np.identity(d) # bar{J}_n^p(mle,delta_bar) as defined in Section 2.5
        evalues1 = np.linalg.eigh(matrix1)[0] #eigenvalues of matrix1
        evalues2 = np.linalg.eigh(matrix2)[0] #eigenvalues of matrix2
        det1 = np.prod(evalues1) # determinant of matrix1
        det2 = np.prod(evalues2) # determinant of matrix2
        trace1 = sum(1/evalues1) # trace of inverse of matrix1
        trace2 = sum(1/evalues2) # trace of inverse of matrix2
        min_evalue2=evalues2[0] # minimum eigevalue of matrix2
        numerator = np.sqrt(det2)/np.sqrt(det1)*np.sqrt(trace1)
        denominator = 1-np.exp(-(delta_bar*np.sqrt(n)-np.sqrt(trace2))**2/2*min_evalue2)
        return numerator/denominator
    b1=en1()
    b2=en2()
    b3=third_constant(d,delta,n,i_mle,m1_hat,m2)

    term3=(delta_bar*np.sqrt(n)+np.sqrt(2*np.pi/min_evalue_imap)+b2)*np.exp(-(delta_bar*np.sqrt(n)-np.sqrt(trace_imap))**2/2*min_evalue_imap) # third summand in the bound of Theorem 3.2
    term2=b3*n**(d/2)*np.exp(-n*kappa_bar)*(b2+np.sqrt(n)*integral) # second summand in the bound on Theorem 3.2
    term1=b1/np.sqrt(n) # first summand in the bound of Theorem 3.2
    return term1+term2+term3



def w2_bound(n,d,delta, delta_bar, i_map, i_mle, kappa_bar, integral_sq, m2,m2_bar,m1_hat):
    """
    The function calculates the bound from Theorem 3.3
    :param n: sample size
    :param d: dimension
    :param delta: value of delta
    :param delta_bar: value of delta_bar
    :param i_map: the value of bar{J}_n(map), i.e. minus Hessian of the log posterior, evaluated at MAP, divided by n
    :param i_mle: the value of hat{J}_n(mle), i.e. minus Hessian of the log likelihood, evaluated at MLE, divided by n
    :param kappa_bar: value of bar{kappa} from Assumption 6
    :param integral_sq: an upper bound on C_4 from Theorem 3.3
    :param m2: value of M_2 from Assumption 1
    :param m2_bar: value of bar{M}_2 from Assumption 3
    :param m1_hat: value of hat{M}_1 from Assumption 2
    :return: value of the bound from Theorem 3.3
    """
    evalues_imap = np.linalg.eigh(i_map)[0]
    inv_evalues_imap = 1 / evalues_imap
    inverse_evalues_imap_sq=np.power(inv_evalues_imap,2)
    trace_imap = sum(inv_evalues_imap)
    trace_imap_sq=sum(inverse_evalues_imap_sq)
    min_evalue_imap = evalues_imap[0]
    def fn1():
        """
        Calculating the value of a constant equal to C_1 divided by sqrt(3) bar{m}_2 sqrt(trace_imap) and multiplied by min_evalue_map - delta_bar m2_bar
        :return: the value of C_1 from Theorem 3.3 divided by sqrt(3) bar{m}_2 sqrt(trace_imap) and multiplied by min_evalue_map - delta_bar m2_bar
        """
        denominator=1-np.exp(-(delta_bar*np.sqrt(n)-np.sqrt(trace_imap))**2/2*min_evalue_imap)
        numerator=trace_imap
        return numerator/denominator
    def fn2():
        """
        Calculating the value of C_2 from Theorem 3.3
        :return: the value of C_2 from Theorem 3.3
        """
        matrix1 = i_map - m2_bar * delta_bar / 3 * np.identity(d) # bar{J}_n^m(mle,delta_bar) as defined in Section 2.5
        matrix2 = i_map + m2_bar * delta_bar / 3 * np.identity(d) # bar{J}_n^p(mle,delta_bar) as defined in Section 2.5
        evalues1 = np.linalg.eigh(matrix1)[0] #eigenvalues of matrix1
        evalues2 = np.linalg.eigh(matrix2)[0] #eigenvalues of matrix2
        det1 = np.prod(evalues1) # determinant of matrix1
        det2 = np.prod(evalues2) # determinant of matrix2
        trace1 = sum(1 / evalues1) # trace of matrix1
        trace2 = sum(1 / evalues2) # trace of matrix2
        min_evalue2 = evalues2[0] # minimum eigevalue of matrix2
        numerator = np.sqrt(det2) / np.sqrt(det1) * np.sqrt(trace1)
        denominator = 1 - np.exp(-(delta_bar * np.sqrt(n) - np.sqrt(trace2)) ** 2 / 2 * min_evalue2)
        return numerator / denominator
    e1=fn1()
    e2=fn2()
    e3=third_constant(d,delta,n,i_mle,m1_hat,m2)

    term1=3/4*trace_imap/(min_evalue_imap/m2_bar-delta_bar)**2*e1/n+np.sqrt(3*trace_imap)/(min_evalue_imap/m2_bar-delta_bar)*e1/np.sqrt(n) #first and second summand in the bound of Theorem 3.3
    term2 = (delta_bar ** 2 * n + np.sqrt(2 * np.pi) / min_evalue_imap) * np.exp(-(delta_bar*np.sqrt(n)-np.sqrt(trace_imap))**2/2*min_evalue_imap) # fourth summand in the bound on Theorem 3.3
    term3=n**(d/2+1/2)*np.exp(-n*kappa_bar)*integral_sq*e3 # third summand in the bound of Theorem 3.3
    term4 = e2*(np.exp(-(delta_bar*np.sqrt(n)-np.sqrt(trace_imap))**2/2*min_evalue_imap)+e3*n**(d/2)*np.exp(-n*kappa_bar)) # last summand in the bound of Theorem 3.3
    return term1+term2+term3+term4