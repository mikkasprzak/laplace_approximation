import numpy as np
import math
import pandas as pd

"""
Generating logistic regression data
"""
d=11 # change d accordingly
num=10**6
np.random.seed(56)
df_x=np.random.multivariate_normal(np.zeros(d),np.identity(d),num)
def stable_sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig
df_y=[]
for i in range(num):
    data_point=df_x[i]
    np.random.seed(i)
    m=np.random.binomial(1,stable_sigmoid(np.inner([1]*d,data_point)))
    if m==0:
        m=-1
    df_y.append(m)
df_x=pd.DataFrame(df_x)
df_y=pd.DataFrame(df_y)

df_x.to_csv('normal_data'+str(d)+'.csv',index=False,header=False)
df_y.to_csv('logistic_data'+str(d)+'.csv',index=False,header=False)

"""
Generating exponential data
"""
num=10**6
np.random.seed(56)
df=np.random.exponential(10,num)
df=pd.DataFrame(df)
df.to_csv('exponential_data.csv',index=False)

df = pd.read_csv(r'exponential_data.csv')
print(list(df['0']))

"""
Generating Weibull data
"""
num=10**6
np.random.seed(56)
df=np.random.weibull(1/2,num)
df=pd.DataFrame(df)
df.to_csv('weibull_data.csv',index=False)


