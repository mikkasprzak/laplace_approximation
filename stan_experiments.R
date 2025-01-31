set.seed(56)

# Simulating some data
y     = read.csv("logistic_data_large5.csv", header=FALSE)
x     = read.csv("normal_data_small5.csv", header=FALSE)
for (i in 1:10**6)
{
  if (y[i,1]<0) {
    y[i,1]<-0
  }
print(i)
}
write.csv(y,"logistic_data5_transformed.csv")
z<-read.csv("logistic_data5_transformed.csv")

l<-read.csv("logistic_indices5_large.csv")
l1=list()

for (i in 1:ncol(l)){
  l1<-l[,i]}
# print(l1[5])

library(rstan)
library(parallel)


posterior_means<-function(sample_size) {
print(paste(as.character(sample_size), " started"))
stan_program = paste("
data {
  matrix[5,",as.character(sample_size),"] x;
  array[",as.character(sample_size),"] int y;
  }
parameters {
  vector[5] beta;
}
model {
  y ~ bernoulli_logit(to_row_vector(x'*beta));
  beta ~ multi_student_t(4, [1, 1,1,1,1], diag_matrix([1, 1,1,1,1]'));
}
")



var1<-z[1:sample_size,2]
var2<-x[1:sample_size,]

# Running stan code

fit = stan(model_code=stan_program,data=list(x=t(var2),y=var1), iter=2000, chains=4, cores=1)
params = extract(fit)
means<-colMeans(params$beta)
covariances<-cov(params$beta)
print(paste(as.character(sample_size), " done"))
s<-summary(fit)
return(list(means,covariances))
}

results<-lapply(l1, posterior_means)
write.csv(results,"mean_covs_large5.csv",col.names = FALSE, row.names=FALSE)


means_error_bounds<-function(sample_size) {
  print(paste(as.character(sample_size), " started"))
  stan_program = paste("
data {
  matrix[5,",as.character(sample_size),"] x;
  array[",as.character(sample_size),"] int y;
  }
parameters {
  vector[5] beta;
}
model {
  y ~ bernoulli_logit(to_row_vector(x'*beta));
  beta ~ multi_student_t(4, [1, 1,1,1,1], diag_matrix([1, 1,1,1,1]'));
}
")
  
  
  
  var1<-z[1:sample_size,2]
  var2<-x[1:sample_size,]
  
  # Running stan code
  
  fit = stan(model_code=stan_program,data=list(x=t(var2),y=var1), iter=2000, chains=4, cores=1)
  s = summary(fit)
  mean_standard_error=s$summary[,'se_mean']
  return(mean_standard_error)
}

results<-lapply(l1, means_error_bounds)
write.csv(results,"mean_error5.csv",col.names = FALSE, row.names=FALSE)
