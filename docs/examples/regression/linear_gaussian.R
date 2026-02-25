library(renv)
renv::activate()
library(regMMD)

#Simulate data
n<-1000
p<-4
beta<-1:p
phi<-1
X<-matrix(data=rnorm(n*p,0,1),nrow=n,ncol=p)
data<-1+X%*%beta+rnorm(n,0,phi)
##Example 1: Linear Gaussian model
y<-data
Est<-mmd_reg(y, X)
summary(Est)