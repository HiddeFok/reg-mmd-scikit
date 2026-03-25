library(renv)
renv::activate()
library(regMMD)

# Simulate data
n <- 1000
p <- 4
beta <- 1:p
phi <- 1
X <- matrix(data=rnorm(n*p,0,1),nrow=n,ncol=p)
data <- X%*%beta+rnorm(n,0,phi)


##Example 2: Logisitic regression model
y<-data
y[data>5]<-1
y[data<=5]<-0
Est<-mmd_reg(y, X, model="logistic", intercept=FALSE, par1=c(0.5, 1.5, 1.1, 1.2))
print(Est$trajectory)
summary(Est)