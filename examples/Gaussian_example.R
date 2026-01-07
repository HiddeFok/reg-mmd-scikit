library(renv)
renv::activate()
library(regMMD)

#simulate data
x = rnorm(50,0,1.5)
# estimate the mean and variance (assuming the data is Gaussian)
Est = mmd_est(x, model="Gaussian")
# print a summary
summary(Est)
# estimate the mean (assuming the data is Gaussian with known standard deviation =1.5)
Est2 = mmd_est(x, model="Gaussian.loc", par2=1.5)
# print a summary
summary(Est2)
# estimate the standard deviation (assuming the data is Gaussian with known mean = 0)
Est3 = mmd_est(x, model="Gaussian.scale", par1=0)
# print a summary
summary(Est3)
# test of the robustness
x[42] = 100
mean(x)
# estimate the mean and variance (assuming the data is Gaussian)
Est4 = mmd_est(x, model="Gaussian")
summary(Est4)