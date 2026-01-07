# model: Pareto par1 = a [a/(x^(a+1)) for x>1]

SGD.MMD.Pareto = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par1)) {
    par = log(2)/log(median(x))
  } else if ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  } else {
    par=par1
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize = par
  norm.grad = epsilon
  res$par1 = par
  res$par2 = NULL
  res$stepsize=stepsize
  trajectory = c(par)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = 1/(runif(n=n, min=0, max=1)^(1/par))
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = 1/par-log(x.sampled)
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par = max(par,1/n)
    trajectory = c(trajectory, par)
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = 1/(runif(n=n, min=0, max=1)^(1/par))
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = 1/par-log(x.sampled)
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par = max(par,1/n)
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = c(trajectory, par_mean)
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}
