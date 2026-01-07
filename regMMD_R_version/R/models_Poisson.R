#' @importFrom stats rpois

# model: Poisson par1 = mean

SGD.MMD.Poisson = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  if (bdwth<1) bdwth=1
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par1)) {
    med = median(x)
    if (med==0) par = 1 else par = med
  } else if ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  } else if (par1<=0) {
    res$error = c(res$error,"par1 must be positive")
  } else {
    par = par1
  }
  if (is.null(res$error)==FALSE) return(res)

  if (is.integer(x)==FALSE) res$error = c(res$error,"Attention: you used the Poisson model on non-integer observations. If this is intentional, you can ignore this message. If your observations are integers but stored as numerical values, you can use x=as.integer(x) to avoid this warning.")
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize = 1
  norm.grad = epsilon
  res$par1 = par
  res$par2 = NULL
  res$stepsize=stepsize
  trajectory = c(par)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = rpois(n = n, lambda = par)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = x.sampled/par-1
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par = max(par,1/n)
    trajectory = c(trajectory, par)
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = rpois(n = n, lambda = par)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = x.sampled/par-1
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
