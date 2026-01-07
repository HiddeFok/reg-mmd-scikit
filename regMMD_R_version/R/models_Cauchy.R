#' @importFrom stats rcauchy

# model: Cauchy par1 = loc

SGD.MMD.Cauchy = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par1)) {
    par = median(x)
  } else if ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  } else {
    par=par1
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize = 1
  norm.grad = epsilon
  res$par1 = par
  res$par2 = NULL
  res$stepsize=stepsize
  trajectory = par
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = rcauchy(n = n, location = par, scale = 1)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = (x.sampled-par)/(1+(x.sampled-par)^2)
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    trajectory = c(trajectory,par)
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = rcauchy(n = n, location = par, scale = 1)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = (x.sampled-par)/(1+(x.sampled-par)^2)
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = c(trajectory,par_mean)
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}
