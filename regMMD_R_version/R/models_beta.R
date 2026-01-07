#' @importFrom stats var

# model: beta par1 = a par2 = b [x^(a-1)(1-x)^(b-1)]

SGD.MMD.beta = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  par = c(0,0)
  ex = mean(x)
  va = var(x)
  scaling = ex*(1-ex)/va-1
  if (is.null(par1)) {
    par[1] = ex*scaling
  } else if ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  } else if (par1<=0) {
    res$error = c(res$error,"par1 must be positive")
  } else {
    par[1]=par1
  }
  if (is.null(par2)) {
    par[2] = (1-ex)*scaling
  } else if ((is.double(par2)==FALSE)||(length(par2)!=1)) {
    res$error = c(res$error,"par2 must be numerical")
  } else if (par2<=0) {
    res$error = c(res$error,"par2 must be positive")
  } else {
    par[2]=par2
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize = 1/scaling
  norm.grad = epsilon
  res$par1 = par[1]
  res$par2 = par[2]
  res$stepsize=stepsize
  trajectory = matrix(data=par,nrow=2,ncol=1)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = rbeta(n = n, shape1 = par[1], shape2 = par[2])
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL1 = digamma(par[1]+par[2])-digamma(par[1])+log(x.sampled)
    gradL2 = digamma(par[1]+par[2])-digamma(par[2])+log(1-x.sampled)
    grad = 2*c( mean(gradL1%*%ker) , mean(gradL2%*%ker) )
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par[1] = max(par[1],1/n)
    par[2] = max(par[2],1/n)
    trajectory = cbind(trajectory,matrix(data=par,nrow=2,ncol=1))
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = rbeta(n = n, shape1 = par[1], shape2 = par[2])
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL1 = digamma(par[1]+par[2])-digamma(par[1])+log(x.sampled)
    gradL2 = digamma(par[1]+par[2])-digamma(par[2])+log(1-x.sampled)
    grad = 2*c( mean(gradL1%*%ker) , mean(gradL2%*%ker) )
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par[1] = max(par[1],1/n)
    par[2] = max(par[2],1/n)
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = cbind(trajectory,matrix(data=par_mean,nrow=2,ncol=1))
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}
