#' @importFrom stats var rgamma

# model: gamma-shape par1 = a, fixed: par2 = b [exp(-bx)x^{a-1}]

SGD.MMD.gamma.shape = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par2)) {
    res$error = c(res$error,"par2 missing")
  } else if ((is.double(par2)==FALSE)||(length(par2)!=1)) {
    res$error = c(res$error,"par2 must be numerical")
  } else if (par2<=0) {
    res$error = c(res$error,"par2 must be positive")
  }
  if (is.null(par1)) {
    par = mean(x)*par2
  } else if ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  } else if (par1<0.5) {
    res$error = c(res$error,"par1 must be >=0.5")
  } else {
    par=par1
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad ## might want to find something safer when mean too large (contaminated)
  
  if (stepsize=="auto") stepsize = par
  norm.grad = epsilon
  res$par1 = par
  res$par2 = par2
  res$stepsize=stepsize
  trajectory = c(par)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = rgamma(n = n, shape = par, rate = par2)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = log(par2)-digamma(par)+log(x.sampled)
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par = max(par,0.5)
    trajectory = c(trajectory, par)
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = rgamma(n = n, shape = par, rate = par2)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = log(par2)-digamma(par)+log(x.sampled)
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par = max(par,0.5)
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = c(trajectory, par_mean)
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}

# model: gamma-rate par2 = b, fixed: par1 = a [exp(-bx)x^{a-1}]

SGD.MMD.gamma.rate = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par1)) {
    res$error = c(res$error,"par1 missing")
  } else if ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  } else if (par1<0.5) {
    res$error = c(res$error,"par1 must be >=0.5")
  }
  if (is.null(par2)) {
    par = par1/mean(x)
  } else if ((is.double(par2)==FALSE)||(length(par2)!=1)) {
    res$error = c(res$error,"par2 must be numerical")
  } else if (par2<=0) {
    res$error = c(res$error,"par2 must be positive")
  } else {
    par=par2
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad ## might want to find something safer when mean too large (contaminated)
  
  if (stepsize=="auto") stepsize = par1/mean(x)
  norm.grad = epsilon
  res$par1 = par1
  res$par2 = par
  res$stepsize=stepsize
  trajectory = c(par)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = rgamma(n = n, shape = par1, rate = par)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = par1/par-x.sampled
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par = max(par,1/n)
    trajectory = c(trajectory, par)
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = rgamma(n = n, shape = par1, rate = par)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = par1/par-x.sampled
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

# model: gamma par1 = a par2 = b [exp(-bx)x^{a-1}]

SGD.MMD.gamma = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  par = c(0,0)
  if (is.null(par1)) {
    par[1] = (mean(x)^2)/var(x)
  } else if ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  } else if (par1<0.5) {
    res$error = c(res$error,"par1 must be >=0.5")
  } else {
    par[1]=par1
  }
  if (is.null(par2)) {
    par[2] = mean(x)/var(x)
  } else if ((is.double(par2)==FALSE)||(length(par2)!=1)) {
    res$error = c(res$error,"par2 must be numerical")
  } else if (par2<=0) {
    res$error = c(res$error,"par2 must be positive")
  } else {
    par[2]=par2
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize = par
  norm.grad = epsilon
  res$par1 = par[1]
  res$par2 = par[2]
  res$stepsize=stepsize
  trajectory = matrix(data=par,nrow=2,ncol=1)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = rgamma(n = n, shape = par[1], rate = par[2])
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL1 = log(par[2])-digamma(par[1])+log(x.sampled)
    gradL2 = par[1]/par[2]-x.sampled
    grad = 2*c( mean(gradL1%*%ker) , mean(gradL2%*%ker) )
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par[1] = max(par[1],0.5)
    par[2] = max(par[2],1/n)
    trajectory = cbind(trajectory,matrix(data=par,nrow=2,ncol=1))
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = rgamma(n = n, shape = par[1], rate = par[2])
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL1 = log(par[2])-digamma(par[1])+log(x.sampled)
    gradL2 = par[1]/par[2]-x.sampled
    grad = 2*c( mean(gradL1%*%ker) , mean(gradL2%*%ker) )
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par[1] = max(par[1],0.5)
    par[2] = max(par[2],1/n)
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = cbind(trajectory,matrix(data=par_mean,nrow=2,ncol=1))
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}