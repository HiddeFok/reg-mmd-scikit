#' @importFrom stats rnorm

# first series of function: SGD

# model: Gaussian-loc par1 = mean, fixed: par2 = sd

SGD.MMD.Gaussian.loc = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par1)) {
    par = median(x)
  } else if ((is.double(par1))&&(length(par1)==1)) {
    par = par1
  } else {
    res$error = c(res$error,"par1 must be numerical")
  }
  if (is.null(par2)) {
    res$error = c(res$error,"par2 missing")
  } else if ((is.double(par2)==FALSE)||(length(par2)!=1)) {
    res$error = c(res$error,"par2 must be numerical")
  } else if (par2<=0) {
    res$error = c(res$error,"par2 must be positive")
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize = par2
  norm.grad = epsilon
  res$par1 = par
  res$par2 = par2
  res$stepsize=stepsize
  trajectory = c(par)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = rnorm(n = n, mean = par, sd = par2)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = (x.sampled-par)/(par2^2)
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    trajectory = c(trajectory, par)
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = rnorm(n = n, mean = par, sd = par2)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = (x.sampled-par)/(par2^2)
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = c(trajectory, par_mean)
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}

# model: Gaussian-scale par2 = sd, fixed: par1 = mean

SGD.MMD.Gaussian.scale = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par1)) {
    res$error = c(res$error,"par1 missing")
  } else if  ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  }
  if (is.null(par2)) {
    par = (5/4)*median(abs(x-par1))
  } else if ((is.double(par2))&&(length(par2)==1)) {
    if (par2>0) {
      par = par2
    } else {
      res$error = c(res$error,"par2 must be positive")
    }
  } else {
    res$error = c(res$error,"par2 must be numerical")
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  par = max(par,1/n)
  if (stepsize=="auto") stepsize = par
  norm.grad = epsilon
  res$par1 = par1
  res$par2 = par
  res$stepsize=stepsize
  trajectory = c(par)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = rnorm(n = n, mean = par1, sd = par)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = (((x.sampled-par1)^2)/(par^2)-1)/par
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad+grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par = max(par,1/n)
    trajectory = c(trajectory, par)
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = rnorm(n = n, mean = par1, sd = par)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = (((x.sampled-par1)^2)/(par^2)-1)/par
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad+grad^2
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

# model: Gaussian par1 = mean par2 = sd

SGD.MMD.Gaussian = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  par = c(0,0)  
  if (is.null(par1)) {
    par[1] = median(x)
  } else if ((is.double(par1))&&(length(par1)==1)) {
    par[1] = par1
  } else {
    res$error = c(res$error,"par1 must be numerical")
  }
  if (is.null(par2)) {
    par[2] = (5/4)*median(abs(x-median(x)))
  } else if  ((is.double(par2)==FALSE)||(length(par2)!=1)) {
    res$error = c(res$error,"par2 must be numerical")
  } else if (par2<=0) {
    res$error = c(res$error,"par2 must be positive")
  } else {
    par[2]=par2
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize = par[2]
  norm.grad = epsilon
  res$par1 = par[1]
  res$par2 = par[2]
  res$stepsize=stepsize
  trajectory = matrix(data=par,nrow=2,ncol=1)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = rnorm(n = n, mean = par[1], sd = par[2])
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL1 = (x.sampled-par[1])/(par[2]^2)
    gradL2 = (((x.sampled-par[1])^2)/(par[2]^2)-1)/par[2]
    grad = 2*c( mean(gradL1%*%ker) , mean(gradL2%*%ker) )
    norm.grad = norm.grad + sum(grad^2)
    par = par-stepsize*grad/sqrt(norm.grad)
    par[2] = max(par[2],1/n^2)
    trajectory = cbind(trajectory,matrix(data=par,nrow=2,ncol=1))
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = rnorm(n = n, mean = par[1], sd = par[2])
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL1 = (x.sampled-par[1])/(par[2]^2)
    gradL2 = (((x.sampled-par[1])^2)/(par[2]^2)-1)/par[2]
    grad = 2*c( mean(gradL1%*%ker) , mean(gradL2%*%ker) )
    norm.grad = norm.grad + sum(grad^2)
    par = par-stepsize*grad/sqrt(norm.grad)
    par[2] = max(par[2],1/n)
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = cbind(trajectory,matrix(data=par_mean,nrow=2,ncol=1))
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}

# second series of function: GD

# model: Gaussian-loc par1 = mean, fixed: par2 = sd

GD.MMD.Gaussian.loc = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par1)) {
    par = median(x)
  } else if (is.double(par1)&&(length(par1)==1)) {
    par = par1
  } else {
    res$error = c(res$error,"par1 must be numerical")
  }
  if (is.null(par2)) {
    res$error = c(res$error,"par2 missing")
  } else if ((is.double(par2)==FALSE)||(length(par2)!=1)) {
    res$error = c(res$error,"par2 must be numerical")
  } else if (par2<=0) {
    res$error = c(res$error,"par2 must be positive")
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize=par2
  norm.grad = epsilon
  res$par1 = par
  res$par2 = par2
  res$stepsize=stepsize
  trajectory = c(par)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    diff = x-par
    grad = -4*mean(diff*exp(-(diff^2)/(2*(par2^2)+bdwth^2)))/sqrt(1+2*(par2^2)/(bdwth^2))
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    trajectory = c(trajectory,par)
  }
  
  # GD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    diff = x-par
    grad = -4*mean(diff*exp(-(diff^2)/(2*(par2^2)+bdwth^2)))/sqrt(1+2*(par2^2)/(bdwth^2))
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
