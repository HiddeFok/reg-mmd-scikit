#' @importFrom stats runif

# first series of function: SGD

# model: continuous.uniform.loc par1 = center, fixed: par2 = length [center-length/2,center+length/2]

SGD.MMD.continuous.uniform.loc = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
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
  if (is.null(par2)) {
    res$error = c(res$error,"par2 missing")
  } else if ((is.double(par2)==FALSE)||(length(par2)!=1)) {
    res$error = c(res$error,"par2 must be numerical")
  } else if (par2<=0) {
    res$error = c(res$error,"par2 must be positive")
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize = par2/sqrt(12)
  norm.grad = epsilon
  res$par1 = par
  res$par2 = par2
  res$stepsize=stepsize
  trajectory = par
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = runif(n = n, min = par-par2/2, max = par+par2/2)
    ker = K1d.diff(x.sampled,x,kernel=kernel,bdwth=bdwth)
    grad = -2*mean(ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    trajectory = c(trajectory,par)
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = runif(n = n, min = par-par2/2, max = par+par2/2)
    ker = K1d.diff(x.sampled,x,kernel=kernel,bdwth=bdwth)
    grad = -2*mean(ker)
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

# model: continuous.uniform.upper par2 = upper, fixed par1, uniform on [par1, upper]

SGD.MMD.continuous.uniform.upper = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par1)) {
    res$error = c(res$error,"par1 missing")
  } else if ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  }
  if (is.null(par2)) {
    par = 2*median(x)-par1
  } else if ((is.double(par2)==FALSE)||(length(par2)!=1)) {
    res$error = c(res$error,"par2 must be numerical")
  } else if (par2<=par1) {
    res$error = c(res$error,"par2 must be > par1")
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize=(par-par1)/sqrt(12)
  norm.grad = epsilon
  res$par1 = par1
  res$par2 = par
  res$stepsize=stepsize
  trajectory = c(par)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = runif(n = n, min = 0, max = 1)
    x.sampled.scaled = par1 + (par-par1)*x.sampled
    ker = K1d.diff(x.sampled.scaled,x.sampled.scaled,kernel=kernel,bdwth=bdwth)/(n-1)- K1d.diff(x.sampled.scaled,x,kernel=kernel,bdwth=bdwth)/n
    grad = 2*mean(x.sampled%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    trajectory = c(trajectory, par)
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = runif(n = n, min = 0, max = 1)
    x.sampled.scaled = par1 + (par-par1)*x.sampled
    ker = K1d.diff(x.sampled.scaled,x.sampled.scaled,kernel=kernel,bdwth=bdwth)/(n-1)- K1d.diff(x.sampled.scaled,x,kernel=kernel,bdwth=bdwth)/n
    grad = 2*mean(x.sampled%*%ker)
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

# second series of function: GD

# model: continuous.uniform.lower.upper par1 = lower par2 = upper, uniform on [lower, upper]

SGD.MMD.continuous.uniform.lower.upper = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  med = median(x)
  dev = 2*median(abs(x-med))
  par = c(0,0) 
  if (is.null(par1)) {
    par[1] = med-dev
  } else if ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  } else {
    par[1] = par1
  }
  if (is.null(par2)) {
    par[2] = med+dev
  } else if ((is.double(par2)==FALSE)||(length(par2)!=1)) {
    res$error = c(res$error,"par2 must be numerical")
  } else if (par2<=par1) {
    res$error = c(res$error,"par2 must be > par1")
  } else {
    par[2] = par2
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize = 2*dev/sqrt(12)
  norm.grad = epsilon
  res$par1 = par[1]
  res$par2 = par[2]
  res$stepsize=stepsize
  trajectory = matrix(data=par,nrow=2,ncol=1)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = runif(n = n, min = 0, max = 1)
    x.sampled.scaled = par[1] + (par[2]-par[1])*x.sampled
    ker1 = K1d.diff(x.sampled.scaled,x.sampled.scaled,kernel=kernel,bdwth=bdwth)/(n-1)
    ker2 = K1d.diff(x.sampled.scaled,x,kernel=kernel,bdwth=bdwth)/n
    grad1 = mean(-x.sampled%*%ker1-(1-x.sampled)%*%ker2)
    grad2 = mean(x.sampled%*%ker1-x.sampled%*%ker2)
    grad = 2*c(grad1,grad2)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    if (par[1]>par[2]) {temp=par[1]; par[1]=par[2]; par[2]=temp}
    trajectory = cbind(trajectory,matrix(data=par,nrow=2,ncol=1))
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = runif(n = n, min = 0, max = 1)
    x.sampled.scaled = par[1] + (par[2]-par[1])*x.sampled
    ker1 = K1d.diff(x.sampled.scaled,x.sampled.scaled,kernel=kernel,bdwth=bdwth)/(n-1)
    ker2 = K1d.diff(x.sampled.scaled,x,kernel=kernel,bdwth=bdwth)/n
    grad1 = mean(-x.sampled%*%ker1-(1-x.sampled)%*%ker2)
    grad2 = mean(x.sampled%*%ker1-x.sampled%*%ker2)
    grad = 2*c(grad1,grad2)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = cbind(trajectory,matrix(data=par_mean,nrow=2,ncol=1))
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}

# model: continuous.uniform.loc par1 = center, fixed: par2 = length [center-length/2,center+length/2]

GD.MMD.continuous.uniform.loc = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
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
  if (is.null(par2)) {
    res$error = c(res$error,"par2 missing")
  } else if ((is.double(par2)==FALSE)||(length(par2)!=1)) {
    res$error = c(res$error,"par2 must be numerical")
  } else if (par2<=0) {
    res$error = c(res$error,"par2 must be positive")
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize=par2/sqrt(12)
  norm.grad = epsilon
  res$par1 = par
  res$par2 = par2
  res$stepsize=stepsize
  trajectory = matrix(data=par,nrow=2,ncol=1)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    diff = x-par
    grad = -2*mean(K1d(par+0.5*par2,x,kernel=kernel,bdwth=bdwth)-K1d(par-0.5*par2,x,kernel=kernel,bdwth=bdwth))
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    trajectory = cbind(trajectory,matrix(data=par,nrow=2,ncol=1))
  }
  
  # GD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    diff = x-par
    grad = -2*mean(K1d(par+0.5*par2,x,kernel=kernel,bdwth=bdwth)-K1d(par-0.5*par2,x,kernel=kernel,bdwth=bdwth))
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = cbind(trajectory,matrix(data=par_mean,nrow=2,ncol=1))
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}
