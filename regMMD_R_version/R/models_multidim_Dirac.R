# model: multidimensional Dirac mass at par1

GD.MMD.multidim.Dirac = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = dim(x)[1]
  p = dim(x)[2]
  if (bdwth<1/sqrt(n)) bdwth=1
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par1)) {
    par = rep(0,p)
    for (i in 1:p) par[i] = median(x[,i])
  } else if ((is.vector(par1)==FALSE)||(is.numeric(par1)==FALSE)) {
    res$error = c(res$error,"par1 must be a numerical vector")
  } else if (length(par1)!=p) {
    res$error = c(res$error,"wrong dimension for par1")
  } else {
    par = par1
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize=1
  norm.grad = epsilon
  res$par1 = par
  res$par2 = NULL
  res$stepsize=stepsize
  trajectory = matrix(data=par,nrow=p,ncol=1)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    grad = -2*(rep(1/n,n)%*%Kmd.diff(x,par,kernel=kernel,bdwth=bdwth))[1,]
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    trajectory = cbind(trajectory,matrix(data=par,nrow=p,ncol=1))
  }
  
  # GD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    grad = -2*(rep(1/n,n)%*%Kmd.diff(x,par,kernel=kernel,bdwth=bdwth))[1,]
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = cbind(trajectory,matrix(data=par_mean,nrow=p,ncol=1))
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}
