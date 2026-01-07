# model: Dirac mass at par1

GD.MMD.Dirac = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  if (bdwth<1/sqrt(n)) bdwth=1
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # initialiation of GD
  
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
  trajectory = c(par)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    grad = -2*mean(K1d.diff(par,x,kernel=kernel,bdwth=bdwth))
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    trajectory = c(trajectory,par)
  }
  
  # GD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    grad = -2*mean(K1d.diff(par,x,kernel=kernel,bdwth=bdwth))
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
