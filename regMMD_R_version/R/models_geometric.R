#' @importFrom stats rgeom

# model: geometric par1 = p [(1-p)^x p, starts from x=0]

SGD.MMD.geometric = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  if (bdwth<1) bdwth=1
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)

  if (is.integer(x)==FALSE) res$error = c(res$error,"Attention: you used the geometric model on non-integer observations. If this is intentional, you can ignore this message. If your observations are integers but stored as numerical values, you can use x=as.integer(x) to avoid this warning.")
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  med = median(x)
  if (med==0) {
    scaling = 0.9
  } else {
    scaling = 1/(median(x)+1)
  }
  if (is.null(par1)) {
    par = scaling
  } else if ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  } else if ((par1<=0)||(par1>=1)) {
    res$error = c(res$error,"par1 must be in (0,1)")
  } else {
    par=par1
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize=scaling
  norm.grad = epsilon
  res$par1 = par
  res$par2 = NULL
  res$stepsize=stepsize
  trajectory = c(par)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = rgeom(n = n, prob = par)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL =  1/par-x.sampled/(1-par)
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    if (par<1/n) par = 1/n
    if (par>1-1/n) par = 1-1/n
    trajectory = c(trajectory, par)
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = rgeom(n = n, prob = par)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL =  1/par-x.sampled/(1-par)
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    if (par<1/n) par = 1/n
    if (par>1-1/n) par = 1-1/n
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = c(trajectory, par_mean)
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}
