# model: discrete.uniform par1 = N in Unif{1,2,...,N}

EXACT.MMD.discrete.uniform = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  if (bdwth<1) bdwth=1
  
  # preparation of the output "res"
  
  res = list(par1=Inf, par2=NULL, stepsize=NULL, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)

  if (is.integer(x)==FALSE) res$error = c(res$error,"Attention: you used the discrete uniform model on non-integer observations. If this is intentional, you can ignore this message. If your observations are integers but stored as numerical values, you can use x=as.integer(x) to avoid this warning.")
  
  # in this model, we have explicit formula, so we do a loop for all possible par
  
  best.crit = K1d(1,1,kernel=kernel,bdwth=bdwth)-2*mean(K1d(1,x,kernel=kernel,bdwth=bdwth))
  best.par = 1
  if (max(x)>=1) for (par in 2:(2*max(x))) {
    x.sampled = c(1:par)
    prob = rep(1/par,par)
    crit = prob%*%K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)%*%prob-2*mean(prob%*%K1d(x.sampled,x,kernel=kernel,bdwth=bdwth))
    if (crit[1,1]<best.crit) {
      best.crit = crit
      best.par = par
    }
  }
  
  res$estimator = best.par
  return(res)
  
}
