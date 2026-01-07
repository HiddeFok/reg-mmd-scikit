#' @importFrom stats dbinom rbinom

# first series of function: SGD

# model: binomial.prob par2 = p fixed par1 = N

SGD.MMD.binomial.prob = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  if (bdwth<1) bdwth=1
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par1)) {
    res$error = c(res$error,"par1 missing")
  } else if ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  } else if (floor(par1)!=par1) {
    res$error = c(res$error,"par1 must be integer")
  } else if (par1<=0) {
    res$error = c(res$error,"par1 must be positive")
  }
  if (is.null(par2)) {
    mea = mean(x)/par1
    if (mea<=1/n) par = 1/n else if (mea>=1-1/n) par = 1-1/n else par = mea
  } else if ((is.double(par2)==FALSE)||(length(par2)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  } else if ((par1<=0)||(par1>=1)) {
    res$error = c(res$error,"par1 must be in (0,1)")
  } else {
    par=par1
  }
  
  if (is.null(res$error)==FALSE) return(res)

  if (is.integer(x)==FALSE) res$error = c(res$error,"Attention: you used the binomial model on non-integer observations. If this is intentional, you can ignore this message. If your observations are integers but stored as numerical values, you can use x=as.integer(x) to avoid this warning.")
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize = 1/par1
  norm.grad = epsilon
  res$par1 = par1
  res$par2 = par
  res$stepsize=stepsize
  trajectory = c(par)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = rbinom(n = n, size = par1, prob = par)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = x.sampled/par-(par1-x.sampled)/(1-par)
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    if (par<1/n) par = 1/n
    if (par>1-1/n) par = 1-1/n
    trajectory = c(trajectory,par)
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = rbinom(n = n, size = par1, prob = par)
    ker = (K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = x.sampled/par-(par1-x.sampled)/(1-par)
    grad = 2*mean(gradL%*%ker)
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    if (par<1/n) par = 1/n
    if (par>1-1/n) par = 1-1/n
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = c(trajectory,par_mean)
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}

# second series of function: GD

# model: binomial.prob par2 = p fixed par1 = N

GD.MMD.binomial.prob = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  if (bdwth<1) bdwth=1
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par1)) {
    res$error = c(res$error,"par1 missing")
  } else if ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  } else if (floor(par1)!=par1) {
    res$error = c(res$error,"par1 must be integer")
  } else if (par1<=0) {
    res$error = c(res$error,"par1 must be positive")
  }
  if (is.null(par2)) {
    mea = mean(x)/par1
    if (mea<=1/n) par = 1/n else if (mea>=1-1/n) par = 1-1/n else par = mea
  } else if ((is.double(par2)==FALSE)||(length(par2)!=1)) {
    res$error = c(res$error,"par2 must be numerical")
  } else if ((par2<=0)||(par2>=1)) {
    res$error = c(res$error,"par2 must be in (0,1)")
  } else {
    if (par2<=1/n) par = 1/n else if (par2>=1-1/n) par = 1-1/n else par = par2
  }
  if (is.null(res$error)==FALSE) return(res)

  if (is.integer(x)==FALSE) res$error = c(res$error,"Attention: you used the binomial model on non-integer observations. If this is intentional, you can ignore this message. If your observations are integers but stored as numerical values, you can use x=as.integer(x) to avoid this warning.")
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize=1/par1
  norm.grad = epsilon
  res$par1 = par1
  res$par2 = par
  res$stepsize=stepsize
  trajectory = c(par)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled = c(0:par1)
    gradL = x.sampled/par-(par1-x.sampled)/(1-par)
    prob = dbinom(x=x.sampled,size = par1, prob = par)
    grad = 2*((gradL*prob)%*%K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)%*%prob-mean((gradL*prob)%*%K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)))[1,1]
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    if (par<1/n) par = 1/n
    if (par>1-1/n) par = 1-1/n
    trajectory = c(trajectory,par)
  }
  
  # GD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled = c(0:par1)
    gradL = x.sampled/par-(par1-x.sampled)/(1-par)
    prob = dbinom(x=x.sampled,size = par1, prob = par)
    grad = 2*((gradL*prob)%*%K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)%*%prob-mean((gradL*prob)%*%K1d(x.sampled,x,kernel=kernel,bdwth=bdwth)))[1,1]
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    if (par<1/n) par = 1/n
    if (par>1-1/n) par = 1-1/n
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = c(trajectory,par_mean)
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}

# model: binomial par1 = N par2 = p // here, if par1 provided, we take it as the maximum value to test

GD.MMD.binomial = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  if (bdwth<1) bdwth=1
  
  # preparation of the output "res"
  
  res = list(par1=NULL, par2=NULL, stepsize=NULL, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par1)) {
    parmax = Inf
  } else if ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  } else if (floor(par1)!=par1) {
    res$error = c(res$error,"par1 must be integer")
  } else if (par1<=0) {
    res$error = c(res$error,"par1 must be positive")
  } else {
    parmax = par1
  }
  if (is.null(res$error)==FALSE) return(res)

  if (is.integer(x)==FALSE) res$error = c(res$error,"Attention: you used the binomial model on non-integer observations. If this is intentional, you can ignore this message. If your observations are integers but stored as numerical values, you can use x=as.integer(x) to avoid this warning.")
  
  # in this model, we have explicit formula, so we do a loop for all possible par
  
  res$par1 = parmax
  res$par2 = 0.5
  best.crit = K1d(0,0,kernel=kernel,bdwth=bdwth)-2*mean(K1d(0,x,kernel=kernel,bdwth=bdwth))
  best.size = 0
  best.p = 0.5
  size = 0
  crit.old = best.crit
  carryon = TRUE
  while ((carryon)&&(size<parmax)) {
    size = size + 1
    x.sampled = c(0:size)
    estimation = GD.MMD.binomial.prob(x=x, par1=size, par2=NULL, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    p = estimation$estimator
    prob = dbinom(x=x.sampled,size = size, prob = p)
    crit = prob%*%K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)%*%prob-2*mean(prob%*%K1d(x.sampled,x,kernel=kernel,bdwth=bdwth))
    if ((parmax==Inf)&&(crit[1,1]>crit.old)) {
      carryon = FALSE
    } else if (crit[1,1]<best.crit) {
      best.crit = crit[1,1]
      best.size = size
      best.p = p
      trajectory = estimation$trajectory
    }
    crit.old = crit[1,1]
  }
  
  res$estimator = c(best.size,best.p)
  res$trajectory = trajectory
  return(res)
  
}

# third series of functions: EXACT

# model: binomial.size par1 = N fixed par2 = p // here, if par1 provided, we take it as the maximum value to test

EXACT.MMD.binomial.size = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = length(x)
  if (bdwth<1) bdwth=1
  
  # preparation of the output "res"
  
  res = list(par1=NULL, par2=NULL, stepsize=NULL, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # initialiation
  
  if (is.null(par1)) {
    parmax = Inf
  } else if ((is.double(par1)==FALSE)||(length(par1)!=1)) {
    res$error = c(res$error,"par1 must be numerical")
  } else if (floor(par1)!=par1) {
    res$error = c(res$error,"par1 must be integer")
  } else if (par1<=0) {
    res$error = c(res$error,"par1 must be positive")
  } else {
    parmax = par1
  }
  if (is.null(par2)) {
    res$error = c(res$error,"par2 missing")
  } else if ((is.double(par2)==FALSE)||(length(par2)!=1)) {
    res$error = c(res$error,"par2 must be numerical")
  } else if ((par2<=0)||(par2>=1)) {
    res$error = c(res$error,"par2 must be in (0,1)")
  }
  if (is.null(res$error)==FALSE) return(res)

  if (is.integer(x)==FALSE) res$error = c(res$error,"Attention: you used the binomial model on non-integer observations. If this is intentional, you can ignore this message. If your observations are integers but stored as numerical values, you can use x=as.integer(x) to avoid this warning.")
  
  # in this model, we have explicit formula, so we do a loop for all possible par
  
  res$par1 = parmax
  res$par2 = par2
  best.crit = Inf
  crit.old = best.crit
  best.par = -1
  par = 0
  carryon = TRUE
  while ((par<parmax)&&(carryon)) {
    par = par+1
    x.sampled = c(0:par)
    prob = dbinom(x=x.sampled,size = par, prob = par2)
    crit = prob%*%K1d(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)%*%prob-2*mean(prob%*%K1d(x.sampled,x,kernel=kernel,bdwth=bdwth))
    if ((parmax==Inf)&&(crit[1,1]>crit.old)) {
      carryon = FALSE
    } else if (crit[1,1]<best.crit) {
      best.crit = crit
      best.par = par
    }
    crit.old = crit[1,1]
  }
  
  res$estimator = best.par
  return(res)
  
}

