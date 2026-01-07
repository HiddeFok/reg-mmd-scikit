#' @importFrom stats cov rnorm

# first series of function: SGD

# model: multidim-Gaussian-loc par1 = mean, fixed: par2 = sd in N(mean, sd^2 * I)

SGD.MMD.multidim.Gaussian.loc = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = dim(x)[1]
  p = dim(x)[2]
  
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
  trajectory = matrix(data=par,nrow=p,ncol=1)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    x.sampled.centered = matrix(data=rnorm(n = n*p, mean = 0, sd = par2), nrow=n, ncol = p)
    x.sampled = x.sampled.centered + matrix(data=par, nrow=n, ncol = p, byrow=TRUE)
    ker = (Kmd(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-Kmd(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = x.sampled.centered/(par2^2)
    grad = c(2*t(gradL)%*%ker%*%matrix(data=1/n,nrow=n,ncol=1))
    norm.grad = norm.grad + sum(grad^2)
    par = par-stepsize*grad/sqrt(norm.grad)
    trajectory = cbind(trajectory,matrix(data=par,nrow=p,ncol=1))
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    x.sampled.centered = matrix(data=rnorm(n = n*p, mean = 0, sd = par2), nrow=n, ncol = p)
    x.sampled = x.sampled.centered + matrix(data=par, nrow=n, ncol = p, byrow=TRUE)
    ker = (Kmd(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-Kmd(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL = x.sampled.centered/(par2^2)
    grad = c(2*t(gradL)%*%ker%*%matrix(data=1/n,nrow=n,ncol=1))
    norm.grad = norm.grad + sum(grad^2)
    par = par-stepsize*grad/sqrt(norm.grad)
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = cbind(trajectory,matrix(data=par_mean,nrow=p,ncol=1))
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}

# model: multidim-Gaussian-scale par2 = U, fixed par1 = mean in N(mean, U'U)

SGD.MMD.multidim.Gaussian.scale = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = dim(x)[1]
  p = dim(x)[2]
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par1)) {
    res$error = c(res$error,"par1 missing")
  } else if ((is.vector(par1)==FALSE)||(is.numeric(par1)==FALSE)) {
    res$error = c(res$error,"par1 must be a numerical vector")
  } else if (length(par1)!=p) {
    res$error = c(res$error,"wrong dimension for par1")
  }
  if (is.null(par2)) {
    eig = eigen(cov(x))
    par = eig$vectors%*%diag(sqrt(eig$values))%*%t(eig$vectors)
  } else if (is.matrix(par2)==FALSE) {
    res$error = c(res$error,"par2 must be a matrix")
  } else if (dim(par2)!=c(p,p)) {
    res$error = c(res$error,"wrong dimension for par2")
  } else {
    par = par2
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize = mean(eigen(par)$values)
  norm.grad = epsilon
  res$par1 = par1
  res$par2 = par
  res$stepsize=stepsize
  trajectory = par
  
  # BURNIN period
  
  for (i in 1:burnin) {
    sigminus = solve(par%*%t(par))
    x.sampled.centered = (matrix(data=rnorm(n = n*p, mean = 0, sd = 1), nrow=n, ncol = p))%*%t(par)
    x.sampled = x.sampled.centered + matrix(data=par1, nrow=n, ncol = p, byrow=TRUE)
    ker = (Kmd(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-Kmd(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    grad = -2*sigminus%*%(sum(ker)*diag(p)-t(x.sampled.centered)%*%diag((ker%*%matrix(data=1,nrow=n,ncol=1))[,1])%*%(x.sampled.centered)%*%sigminus)%*%par/n
    norm.grad = norm.grad + norm(grad, type="F")^2
    par = par-stepsize*grad/sqrt(norm.grad)
    trajectory = cbind(trajectory,par)
  }
  
  # SGD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    sigminus = solve(par%*%t(par))
    x.sampled.centered = (matrix(data=rnorm(n = n*p, mean = 0, sd = 1), nrow=n, ncol = p))%*%t(par)
    x.sampled = x.sampled.centered + matrix(data=par1, nrow=n, ncol = p, byrow=TRUE)
    ker = (Kmd(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-Kmd(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    grad = -2*sigminus%*%(sum(ker)*diag(p)-t(x.sampled.centered)%*%diag((ker%*%matrix(data=1,nrow=n,ncol=1))[,1])%*%(x.sampled.centered)%*%sigminus)%*%par/n
    norm.grad = norm.grad + norm(grad, type="F")^2
    par = par-stepsize*grad/sqrt(norm.grad)
    par_mean = (par_mean*i + par)/(i+1)
    trajectory = cbind(trajectory,par_mean)
  }
  
  # return
  
  res$estimator = par_mean
  res$trajectory = trajectory
  return(res)
  
}

# model: multidim-Gaussian par1 = mean par2 = U in N(mean, U'U)

SGD.MMD.multidim.Gaussian = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = dim(x)[1]
  p = dim(x)[2]
  
  # preparation of the output "res"
  
  res = list(par1=par1, par2=par2, stepsize=stepsize, bdwth=bdwth, error=NULL, estimator=NULL, trajectory=NULL)
  
  # sanity check for the initialization, otherwise, set the default initialization for SGD
  
  if (is.null(par1)) {
    par1 = rep(0,p)
    for (i in 1:p) par1[i] = median(x[,i])
  } else if ((is.vector(par1)==FALSE)||(is.numeric(par1)==FALSE)) {
    res$error = c(res$error,"par1 must be a numerical vector")
  } else if (length(par1)!=p) {
    res$error = c(res$error,"wrong dimension for par1")
  }
  if (is.null(par2)) {
    eig = eigen(cov(x))
    par2 = eig$vectors%*%diag(sqrt(eig$values))%*%t(eig$vectors)
  } else if (is.matrix(par2)==FALSE) {
    res$error = c(res$error,"par2 must be a matrix")
  } else if (dim(par2)!=c(p,p)) {
    res$error = c(res$error,"wrong dimension for par2")
  }
  if (is.null(res$error)==FALSE) return(res)
  
  # initialization of norm.grad
  
  if (stepsize=="auto") stepsize = mean(eigen(par2)$values)
  norm.grad = epsilon
  res$par1 = par1
  res$par2 = par2
  res$stepsize=stepsize
  trajectory = list(t1=matrix(data=par1,nrow=p,ncol=1),t2=par2)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    sigminus = solve(par2%*%t(par2))
    x.sampled.centered = matrix(data=rnorm(n = n*p, mean = 0, sd = 1), nrow=n, ncol = p)%*%t(par2)
    x.sampled = x.sampled.centered + matrix(data=par1, nrow=n, ncol = p, byrow=TRUE)
    ker = (Kmd(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-Kmd(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL1 = x.sampled.centered%*%t(sigminus)
    grad1 = c(2*t(gradL1)%*%ker%*%matrix(data=1/n,nrow=n,ncol=1))
    grad2 = -2*sigminus%*%(sum(ker)*diag(p)-t(x.sampled.centered)%*%diag((ker%*%matrix(data=1,nrow=n,ncol=1))[,1])%*%(x.sampled.centered)%*%sigminus)%*%par2/n
    norm.grad = norm.grad + sum(grad1^2) + norm(grad2, type="F")^2
    par1 = par1-stepsize*grad1/sqrt(norm.grad)
    par2 = par2-stepsize*grad2/sqrt(norm.grad)
    trajectory$t1 = cbind(trajectory$t1, matrix(data=par1,nrow=p,ncol=1))
    trajectory$t2 = cbind(trajectory$t2, par2)
  }
  
  # SGD period
  
  par1_mean = par1
  par2_mean = par2
  
  for (i in 1:nstep) {
    sigminus = solve(par2%*%t(par2))
    x.sampled.centered = matrix(data=rnorm(n = n*p, mean = 0, sd = 1), nrow=n, ncol= p)%*%t(par2)
    x.sampled = x.sampled.centered + matrix(data=par1, nrow=n, ncol= p, byrow=TRUE)
    ker = (Kmd(x.sampled,x.sampled,kernel=kernel,bdwth=bdwth)-diag(n))/(n-1)-Kmd(x.sampled,x,kernel=kernel,bdwth=bdwth)/n
    gradL1 = x.sampled.centered%*%t(sigminus)
    grad1 = c(2*t(gradL1)%*%ker%*%matrix(data=1/n,nrow=n,ncol=1))
    grad2 = -2*sigminus%*%(sum(ker)*diag(p)-t(x.sampled.centered)%*%diag((ker%*%matrix(data=1,nrow=n,ncol=1))[,1])%*%(x.sampled.centered)%*%sigminus)%*%par2/n
    norm.grad = norm.grad + sum(grad1^2) + norm(grad2, type="F")^2
    par1 = par1-stepsize*grad1/sqrt(norm.grad)
    par2 = par2-stepsize*grad2/sqrt(norm.grad)
    par1_mean = (par1_mean*i + par1)/(i+1)
    par2_mean = (par2_mean*i + par2)/(i+1)
    trajectory$t1 = cbind(trajectory$t1, matrix(data=par1_mean,nrow=p,ncol=1))
    trajectory$t2 = cbind(trajectory$t2, par2_mean)
  }
  
  # return
  
  res$estimator = list(par1 = par1_mean, par2 = par2_mean)
  res$trajectory = trajectory
  return(res)
  
}

# second series of function: GD

# model: multidim-Gaussian-loc par1 = mean, fixed: par2 = sd in N(mean, sd^2 * I)

GD.MMD.multidim.Gaussian.loc = function(x, par1, par2, kernel, bdwth, burnin, nstep, stepsize, epsilon) {
  
  n = dim(x)[1]
  p = dim(x)[2]
  
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
  trajectory = matrix(data=par,nrow=p,ncol=1)
  
  # BURNIN period
  
  for (i in 1:burnin) {
    diff = (matrix(data=par,nrow=n,ncol=p,byrow=TRUE)-x)/bdwth
    w = ((diff^2)%*%rep(1,p))[,1]/(2*(par2^2)+bdwth^2)
    grad = 4*(1/(1+2*(par2^2)/bdwth))*(rep(1/n,n)%*%(diff*exp(-w)))[1,]
    norm.grad = norm.grad + grad^2
    par = par-stepsize*grad/sqrt(norm.grad)
    trajectory = cbind(trajectory,matrix(data=par,nrow=p,ncol=1))
  }
  
  # GD period
  
  par_mean = par
  
  for (i in 1:nstep) {
    diff = (matrix(data=par,nrow=n,ncol=p,byrow=TRUE)-x)/bdwth
    w = ((diff^2)%*%rep(1,p))[,1]/(2*(par2^2)+bdwth^2)
    grad = 4*(1/(1+2*(par2^2)/bdwth))*(rep(1/n,n)%*%(diff*exp(-w)))[1,]
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
