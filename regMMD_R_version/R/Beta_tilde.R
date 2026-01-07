DEFAULT_SCALE<-1
EPS_BETA<-10^{-3}
EPS_BETA2<-10^{-15}

#' @importFrom stats glm
#' @importFrom stats binomial
#' @importFrom stats rbeta

# model: Beta regression model par1 = beta, fixed: par2 = precision

Beta_loc_tilde<-function(y, Z, intercept, sd.z, par1, par2, kernel, bdwth, burnin,  nsteps, stepsize, epsilon, eps_sg){
	#preparation of the output "res"
	res<-list()
   	#Initial parameter value
	if(length(par1)==1 &&  par1=="auto"){
		suppressWarnings(par<-glm(y~Z-1, family = binomial)$coefficients)
	}else{
		if(intercept==FALSE){
			par<-par1
		}else{
			suppressWarnings(work<-glm(y~Z-1, family = binomial)$coefficients)
			par<-c(work[1],par1)
		}
	}
	
	#store parameters of the algorithms that will be used
	res$par1<- par
	res$par2<- par2
	res$phi<-par2
   
	#initialization
	n<-length(y)
	n.par<-length(par)
	norm.grad<- epsilon
	store<-matrix(0, nsteps,n.par)
	grad_all<-rep(0, n.par)
	log.eps<-log(eps_sg)
	res$convergence<-1
	 
  	##running time
  	if(burnin>0){
		for (i in 1:burnin) {
			mu <- 1/(1+exp(-c(Z%*%as.matrix(par))))
			mu[mu<EPS_BETA]<-EPS_BETA
			mu[mu>1-EPS_BETA]<-1-EPS_BETA
			y.sampled <-rbeta(n, shape1=mu*par2,shape2=(1-mu)*par2)
			y.sampled2 <-rbeta(n, shape1=mu*par2,shape2=(1-mu)*par2)
			y.sampled[y.sampled<EPS_BETA2]<-EPS_BETA2
			y.sampled[y.sampled>1-EPS_BETA2]<-1-EPS_BETA2
			y.sampled2[y.sampled2<EPS_BETA2]<-EPS_BETA2
			y.sampled2[y.sampled2>1-EPS_BETA2]<-1-EPS_BETA2
			ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
			p1<- -par2*digamma(mu*par2)+par2*digamma( (1-mu)*par2)+par2*log(y.sampled)-par2*log(1-y.sampled)
			grad<- apply(c(ker*mu*(1-mu)*p1)*Z,2,mean)
			grad_all<-grad_all+grad
			norm.grad<- norm.grad + sum(grad^2)
			par<- par-stepsize*grad/sqrt(norm.grad)
		}
	} 
	
	for (i in 1:nsteps) {
		mu <- 1/(1+exp(-c(Z%*%as.matrix(par))))
		mu[mu<EPS_BETA]<-EPS_BETA
		mu[mu>1-EPS_BETA]<-1-EPS_BETA
		y.sampled <-rbeta(n, shape1=mu*par2,shape2=(1-mu)*par2)
		y.sampled2 <-rbeta(n, shape1=mu*par2,shape2=(1-mu)*par2)
		y.sampled[y.sampled<EPS_BETA2]<-EPS_BETA2
		y.sampled[y.sampled>1-EPS_BETA2]<-1-EPS_BETA2
		y.sampled2[y.sampled2<EPS_BETA2]<-EPS_BETA2
		y.sampled2[y.sampled2>1-EPS_BETA2]<-1-EPS_BETA2
		ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
		p1<- -par2*digamma(mu*par2)+par2*digamma( (1-mu)*par2)+par2*log(y.sampled)-par2*log(1-y.sampled)
		grad<- apply(c(ker*mu*(1-mu)*p1)*Z,2,mean)
		grad_all<-grad_all+grad
		norm.grad<- norm.grad + sum(grad^2)
		par<- par-stepsize*grad/sqrt(norm.grad)
		store[i,]<-par
		if(is.nan(mean(grad_all))){
			res$convergence<- -1
			break
		}
		g1<-sqrt(  sum((grad_all/(burnin+i))^2) )/n.par
		
		if(log(g1)< log.eps){
  				res$convergence<-0
  				break
  		}
	}
	nsteps<-i
  	store<- matrix(store[1:nsteps,],nrow=nsteps,ncol=n.par, byrow=FALSE)
	#compute the estimator
 	store<-t( t(store)/sd.z)
	if(nsteps>1)store<-apply(store,2,cumsum)/(1:nsteps) 
	
	#return the results
	res$coefficients <- store[nsteps,]
  	res$trajectory<-store
 
  	return(res) 
}



# model: Beta regression model par1 = beta,  par2 = precision

Beta_tilde<-function(y, Z, intercept, sd.z, par1, par2, kernel, bdwth, burnin,  nsteps, stepsize, epsilon, eps_sg){

	#preparation of the output "res"
	res<-list()
	
	#Initial parameter value
	if(length(par1)==1 &&  par1=="auto"){
		suppressWarnings(mle<-glm(y~Z-1, family = binomial))
		p1<-mle$coefficients
		if(par2=="auto"){
			p2<-DEFAULT_SCALE
		}else{
			p2<-par2
		}
	}else{
		if(intercept==FALSE){
			p1<-par1
			if(par2=="auto"){
				p2<-DEFAULT_SCALE 
			}else{
				p2<-par2
			}
		}else{
			suppressWarnings(mle<-glm(y~Z-1, family = binomial))
			p1<-c(mle[1],par1)
			if(par2=="auto"){
				p2<-DEFAULT_SCALE
			}else{
				p2<-par2
			}
		}
	}
	par<-c(p1,log(p2))
	
	#store parameters of the algorithms that will be used
	res$par1<- p1
	res$par2<- p2
	
 	#initialization
 	n<-length(y)
 	d<-length(p1)
 	n.par<-length(par)
	norm.grad<- epsilon
	store<-matrix(0, nsteps,n.par)
	grad_all<-rep(0,n.par)
	log.eps<-log(eps_sg)
	res$convergence<-1
	
   	if(burnin>0){
		for (i in 1:burnin) {
			mu <- 1/(1+exp(-c(Z%*%as.matrix(par[1:d]))))
			s.par<-exp(par[d+1])
			mu[mu<EPS_BETA]<-EPS_BETA
			mu[mu>1-EPS_BETA]<-1-EPS_BETA
			y.sampled <-rbeta(n, shape1=mu*s.par,shape2=(1-mu)*s.par)
			y.sampled2 <-rbeta(n, shape1=mu*s.par,shape2=(1-mu)*s.par)
			y.sampled[y.sampled<EPS_BETA2]<-EPS_BETA2
			y.sampled[y.sampled>1-EPS_BETA2]<-1-EPS_BETA2
			y.sampled2[y.sampled2<EPS_BETA2]<-EPS_BETA2
			y.sampled2[y.sampled2>1-EPS_BETA2]<-1-EPS_BETA2
			ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
			dgam1<-digamma(mu*s.par)
			dgam2<-digamma( (1-mu)*s.par)
			p1<- -s.par*dgam1+s.par*dgam2+s.par*log(y.sampled)-s.par*log(1-y.sampled)
			p2<- digamma(s.par)-mu*dgam1-(1-mu)*dgam2+mu*log(y.sampled)+(1-mu)*log(1-y.sampled)
			grad<- c(apply(c(ker*mu*(1-mu)*p1)*Z,2,mean), mean(ker*p2)) 
			grad_all<-grad_all+grad
    			norm.grad <- norm.grad + sum(grad^2)
    			par <- par-stepsize*grad/sqrt(norm.grad)
    		}
  	}
  	
  	for (i in 1:nsteps) {
		mu <- 1/(1+exp(-c(Z%*%as.matrix(par[1:d]))))
		s.par<-exp(par[d+1])
		mu[mu<EPS_BETA]<-EPS_BETA
		mu[mu>1-EPS_BETA]<-1-EPS_BETA
		y.sampled <-rbeta(n, shape1=mu*s.par,shape2=(1-mu)*s.par)
		y.sampled2 <-rbeta(n, shape1=mu*s.par,shape2=(1-mu)*s.par)
		y.sampled[y.sampled<EPS_BETA2]<-EPS_BETA2
		y.sampled[y.sampled>1-EPS_BETA2]<-1-EPS_BETA2
		y.sampled2[y.sampled2<EPS_BETA2]<-EPS_BETA2
		y.sampled2[y.sampled2>1-EPS_BETA2]<-1-EPS_BETA2
		ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
		dgam1<-digamma(mu*s.par)
		dgam2<-digamma( (1-mu)*s.par)
		p1<- -s.par*dgam1+s.par*dgam2+s.par*log(y.sampled)-s.par*log(1-y.sampled)
		p2<- digamma(s.par)-mu*dgam1-(1-mu)*dgam2+mu*log(y.sampled)+(1-mu)*log(1-y.sampled)
		grad<- c(apply(c(ker*mu*(1-mu)*p1)*Z,2,mean), mean(ker*p2)) 
		grad_all
    		norm.grad <- norm.grad + sum(grad^2)
    		par <- par-stepsize*grad/sqrt(norm.grad)
    		store[i,]<-par
    		if(is.nan(mean(grad_all))){
			res$convergence<- -1
			break
		}
		g1<-sqrt(  sum((grad_all/(burnin+i))^2) )/n.par
		
		if(log(g1)< log.eps){
  				res$convergence<-0
  				break
  		}
  	}
  	nsteps<-i
  	store<- matrix(store[1:nsteps,],nrow=nsteps,ncol=n.par, byrow=FALSE)
  	#compute the estimator
 	store[,1:d]<-t( t(store[,1:d])/sd.z)
 	store[,d+1]<-exp(store[, d+1])  
	if(nsteps>1)store<-apply(store,2,cumsum)/(1:nsteps) 
	
	#return the results
	res$coefficients <- store[nsteps,1:d]
	res$phi<-store[nsteps,d+1]
  	res$trajectory<-store
  	return(res)
   
}


 


