

#' @importFrom stats glm
#' @importFrom stats rpois
 


Poisson_tilde<-function(y, Z, intercept, sd.z, par1, kernel, bdwth, burnin,  nsteps, stepsize, epsilon, eps_sg){

	#preparation of the output "res"
  	res<-list()
  	
  	#Initial parameter value
   	if(length(par1)==1 &&  par1=="auto"){
     		suppressWarnings(par<-glm(y~Z-1, family = "poisson")$coefficients)
     	}else{
		if(intercept==FALSE){
			par<-par1
		}else{
			work<-glm(y~Z-1, family = "poisson")$coefficients
			par<-c(work[1],par1)
		}
	}
   
  	#store parameters of the algorithms that will be used
	res$par1<- par
	
	#initialization
	norm.grad<- epsilon
	n.par<-length(par)
	store<-matrix(0, nsteps, n.par)
	grad_all<-rep(0,n.par)
	log.eps<-log(eps_sg)
	res$convergence<-1
 	n<-length(y)
 	
  	##running time
  	if(burnin>0){
		for (i in 1:burnin) {
			mu <- exp(c(Z%*%as.matrix(par)))
			y.sampled <-rpois(n, mu)
			y.sampled2 <-rpois(n, mu)
			ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
			grad<- apply(c(ker*(y.sampled-mu))*Z,2,mean)
			grad_all<-grad_all+grad
			norm.grad<- norm.grad + sum(grad^2)
			par<- par-stepsize*grad/sqrt(norm.grad)
		}
	}
	
	for (i in 1:nsteps) {
		mu <- exp(c(Z%*%as.matrix(par)))
		y.sampled <-rpois(n, mu)
		y.sampled2 <-rpois(n, mu)
		ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
		grad<- apply(c(ker*(y.sampled-mu))*Z,2,mean)
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
  	
  	
  	
  	
  	
  	
  	
  	
  	
  	
  	
  	
  	
  	
