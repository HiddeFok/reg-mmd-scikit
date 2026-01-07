
#' @importFrom stats glm
#' @importFrom stats rpois
 



Poisson_hat<-function(y, Z, intercept, sd.z, par1, kernel, M.det, M.rand, bdwth, burnin, nsteps, stepsize, epsilon, sorted_obs, KX, eps_sg){

	#preparation of the output "res"
  	res<-list()
  	
  	#Initial parameter value
   	if(length(par1)==1 &&  par1=="auto"){
     		suppressWarnings(par<-glm(y~Z-1, family = "poisson")$coefficients)
     	}else{
		if(intercept==FALSE){
			par<-par1
		}else{
			suppressWarnings(work<-glm(y~Z-1, family = "poisson")$coefficients)
			par<-c(work[1],par1)
		}
	}
   	
  	#store parameters of the algorithms that will be used
	res$par1 <- par
	
	#initialization 
	n<-length(y)
	cons<-((n-1)*n-2*M.det)/M.rand
   	KX.length<-length(KX)
  
  	norm.grad <- epsilon
	n.par<-length(par)
	store<-matrix(0, nsteps, n.par)
	grad_all<-rep(0,n.par)
	log.eps<-log(eps_sg)
	res$convergence<-1	

 	#running time
 	if(burnin>0){
   		for (i in 1:burnin) {
    			mu <- exp(c(Z%*%as.matrix(par)))
    			y.sampled <-rpois(n, mu)
			y.sampled2 <-rpois(n, mu)
    	
    			ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
    			grad_p1<- c(ker*(y.sampled-mu))*Z
    
    			set1<-sorted_obs$IND[(n+1):(n+M.det),1]
    			set2<-sorted_obs$IND[(n+1):(n+M.det),2]
    			ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    			grad_p2<- c(KX[(n+1):(n+M.det)]*ker*(y.sampled[set1]-mu[set1]))*Z[set1,] 
    		
    			use_X<- sample((n+M.det+1):KX.length ,M.rand, replace=FALSE)
    			set1<-sorted_obs$IND[use_X,1]
    			set2<-sorted_obs$IND[use_X,2]
    
    			ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    			grad_p3<- c(KX[use_X]*ker*(y.sampled[set1]-mu[set1]))*Z[set1,]
    
    			grad<-(colSums(as.matrix(grad_p1))+2*colSums(as.matrix(grad_p2))+cons*colSums(as.matrix(grad_p3)) )/n
    			grad_all<-grad_all+grad
    
    			norm.grad <-norm.grad + sum(grad^2)
    			par <- par-stepsize*grad/sqrt(norm.grad)
    		}
  	}
  	
  	for (i in 1:nsteps) {
    		mu <- exp(c(Z%*%as.matrix(par)))
    		y.sampled <-rpois(n, mu)
		y.sampled2 <-rpois(n, mu)
    	
    		ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
    		grad_p1<- c(ker*(y.sampled-mu))*Z
    
    		set1<-sorted_obs$IND[(n+1):(n+M.det),1]
    		set2<-sorted_obs$IND[(n+1):(n+M.det),2]
    		ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    		grad_p2<- c(KX[(n+1):(n+M.det)]*ker*(y.sampled[set1]-mu[set1]))*Z[set1,] 
    		
    		use_X<- sample((n+M.det+1):KX.length ,M.rand, replace=FALSE)
    		set1<-sorted_obs$IND[use_X,1]
    		set2<-sorted_obs$IND[use_X,2]
    
    		ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    		grad_p3<- c(KX[use_X]*ker*(y.sampled[set1]-mu[set1]))*Z[set1,]
    
    		grad<-(colSums(as.matrix(grad_p1))+2*colSums(as.matrix(grad_p2))+cons*colSums(as.matrix(grad_p3)) )/n
    		grad_all<-grad_all+grad
    
    		norm.grad <-norm.grad + sum(grad^2)
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
 	store<-t( t(store)/sd.z)
	if(nsteps>1)store<-apply(store,2,cumsum)/(1:nsteps) 
	
	#return the results
	res$coefficients <- store[nsteps,]
  	res$trajectory<-store
    
  	return(res)   
}	
  	
  	
  	
  	
  	
  	
  	
  	
  	
  	
  	
  	
  	
  	
