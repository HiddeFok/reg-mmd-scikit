
#' @importFrom stats glm
#' @importFrom stats rgamma
 

# model: Gamma regressiion model par1 = beta, fixed: par2 = shape

Gamma_loc_hat<-function(y, Z, intercept, sd.z, par1, par2, kernel, M.det, M.rand, bdwth, burnin, nsteps, stepsize, epsilon, sorted_obs, KX, eps_sg){
   	#preparation of the output "res"
	res<-list()
	
   	#Initial parameter value
	if(length(par1)==1 &&  par1=="auto"){
		suppressWarnings(par <-glm(y~Z-1, family="Gamma")$coefficients)

	}else{
		if(intercept==FALSE){
			par<-par1
		}else{
			suppressWarnings(work<-glm(y~Z-1, family="Gamma")$coefficients)
			par<-c(work[1],par1)
		}
	}

	#store parameters of the algorithms that will be used
	res$par1 <- par
	res$par2 <- par2
	res$phi<-par2
	
	#initialization 
	n<-length(y)
	cons<-((n-1)*n-2*M.det)/M.rand
   	KX.length<-length(KX)
  
  	norm.grad <- epsilon
	n.par<-length(par)
   	store<-matrix(0, nsteps,n.par)
   	grad_all<-rep(0,n.par)
   	log.eps<-log(eps_sg)
	res$convergence<-1

 	#running time
 	if(burnin>0){
   		for (i in 1:burnin) {
    			mu = exp(c(Z%*%as.matrix(par)))
    			y.sampled <-suppressWarnings(rgamma(n, shape=par2, rate=par2/mu))
    			y.sampled2 <-suppressWarnings(rgamma(n, shape=par2, rate=par2/mu))
      
    			ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
    			grad_p1<- -c(ker*(1-y.sampled/mu)/mu)*Z 
    
    			set1<-sorted_obs$IND[(n+1):(n+M.det),1]
    			set2<-sorted_obs$IND[(n+1):(n+M.det),2]
	    		ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
	    		grad_p2<-  -c(KX[(n+1):(n+M.det)]*ker*(1-y.sampled[set1]/mu[set1])/mu[set1])*Z[set1,]  

    			use_X<- sample((n+M.det+1):KX.length ,M.rand, replace=FALSE)
    			set1<-sorted_obs$IND[use_X,1]
    			set2<-sorted_obs$IND[use_X,2]
    
    			ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    			grad_p3<-  -c(KX[use_X]*ker*(1-y.sampled[set1]/mu[set1])/mu[set1])*Z[set1,]  
    
    			grad<-(colSums(as.matrix(grad_p1))+2*colSums(as.matrix(grad_p2))+cons*colSums(as.matrix(grad_p3)) )/n
    			grad_all<-grad_all+grad
    
    			norm.grad <-norm.grad + sum(grad^2)
    			par <- par-stepsize*grad/sqrt(norm.grad)
    			
    		}
  	}
  	
  	for (i in 1:nsteps) {
    		mu = exp(c(Z%*%as.matrix(par)))
    		y.sampled <-suppressWarnings(rgamma(n, shape=par2, rate=par2/mu))
    		y.sampled2 <-suppressWarnings(rgamma(n, shape=par2, rate=par2/mu))
      
    		ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
    		grad_p1<- -c(ker*(1-y.sampled/mu)/mu)*Z 
    
    		set1<-sorted_obs$IND[(n+1):(n+M.det),1]
    		set2<-sorted_obs$IND[(n+1):(n+M.det),2]
    		ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    		grad_p2<-  -c(KX[(n+1):(n+M.det)]*ker*(1-y.sampled[set1]/mu[set1])/mu[set1])*Z[set1,]  

    		use_X<- sample((n+M.det+1):KX.length ,M.rand, replace=FALSE)
    		set1<-sorted_obs$IND[use_X,1]
    		set2<-sorted_obs$IND[use_X,2]
    
    		ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    		grad_p3<-  -c(KX[use_X]*ker*(1-y.sampled[set1]/mu[set1])/mu[set1])*Z[set1,]  
    
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
	if(nsteps>1) store<-apply(store,2,cumsum)/(1:nsteps) 
	
	#return the results
	res$coefficients <- store[nsteps,]
  	res$trajectory<-store
  	return(res)
}



# model: Gamma model par1 = beta, par2 = shape

Gamma_hat<-function(y, Z, intercept, sd.z, par1, par2, kernel, M.det, M.rand, bdwth, burnin, nsteps, stepsize, epsilon, sorted_obs, KX, eps_sg){
 	#preparation of the output "res"
	res<-list()
	
	#Initial parameter value
	if(length(par1)==1 &&  par1=="auto"){
		suppressWarnings(mle<-glm(y~Z-1,family="Gamma"))
		p1<-mle$coefficients
		if(par2=="auto"){
			p2<-1/summary(mle)$dispersion
		}else{
			p2<-par2
		}
	}else{
		if(intercept==FALSE){
			p1<-par1
			if(par2=="auto"){
				suppressWarnings(mle<-glm(y~Z-1,family="Gamma"))
				p2<-1/summary(mle)$dispersion
			}else{
				p2<-par2
			}
		}else{
			suppressWarnings(mle<-glm(y~Z-1,family="Gamma"))
			p1<-c(mle[1],par1)
			if(par2=="auto"){
				suppressWarnings(mle<-glm(y~Z-1,family="Gamma"))
				p2<-1/summary(mle)$dispersion
			}else{
				p2<-par2
			}
		}
	}
	par<-c(p1,log(p2))
	
	#store parameters of the algorithms that will be used
	res$par1 <- p1
	res$par2 <- p2

	
	#initialization
	norm.grad<-epsilon
	n.par<-length(par)
	store<-matrix(0, nsteps,n.par)
	grad_all<-rep(0,n.par)
	n<-length(y)
	d<-length(p1)
   	cons<-((n-1)*n-2*M.det)/M.rand
   	KX.length<-length(KX)
  	log.eps<-log(eps_sg)
	res$convergence<-1

   	#running time  
   	if(burnin>0){
		for (i in 1:burnin) {
			mu <- exp(c(Z%*%as.matrix(par[1:d])))
			s.par<-exp(par[d+1])
    			y.sampled <-suppressWarnings(rgamma(n, shape=s.par, rate=s.par/mu))
    			y.sampled2 <-suppressWarnings(rgamma(n, shape=s.par, rate=s.par/mu))
      
    			ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
    			grad_p1<-cbind(-c(ker*(1-y.sampled/mu)/mu)*Z,  -c(digamma(s.par)-log(s.par)-log(y.sampled)+log(mu)-1+y.sampled/mu)*ker*s.par)
     
    			set1<-sorted_obs$IND[(n+1):(n+M.det),1]
   			set2<-sorted_obs$IND[(n+1):(n+M.det),2]
    			ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    			grad_p2<-cbind(-c(KX[(n+1):(n+M.det)]*ker*(1-y.sampled[set1]/mu[set1])/mu[set1])*Z[set1,],  
                 		  -c(digamma(s.par)-log(s.par)-log(y.sampled[set1])+log(mu[set1])-1+y.sampled[set1]/mu[set1])*KX[(n+1):(n+M.det)]*ker*s.par)
                   
   			use_X<- sample((n+M.det+1):KX.length ,M.rand, replace=FALSE)
   			set1<-sorted_obs$IND[use_X,1]
   			set2<-sorted_obs$IND[use_X,2]   
   			ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    			grad_p3<-cbind(-c(KX[use_X]*ker*(1-y.sampled[set1]/mu[set1])/mu[set1])*Z[set1,],  
                   		-c(digamma(s.par)-log(s.par)-log(y.sampled[set1])+log(mu[set1])-1+y.sampled[set1]/mu[set1])*KX[use_X]*ker*s.par) 
                     
			grad<- ( colSums(grad_p1)+2*colSums(grad_p2)+cons*colSums(grad_p3)  )/n
			grad_all<-grad_all+grad
		
			norm.grad <- norm.grad + sum(grad^2)
			par <- par-stepsize*grad/sqrt(norm.grad)
    		}

	}
	
	for (i in 1:nsteps) {
		mu <- exp(c(Z%*%as.matrix(par[1:d])))
		s.par<-exp(par[d+1])
    		y.sampled <-suppressWarnings(rgamma(n, shape=s.par, rate=s.par/mu))
    		y.sampled2 <-suppressWarnings(rgamma(n, shape=s.par, rate=s.par/mu))
     
    		ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
    		grad_p1<-cbind(-c(ker*(1-y.sampled/mu)/mu)*Z,  -c(digamma(s.par)-log(s.par)-log(y.sampled)+log(mu)-1+y.sampled/mu)*ker*s.par)
     
    		set1<-sorted_obs$IND[(n+1):(n+M.det),1]
   		set2<-sorted_obs$IND[(n+1):(n+M.det),2]
  		ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    		grad_p2<-cbind(-c(KX[(n+1):(n+M.det)]*ker*(1-y.sampled[set1]/mu[set1])/mu[set1])*Z[set1,],  
                		  -c(digamma(s.par)-log(s.par)-log(y.sampled[set1])+log(mu[set1])-1+y.sampled[set1]/mu[set1])*KX[(n+1):(n+M.det)]*ker*s.par)
                   
   		use_X<- sample((n+M.det+1):KX.length ,M.rand, replace=FALSE)
   		set1<-sorted_obs$IND[use_X,1]
   		set2<-sorted_obs$IND[use_X,2]   
   		ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    		grad_p3<-cbind(-c(KX[use_X]*ker*(1-y.sampled[set1]/mu[set1])/mu[set1])*Z[set1,],  
                   		-c(digamma(s.par)-log(s.par)-log(y.sampled[set1])+log(mu[set1])-1+y.sampled[set1]/mu[set1])*KX[use_X]*ker*s.par) 
                     
		grad<- ( colSums(grad_p1)+2*colSums(grad_p2)+cons*colSums(grad_p3)  )/n
		grad_all<-grad_all+grad
		
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
