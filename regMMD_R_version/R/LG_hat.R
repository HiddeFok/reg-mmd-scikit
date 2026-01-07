
#' @importFrom stats lm
#' @importFrom stats rnorm
 

#Model: Linear Gaussian model par1 = beta, fixed: par2 = sd
#Algorithm: Stochastic gradient descent (Adagrad)

								 
LG_SGD_loc_hat<-function(y, Z, intercept, sd.z, par1, par2, kernel, M.det, M.rand, bdwth, burnin, nsteps, stepsize, epsilon, sorted_obs, KX, eps_sg){

   	#preparation of the output "res"
	res <-list()

	#Initial parameter value
	if(length(par1)==1 &&  par1=="auto"){
		par <-lm(y~Z-1)$coefficients

	}else{
		if(intercept==FALSE){
			par<-par1
		}else{
			par<-c(mean(y),par1)
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
    			mu <- c(Z%*%as.matrix(par))
			y.sampled <-rnorm(n,mu, par2)
			y.sampled2 <-rnorm(n,mu, par2)
      
			ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
			grad_p1<- 2*c((y.sampled-mu)*ker)*Z  
    
			set1<-sorted_obs$IND[(n+1):(n+M.det),1]
			set2<-sorted_obs$IND[(n+1):(n+M.det),2]
			ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
			grad_p2<-  c(KX[(n+1):(n+M.det)]*ker*(y.sampled[set1]-mu[set1]))*Z[set1,]  

			use_X<- sample((n+M.det+1):KX.length ,M.rand, replace=FALSE)
			set1<-sorted_obs$IND[use_X,1]
			set2<-sorted_obs$IND[use_X,2]
			ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    			grad_p3<-  c(KX[use_X]*ker*(y.sampled[set1]-mu[set1]))*Z[set1,]  
    
    			grad<- (colSums(as.matrix(grad_p1))+2*colSums(as.matrix(grad_p2))+cons*colSums(as.matrix(grad_p3)))/n
    			grad_all <- grad_all+grad
    			norm.grad <- norm.grad + sum(grad^2)
    			par <- par-stepsize*grad/sqrt(norm.grad)
    		}
	}
	for (i in 1:nsteps) {
    		mu <- c(Z%*%as.matrix(par))
		y.sampled <-rnorm(n,mu, par2)
		y.sampled2 <-rnorm(n,mu, par2)
      
		ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
		grad_p1<- 2*c((y.sampled-mu)*ker)*Z  
    
		set1<-sorted_obs$IND[(n+1):(n+M.det),1]
		set2<-sorted_obs$IND[(n+1):(n+M.det),2]
		ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
		grad_p2<-  c(KX[(n+1):(n+M.det)]*ker*(y.sampled[set1]-mu[set1]))*Z[set1,]  

		use_X<- sample((n+M.det+1):KX.length ,M.rand, replace=FALSE)
		set1<-sorted_obs$IND[use_X,1]
		set2<-sorted_obs$IND[use_X,2]
		ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    		grad_p3<-  c(KX[use_X]*ker*(y.sampled[set1]-mu[set1]))*Z[set1,]  
    
    		grad<- (colSums(as.matrix(grad_p1))+2*colSums(as.matrix(grad_p2))+cons*colSums(as.matrix(grad_p3)))/n
    		norm.grad <- norm.grad + sum(grad^2)
    		par <- par-stepsize*grad/sqrt(norm.grad)
    		store[i,]<-par
    		
    		grad_all <- grad_all+grad
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
  	store<- matrix(store[1:nsteps,],nrow=nsteps,ncol=length(par), byrow=FALSE)
	#compute the estimator
      	store<-t( t(store)/sd.z)
 	if(nsteps>1) store<-apply(store,2,cumsum)/(1:nsteps) 
 	     
  	# return the resuls
  	res$coefficients <- store[nsteps,]
  	res$trajectory<-store
 	return(res)
   
}

#Model: Linear Gaussian model par1 = beta, fixed: par2 = sd
#Estimator: hat{theta} with a Gaussian kernel on Y
#Algorithm: stochastic gradient descent (Adagrad) with explicit computation of expectation of the kernels.


LG_PartialSGD_loc_hat<-function(y, Z, intercept, sd.z, par1, par2, M.det, M.rand, bdwth, burnin, nsteps, stepsize, epsilon, sorted_obs, KX, eps_sg) {
	#preparation of the output "res"
	res <-list()

	#Initial parameter value
	if(length(par1)==1 &&  par1=="auto"){
		par <-lm(y~Z-1)$coefficients

	}else{
		if(intercept==FALSE){
			par<-par1
		}else{
			par<-c(mean(y),par1)
		}
	}
	n<-length(y)
 	cons0<-((n-1)*n-2*M.det)/M.rand
   	KX.length<-length(KX)
   	
   	# store parameters of the algorithms that will be used
	res$par1 <- par
	res$par2 <- par2
	res$phi<-par2
	
	# initialization 
	norm.grad <- epsilon
   	n.par<-length(par)
   	store<-matrix(0, nsteps,n.par)
   	grad_all<-rep(0,n.par)
	log.eps<-log(eps_sg)
	res$convergence<-1
   	
   	cons<-2*par2^2+ bdwth^2
	cons2<-sqrt(bdwth^2/(2*par2^2+ bdwth^2))
	cons3<-sqrt(bdwth^2/(4*par2^2+ bdwth^2))
	cons4<-4*par2^2+ bdwth^2
   
   	# running time
   	if(burnin>0){
   		for(i in 1:burnin){
			mu <-c(Z%*%as.matrix(par))
	    		diff <- y-mu
			work<- exp(-(diff^2)/cons)
    			grad_p1<- -(4*cons2/cons)*c(diff*work)*Z
    
    			set1<-sorted_obs$IND[(n+1):(n+M.det),1]
    			set2<-sorted_obs$IND[(n+1):(n+M.det),2]
    			diff<-mu[set1]-mu[set2]
    			grad_p21<- -2*cons3*c(KX[(n+1):(n+M.det)]*diff*exp(-(diff^2)/cons4))*(Z[set1,]-Z[set2,])
    			diff <- y[set2]-mu[set1]
    			work<- exp(-(diff^2)/cons)
    			grad_p22<- -(4*cons2/cons)*c(KX[(n+1):(n+M.det)]*diff*work)*Z[set1,]
    			grad_p2<-grad_p21+grad_p22

    			use_X<- sample((n+M.det+1):KX.length,M.rand, replace=FALSE)
    			set1<-sorted_obs$IND[use_X,1]
   			set2<-sorted_obs$IND[use_X,2]
    
    			diff<-mu[set1]-mu[set2]
    			grad_p31<- -2*cons3*c(KX[use_X]*diff*exp(-(diff^2)/cons4))*(Z[set1,]-Z[set2,])
    			diff <- y[set2]-mu[set1]
    			work<- exp(-(diff^2)/cons)
    			grad_p32<- -(4*cons2/cons)*c(KX[use_X]*diff*work)*Z[set1,]
    			grad_p3<-grad_p31+grad_p32
  
   			grad<- (colSums(as.matrix(grad_p1))+2*colSums(as.matrix(grad_p2))+cons0*colSums(as.matrix(grad_p3)))/n
    			norm.grad <- norm.grad + sum(grad^2)
    			grad_all <- grad_all+grad
    			par<- par-stepsize*grad/sqrt(norm.grad)
    		}
  	}
  	
  	for(i in 1:nsteps){
		mu <-c(Z%*%as.matrix(par))
	    	diff <- y-mu
		work<- exp(-(diff^2)/cons)
    		grad_p1<- -(4*cons2/cons)*c(diff*work)*Z
    
    		set1<-sorted_obs$IND[(n+1):(n+M.det),1]
    		set2<-sorted_obs$IND[(n+1):(n+M.det),2]
    		diff<-mu[set1]-mu[set2]
    		grad_p21<- -2*cons3*c(KX[(n+1):(n+M.det)]*diff*exp(-(diff^2)/cons4))*(Z[set1,]-Z[set2,])
    		diff <- y[set2]-mu[set1]
    		work<- exp(-(diff^2)/cons)
    		grad_p22<- -(4*cons2/cons)*c(KX[(n+1):(n+M.det)]*diff*work)*Z[set1,]
    		grad_p2<-grad_p21+grad_p22

    		use_X<- sample((n+M.det+1):KX.length,M.rand, replace=FALSE)
    		set1<-sorted_obs$IND[use_X,1]
   		set2<-sorted_obs$IND[use_X,2]
    
    		diff<-mu[set1]-mu[set2]
    		grad_p31<- -2*cons3*c(KX[use_X]*diff*exp(-(diff^2)/cons4))*(Z[set1,]-Z[set2,])
    		diff <- y[set2]-mu[set1]
    		work<- exp(-(diff^2)/cons)
    		grad_p32<- -(4*cons2/cons)*c(KX[use_X]*diff*work)*Z[set1,]
    		grad_p3<-grad_p31+grad_p32
  
   		grad<- (colSums(as.matrix(grad_p1))+2*colSums(as.matrix(grad_p2))+cons0*colSums(as.matrix(grad_p3)))/n
   		norm.grad <- norm.grad + sum(grad^2)
    		par<- par-stepsize*grad/sqrt(norm.grad)
   		store[i,]<-par
   		
   		grad_all <- grad_all+grad
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
  	store<- matrix(store[1:nsteps,],nrow=nsteps,ncol=length(par), byrow=FALSE)
  	#compute the estimator
  	store<-t( t(store)/sd.z)
 	if(nsteps>1) store<-apply(store,2,cumsum)/(1:nsteps) 
 	
  	#return the resuls
  	res$coefficients <- store[nsteps,]
  	res$trajectory<-store
 
 	return(res)
   
}

#Model: Linear Gaussian model par1 = beta, par2 = sd
#Algorithm: Stochastic gradient (adagrad)

LG_SGD_hat<-function(y, Z, intercept, sd.z, par1, par2, kernel, M.det, M.rand, bdwth, burnin, nsteps, stepsize, epsilon, sorted_obs, KX, eps_sg){

	#preparation of the output "res"
	res<-list()
	
   	#Initial parameter value
	if(length(par1)==1 &&  par1=="auto"){
		ols<-lm(y~Z-1)
		p1<-ols$coefficients
		if(par2=="auto"){
			p2<-summary(ols)$sigma
		}else{
			p2<-par2
		}
	}else{
		if(intercept==FALSE){
			p1<-par1
			if(par2=="auto"){
				ols<-lm(y~Z-1)
				p2<-summary(ols)$sigma
			}else{
				p2<-par2
			}
		}else{
			p1<-c(mean(y),par1)
			if(par2=="auto"){
				ols<-lm(y~Z-1)
				p2<-summary(ols)$sigma
			}else{
				p2<-par2
			}
		}
	}
	par<-c(p1,2*log(p2))
	
	#store parameters of the algorithms that will be used
	res$par1 <- p1
	res$par2 <- p2

	#initialization
	norm.grad<-epsilon
	n.par<-length(par)
	store<-matrix(0, nsteps,n.par)
	grad_all<-rep(0, n.par)
	log.eps<-log(eps_sg)
	res$convergence<-1
	n<-length(y)
   	cons<-((n-1)*n-2*M.det)/M.rand
   	KX.length<-length(KX)
   	d<-length(p1)

    	##running time
   	if(burnin>0){
   		for (i in 1:burnin) {
			mu <- c(Z%*%as.matrix(par[1:d]))
			var.y<- exp(par[d+1])
			y.sampled <-rnorm(n,mu, sqrt(var.y))
			y.sampled2 <-rnorm(n,mu, sqrt(var.y))
      
			ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
			diff<-y.sampled-mu
			grad_p1<-cbind( c(ker*diff/var.y)*Z, ker*(-var.y+diff^2)/var.y)
        
			set1<-sorted_obs$IND[(n+1):(n+M.det),1]
			set2<-sorted_obs$IND[(n+1):(n+M.det),2]
    			ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    			diff<-y.sampled[set1]-mu[set1]
    			grad_p2<- cbind( c(KX[(n+1):(n+M.det)]*ker*diff/var.y)*Z[set1,], KX[(n+1):(n+M.det)]*ker*(-var.y+diff^2)/var.y) 

    			use_X<- sample((n+M.det+1):KX.length,M.rand, replace=FALSE)
    			set1<-sorted_obs$IND[use_X,1]
    			set2<-sorted_obs$IND[use_X,2]
    
    			ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    			diff<-y.sampled[set1]-mu[set1]
    			grad_p3<- cbind( c(KX[use_X]*ker*diff/var.y)*Z[set1,], KX[use_X]*ker*(-var.y+diff^2)/var.y)    
    			grad<-  (colSums(grad_p1)+2*colSums(grad_p2)+cons*colSums(grad_p3))/n
    			norm.grad<- norm.grad + sum(grad^2)
    			grad_all <- grad_all+grad
    			par <-par-stepsize*grad/sqrt(norm.grad)
    		}
 	}
 	
 	for (i in 1:nsteps) {
		mu <- c(Z%*%as.matrix(par[1:d]))
		var.y<- exp(par[d+1])
		y.sampled <-rnorm(n,mu, sqrt(var.y))
		y.sampled2 <-rnorm(n,mu, sqrt(var.y))
      
		ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
		diff<-y.sampled-mu
		grad_p1<-cbind( c(ker*diff/var.y)*Z, ker*(-var.y+diff^2)/var.y)
        
		set1<-sorted_obs$IND[(n+1):(n+M.det),1]
		set2<-sorted_obs$IND[(n+1):(n+M.det),2]
    		ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    		diff<-y.sampled[set1]-mu[set1]
    		grad_p2<- cbind( c(KX[(n+1):(n+M.det)]*ker*diff/var.y)*Z[set1,], KX[(n+1):(n+M.det)]*ker*(-var.y+diff^2)/var.y) 

    		use_X<- sample((n+M.det+1):KX.length,M.rand, replace=FALSE)
    		set1<-sorted_obs$IND[use_X,1]
    		set2<-sorted_obs$IND[use_X,2]
    
    		ker<-K1d_dist(y.sampled[set1]-y.sampled2[set2],kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled[set1]-y[set2],kernel=kernel,bdwth=bdwth)
    		diff<-y.sampled[set1]-mu[set1]
    		grad_p3<- cbind( c(KX[use_X]*ker*diff/var.y)*Z[set1,], KX[use_X]*ker*(-var.y+diff^2)/var.y)    
    		grad<-  (colSums(grad_p1)+2*colSums(grad_p2)+cons*colSums(grad_p3))/n
    		norm.grad<- norm.grad + sum(grad^2)
    		par <-par-stepsize*grad/sqrt(norm.grad)
    		store[i,]<-par
    		
    		grad_all <- grad_all+grad
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
 	store<- matrix(store[1:nsteps,],nrow=nsteps,ncol=length(par), byrow=FALSE)
 	
 	#compute the estimator
   	store[,1:d]<-t(t(store[,1:d])/sd.z)
	store[,d+1]<-exp(store[, d+1]/2)  
	if(nsteps>1) store<-apply(store,2,cumsum)/(1:nsteps) 
	
	#return the results
	res$coefficients<-store[nsteps,1:d]
	res$phi<-store[nsteps,d+1]
	res$trajectory<-store
  	return(res)
}

#Model: Linear Gaussian model par1 = beta, par2 = sd
#Estimator: hat{theta} with a Gaussian kernel on Y
#Algorithm: stochastic gradient descent (Adagrad) with explicit computation of the gradient
#with respect to expectation of kernels.

LG_PartialSGD_hat<-function(y, Z, intercept, sd.z, par1, par2, M.det, M.rand, bdwth, burnin, nsteps, stepsize, epsilon, sorted_obs, KX, eps_sg){
   	#preparation of the output "res"
	res<-list()
	
   	#Initial parameter value
	if(length(par1)==1 &&  par1=="auto"){
		ols<-lm(y~Z-1)
		p1<-ols$coefficients
		if(par2=="auto"){
			p2<-summary(ols)$sigma
		}else{
			p2<-par2
		}
	}else{
		if(intercept==FALSE){
			p1<-par1
			if(par2=="auto"){
				ols<-lm(y~Z-1)
				p2<-summary(ols)$sigma
			}else{
				p2<-par2
			}
		}else{
			p1<-c(mean(y),par1)
			if(par2=="auto"){
				ols<-lm(y~Z-1)
				p2<-summary(ols)$sigma
			}else{
				p2<-par2
			}
		}
	}
	par<-c(p1, 2*log(p2))
	
	#store parameters of the algorithms that will be used
	res$par1 <- p1
	res$par2 <- p2

	#initialization 
	norm.grad<-epsilon
	n.par<-length(par)
	store<-matrix(0, nsteps, n.par)
	grad_all<-rep(0,n.par)
	log.eps<-log(eps_sg)
	res$convergence<-1
	n<-length(y)
	d<-length(p1)
   	cons1<-((n-1)*n-2*M.det)/M.rand
   	KX.length<-length(KX)
 	bdwth2<-bdwth^2
   
	#running time
	if(burnin>0){
   		for (i in 1:burnin) {
			mu<-c(Z%*%as.matrix(par[1:d]))
    			diff <- y-mu 
    			par22<-exp(par[d+1])
    			cons2<- 1/(2*par22+ bdwth2)
    			work<- exp(-(diff^2)*cons2)
    
    			cons2<- 1/(2*par22+ bdwth2)
    			cons3<- 1/(4*par22+ bdwth2)
    			g11<- c(-4*diff*work*cons2^{3/2})*Z
    			g12<- par22*(-2*cons3^{3/2}+2*work*cons2^{3/2}-4*(cons2^{5/2})*work*diff^2)
    			grad_p1<-cbind(g11,g12)
      
    			set1<-sorted_obs$IND[(n+1):(n+M.det),1]
   			set2<-sorted_obs$IND[(n+1):(n+M.det),2]
    			diff<-mu[set1]-mu[set2]
    			work<-exp(-(diff^2)*cons3) 
    			grad_p21<- cbind(-4*(cons3^{3/2})*c(KX[(n+1):(n+M.det)]*diff*work)*(Z[set1,]-Z[set2,]),c(-2*par22*KX[(n+1):(n+M.det)]*(2*work*cons3^{3/2}-4*(cons3^{5/2})*work*diff^2)))
    
    			diff <- y[set2]-mu[set1]
    			work<- exp(-(diff^2)*cons2)
    			grad_p22<- cbind(-(4*cons2^{3/2})*c(KX[(n+1):(n+M.det)]*diff*work)*Z[set1,], c(par22*KX[(n+1):(n+M.det)]*(2*work*cons2^{3/2}-4*(cons2^{5/2})*work*diff^2)))
    			grad_p2<-grad_p21+grad_p22

    			use_X<- sample((n+M.det+1):KX.length,M.rand, replace=FALSE)
    			set1<-sorted_obs$IND[use_X,1]
    			set2<-sorted_obs$IND[use_X,2]
    
    			diff<-mu[set1]-mu[set2]
    			work<-exp(-(diff^2)*cons3)
    			grad_p31<- cbind(-4*(cons3^{3/2})*c(KX[use_X]*diff*work)*(Z[set1,]-Z[set2,]),c(-2*par22*KX[use_X]*(2*work*cons3^{3/2}-4*(cons3^{5/2})*work*diff^2)))
    
    			diff <- y[set2]-mu[set1]
    			work<- exp(-(diff^2)*cons2)
    			grad_p32<- cbind(-(4*cons2^{3/2})*c(KX[use_X]*diff*work)*Z[set1,], c(par22*KX[use_X]*(2*work*cons2^{3/2}-4*(cons2^{5/2})*work*diff^2)))
    			grad_p3<-grad_p31+grad_p32
  
   			grad<- ( c(colSums(grad_p1))+2*colSums(grad_p2)+cons1*colSums(grad_p3))/n
   			norm.grad<- norm.grad + sum(grad^2)
   			grad_all <- grad_all+grad
    			par <- par-stepsize*grad/sqrt(norm.grad)
    		}
	}
	for (i in 1:nsteps) {
		mu<-c(Z%*%as.matrix(par[1:d]))
    		diff <- y-mu 
    		par22<-exp(par[d+1])
    		cons2<- 1/(2*par22+ bdwth2)
    		work<- exp(-(diff^2)*cons2)
    
    		cons2<- 1/(2*par22+ bdwth2)
    		cons3<- 1/(4*par22+ bdwth2)
    		g11<- c(-4*diff*work*cons2^{3/2})*Z
    		g12<- par22*(-2*cons3^{3/2}+2*work*cons2^{3/2}-4*(cons2^{5/2})*work*diff^2)
    		grad_p1<-cbind(g11,g12)
      
    		set1<-sorted_obs$IND[(n+1):(n+M.det),1]
   		set2<-sorted_obs$IND[(n+1):(n+M.det),2]
    		diff<-mu[set1]-mu[set2]
    		work<-exp(-(diff^2)*cons3) 
    		grad_p21<- cbind(-4*(cons3^{3/2})*c(KX[(n+1):(n+M.det)]*diff*work)*(Z[set1,]-Z[set2,]),c(-2*par22*KX[(n+1):(n+M.det)]*(2*work*cons3^{3/2}-4*(cons3^{5/2})*work*diff^2)))
    
    		diff <- y[set2]-mu[set1]
    		work<- exp(-(diff^2)*cons2)
    		grad_p22<- cbind(-(4*cons2^{3/2})*c(KX[(n+1):(n+M.det)]*diff*work)*Z[set1,], c(par22*KX[(n+1):(n+M.det)]*(2*work*cons2^{3/2}-4*(cons2^{5/2})*work*diff^2)))
    		grad_p2<-grad_p21+grad_p22

    		use_X<- sample((n+M.det+1):KX.length,M.rand, replace=FALSE)
    		set1<-sorted_obs$IND[use_X,1]
    		set2<-sorted_obs$IND[use_X,2]
    
    		diff<-mu[set1]-mu[set2]
    		work<-exp(-(diff^2)*cons3)
    		grad_p31<- cbind(-4*(cons3^{3/2})*c(KX[use_X]*diff*work)*(Z[set1,]-Z[set2,]),c(-2*par22*KX[use_X]*(2*work*cons3^{3/2}-4*(cons3^{5/2})*work*diff^2)))
    
    		diff <- y[set2]-mu[set1]
    		work<- exp(-(diff^2)*cons2)
    		grad_p32<- cbind(-(4*cons2^{3/2})*c(KX[use_X]*diff*work)*Z[set1,], c(par22*KX[use_X]*(2*work*cons2^{3/2}-4*(cons2^{5/2})*work*diff^2)))
    		grad_p3<-grad_p31+grad_p32
  
   		grad<- ( c(colSums(grad_p1))+2*colSums(grad_p2)+cons1*colSums(grad_p3))/n
   		norm.grad<- norm.grad + sum(grad^2)
    		par <- par-stepsize*grad/sqrt(norm.grad)
    		store[i,]<-par
    		
    		grad_all <- grad_all+grad
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
  	store[,1:d]<-t(t(store[,1:d])/sd.z)
	store[,d+1]<-exp(store[, d+1]/2)   
 	if(nsteps>1)store<-apply(store,2,cumsum)/(1:nsteps) 
    
 	#return the results
  	res$coefficients<-store[nsteps,1:d]
  	res$phi<-store[nsteps,d+1]
  	res$trajectory<-store  
 	return(res)
}

