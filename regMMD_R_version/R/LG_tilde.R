#' @importFrom stats lm
#' @importFrom stats rnorm
 


#model: Linear Gaussian model par1 = beta, fixed: par2 = sd
#Algorithm: stochastic gradient descent (Adagrad)

LG_SGD_loc_tilde<-function(y, Z, intercept, sd.z, par1, par2, kernel, bdwth, burnin,  nsteps, stepsize, epsilon, eps_sg) {

	#preparation of the output "res"
	res<-list()
   
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
	res$par1<- par
	res$par2<- par2
	res$phi<- par2
	
	#initialization
	n<-length(y)
	norm.grad<- epsilon
	n.par<-length(par)
	store<-matrix(0, nsteps, n.par)
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
			grad<- apply( c((y.sampled-mu)*ker)*Z,2,mean) 
			norm.grad <- norm.grad + sum(grad^2)
			grad_all <- grad_all+grad
			par <- par-stepsize*grad/sqrt(norm.grad)
			
		}
	}
	for (i in 1:nsteps) {
		mu <- c(Z%*%as.matrix(par))
		y.sampled <-rnorm(n,mu, par2)
		y.sampled2 <-rnorm(n,mu, par2)  
		ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
		grad<- apply( c((y.sampled-mu)*ker)*Z,2,mean) 
		norm.grad <- norm.grad + sum(grad^2)
		grad_all <- grad_all+grad
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
	store<-apply(store,2,cumsum)/(1:nsteps) 
	
	#return the results
  	res$coefficients <- store[nsteps,]
  	res$trajectory<-store
  	return(res) 
}



#model: Linear Gaussian model par1 = beta, fixed: par2 = sd
#Estimator: tilde{theta} with a Gaussian kernel on Y
#Algorithm: Gradient descent with backtraking line search
 

LG_GD_loc_tilde<-function(y, Z, intercept, sd.z, par1, par2, bdwth, nsteps, alpha, eps_gd) {
   
	#preparation of the output "res"
	res <- list(par1=par1, par2=par2)
   
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
	res$par1<- par
	res$par2<- par2
	res$phi<- par2

	#initialization
	store<-matrix(0, nsteps, length(par))
 	cons<-2*par2^2+ bdwth^2
   
   	#running time
	diff <- y-c(Z%*%as.matrix(par))
	work<- exp(-(diff^2)/cons)
	f1<-  -mean(work)
	g1<-  -(2/cons)*apply(c(diff*work)*Z,2,mean)
	ng<-sum(g1^2)
	log.eps<-log(eps_gd)
	
	res$convergence<-1
	for(i in 1:nsteps) { 
		v<-1
		diff <- y-c(Z%*%as.matrix(par-v*g1))
		work<- exp(-(diff^2)/cons)
		f2<- -mean(work)
		while(f2>f1-0.5*v*ng){
			v<-alpha*v
			diff <- y-c(Z%*%as.matrix(par-v*g1))
			work<- exp(-(diff^2)/cons)
			f2<- -mean(work)
		}
   		par<-par-v*g1
   		store[i,]<-par
   	
   		g1<-  -(2/cons)*apply(c(diff*work)*Z,2,mean)
   		ng<-sum(g1^2)
   		
   		if(log(abs(f2-f1))-log(abs(f1))< log.eps){
   			res$convergence<-0
   			break
   		}
   		f1<-f2
  	}
  	nsteps<-i
  	store<- matrix(store[1:nsteps,],nrow=nsteps,ncol=length(par), byrow=FALSE)
  	
  	#compute the estimator   
  	store<-t( t(store)/sd.z)
  
  	#return the results
  	res$coefficients<-store[nsteps,]
  	res$trajectory<-store
  	return(res)
   
}


#model: Linear Gaussian model par1 = beta, par2 = sd
#algorithm: Stochastic gradient (adagrad)

LG_SGD_tilde<-function(y, Z, intercept, sd.z, par1, par2, kernel, bdwth, burnin, nsteps, stepsize, epsilon, eps_sg){
 
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
	res$par1<- p1
	res$par2<- p2
  
	#initialization
	d<-length(p1)
	n<-length(y)
	norm.grad<-epsilon
	n.par<-length(par)
	store<-matrix(0, nsteps,n.par)
	grad_all<-rep(0,n.par)
	log.eps<-log(eps_sg)
	res$convergence<- 1
	
	#running time
	if(burnin>0){
		for (i in 1:burnin) {
			mu<-c(Z%*%as.matrix(par[1:d]))
			var.y<- exp(par[d+1])
			y.sampled <- rnorm(n,mu, sqrt(var.y)) 
			y.sampled2 <- rnorm(n,mu, sqrt(var.y))
			ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
			diff<-y.sampled-mu
			grad<-apply(cbind( c(ker*diff/var.y)*Z, ker*(-var.y+diff^2)/var.y),2,mean)
			grad_all<- grad_all+grad
			norm.grad<-norm.grad + sum(grad^2)
			par<- par-stepsize*grad/sqrt(norm.grad)
		}
	}
	for (i in 1:nsteps) {
		mu<-c(Z%*%as.matrix(par[1:d]))
		var.y<- exp(par[d+1])
		y.sampled <- rnorm(n,mu, sqrt(var.y)) 
		y.sampled2 <- rnorm(n,mu, sqrt(var.y))
		ker<- K1d_dist(y.sampled-y.sampled2,kernel=kernel,bdwth=bdwth)-K1d_dist(y.sampled-y,kernel=kernel,bdwth=bdwth)
		diff<-y.sampled-mu
		grad<-apply(cbind( c(ker*diff/var.y)*Z, ker*(-var.y+diff^2)/var.y),2,mean)
		grad_all<- grad_all+grad
		norm.grad<-norm.grad + sum(grad^2)
		par<- par-stepsize*grad/sqrt(norm.grad)
		store[i,]<-par
		if(is.nan(mean(grad_all))){
			res$convergence<- -1
			break
		}
		g1<-sqrt(  sum((grad_all/(burnin+i))^2) )/n.par
		
		if(log(g1)< log.eps){
  				res$convergence<- 0
  				break
  		}
	}
	nsteps<-i
	store<- matrix(store[1:nsteps,],nrow=nsteps,ncol=n.par, byrow=FALSE)
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



#model: Linear Gaussian model par1 = beta, par2 = sd
#Estimator: tilde{theta} with a Gaussian kernel on Y
#Algorithm: Gradient descent with backtraking line search

LG_GD_tilde<-function(y, Z, intercept, sd.z, par1, par2, bdwth, nsteps, alpha, eps_gd){

	#preparation of the output "res"
	res<-list(par1=par1, par2=par2, bdwth=bdwth)

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
	res$par1<- p1
	res$par2<- p2

	#initialization
	v<-1
	d<-length(p1)
	n<-length(y)
   	store<-matrix(0, nsteps, d+1)
   	bdwth2<-bdwth^2
   
	#running time
	diff <- y-c(Z%*%as.matrix(par[1:d]))
	sigma2<-exp(par[d+1])
	new_var<-2*sigma2+bdwth2
	work<- exp(-(diff^2)/new_var)
	f1<- 1/sqrt(bdwth2+4*sigma2)-2*mean(work)/sqrt(new_var)
	g11<- apply( c(-4*diff*work*new_var^{-3/2})*Z,2,mean)
	g12<-sigma2*(-2*(bdwth2+4*sigma2)^{-3/2}+2*mean(work)*new_var^{-3/2}-4*mean(work*diff^2)*new_var^{-5/2})
	g1<-  c(g11,g12)
	ng<-sum(g1^2)
	log.eps<-log(eps_gd)
   
   	res$convergence<-1
	for(i in 1:nsteps) { 
		diff <- y-c(Z%*%as.matrix(par[1:d]-v*g1[1:d]))
		sigma2<-exp(par[d+1]-v*g1[d+1])
		new_var<-2*sigma2+bdwth2
		work<- exp(-(diff^2)/new_var)
		f2<- 1/sqrt(bdwth2+4*sigma2)-2*mean(work)/sqrt(new_var)
		while(f2>f1-0.5*v*ng){
			v<-alpha*v
			diff <- y-c(Z%*%as.matrix(par[1:d]-v*g1[1:d]))
			sigma2<-exp(par[d+1]-v*g1[d+1])
			new_var<-2*sigma2+bdwth2
			work<- exp(-(diff^2)/new_var)
			f2<- 1/sqrt(bdwth2+4*sigma2)-2*mean(work)/sqrt(new_var)
		}
		par<-par-v*g1
		store[i,]<-par
   	
		g11<- apply( c(-4*diff*work*new_var^{-3/2})*Z,2,mean)
		g12<-sigma2*(-2*(bdwth2+4*sigma2)^{-3/2}+2*mean(work)*new_var^{-3/2}-4*mean(work*diff^2)*new_var^{-5/2})
		g1<-  c(g11,g12)
		ng<-sum(g1^2)
		
		if(log(abs(f2-f1))-log(abs(f1))< log.eps){
			res$convergence<-0
   			break
   		}
   		f1<-f2
	}
	nsteps<-i
	store<- matrix(store[1:nsteps,],nrow=nsteps,ncol=length(par), byrow=FALSE)
   	
   	#Compute the estimator
	store[,1:d]<-t(t(store[,1:d])/sd.z) 
  	store[,d+1]<-exp(store[, d+1]/2)
  	    
  	#return the results
  	res$coefficients<-store[nsteps,1:d]
  	res$phi<-store[nsteps,d+1]
 	res$trajectory<-store
 	
  	return(res)
   
}

