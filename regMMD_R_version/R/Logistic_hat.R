
#' @importFrom stats glm


# model: Logistic regression par1 = beta

Logistic_hat<-function(y, Z, intercept, sd.z, par1, kernel, M.det, M.rand, bdwth, burnin, nsteps,  stepsize, epsilon, sorted_obs, KX, eps_sg){
	#preparation of the output "res"
  	res<-list()
  	
  	#Initial parameter valu
   	if(length(par1)==1 &&  par1=="auto") {
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
  	
	#initialization
	norm.grad<-epsilon
	n<-length(y)
   	cons<-((n-1)*n-2*M.det)/M.rand
   	KX.length<-length(KX)
	
	#running time
   	n.par<-length(par)
	store<-matrix(0, nsteps, n.par)
	grad_all<-rep(0,n.par)
	log.eps<-log(eps_sg)
	res$convergence<-1
   	
   	K00<-K1d_dist(rep(0,n),kernel=kernel,bdwth=bdwth)
   	K01<-K1d_dist(rep(1,n),kernel=kernel,bdwth=bdwth)
  	K0y<-K1d_dist(rep(0,n)-y,kernel=kernel,bdwth=bdwth)
   	K1y<-K1d_dist(rep(1,n)-y,kernel=kernel,bdwth=bdwth)
   
   	if(burnin>0){
   		for (i in 1:burnin) {
     			mu<-Z%*%as.matrix(par)
     			p<-1/(1+exp(-mu))
     			g11<- c(K00*(4*(1-p)*p^2-2*p*(1-p))+ K01*(p*(1-p)-2*(1-p)*p^2))*Z
     			g12<- c(p*(1-p)*K1y-p*(1-p)*K0y)*Z
     			grad_p1<- g11-2*g12
    
      
     			set1<-sorted_obs$IND[(n+1):(n+M.det),1]
     			set2<-sorted_obs$IND[(n+1):(n+M.det),2]
    			p<-1/(1+exp(-mu[set1]))
     			dp<- c(p*(1-p))*Z[set1,] 
     			pp<-1/(1+exp(-mu[set2]))
     			dpp<- c(pp*(1-pp))*Z[set2,]
     			A<-K00[set1]*(2*pp*dp+2*p*dpp-dp-dpp)
     			B<-K01[set1]*(dp+dpp-2*p*dpp-2*dp*pp)
     			C<--2*K1y[set1]*dp+2*K0y[set1]*dp
     			grad_p2<-KX[(n+1):(n+M.det)]*(A+B+C)
    

     			use_X<- sample((n+M.det+1):KX.length,M.rand, replace=FALSE)
     			set1<-sorted_obs$IND[use_X,1]
     			set2<-sorted_obs$IND[use_X,2]
    
    			p<-1/(1+exp(-mu[set1]))
     			dp<- c(p*(1-p))*Z[set1,] 
     			pp<-1/(1+exp(-mu[set2]))
     			dpp<- c(pp*(1-pp))*Z[set2,]
    			A<-K00[set1]*(2*pp*dp+2*p*dpp-dp-dpp)
     			B<-K01[set1]*(dp+dpp-2*p*dpp-2*dp*pp)
     			C<--2*K1y[set1]*dp+2*K0y[set1]*dp
     			grad_p3<-KX[use_X]*(A+B+C)
  
     			grad<- ( c(colSums(as.matrix(grad_p1)))+2*colSums(as.matrix(grad_p2))+cons*colSums(as.matrix(grad_p3)))/n
 			grad_all<-grad_all+grad
 		
     			norm.grad<- norm.grad + sum(grad^2)
     			par <- par-stepsize*grad/sqrt(norm.grad)
     		}
  	}
  	for (i in 1:nsteps) {
     		mu<-Z%*%as.matrix(par)
     		p<-1/(1+exp(-mu))
     		g11<- c(K00*(4*(1-p)*p^2-2*p*(1-p))+ K01*(p*(1-p)-2*(1-p)*p^2))*Z
     		g12<- c(p*(1-p)*K1y-p*(1-p)*K0y)*Z
     		grad_p1<- g11-2*g12
    
      
     		set1<-sorted_obs$IND[(n+1):(n+M.det),1]
     		set2<-sorted_obs$IND[(n+1):(n+M.det),2]
    		p<-1/(1+exp(-mu[set1]))
     		dp<- c(p*(1-p))*Z[set1,] 
     		pp<-1/(1+exp(-mu[set2]))
     		dpp<- c(pp*(1-pp))*Z[set2,]
     		A<-K00[set1]*(2*pp*dp+2*p*dpp-dp-dpp)
     		B<-K01[set1]*(dp+dpp-2*p*dpp-2*dp*pp)
     		C<--2*K1y[set1]*dp+2*K0y[set1]*dp
     		grad_p2<-KX[(n+1):(n+M.det)]*(A+B+C)
    

     		use_X<- sample((n+M.det+1):KX.length,M.rand, replace=FALSE)
     		set1<-sorted_obs$IND[use_X,1]
     		set2<-sorted_obs$IND[use_X,2]
    
    		p<-1/(1+exp(-mu[set1]))
     		dp<- c(p*(1-p))*Z[set1,] 
     		pp<-1/(1+exp(-mu[set2]))
     		dpp<- c(pp*(1-pp))*Z[set2,]
    		A<-K00[set1]*(2*pp*dp+2*p*dpp-dp-dpp)
     		B<-K01[set1]*(dp+dpp-2*p*dpp-2*dp*pp)
     		C<--2*K1y[set1]*dp+2*K0y[set1]*dp
     		grad_p3<-KX[use_X]*(A+B+C)
  
     		grad<- ( c(colSums(as.matrix(grad_p1)))+2*colSums(as.matrix(grad_p2))+cons*colSums(as.matrix(grad_p3)))/n
 		grad_all<-grad_all+grad
 		
     		norm.grad<- norm.grad + sum(grad^2)
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
  	#Compute the estimator
      	store<-t( t(store)/sd.z)
	if(nsteps>1)store<-apply(store,2,cumsum)/(1:nsteps) 
    
  	#return the results
  	res$coefficients<-store[nsteps,]
 	res$trajectory<-store
  	return(res)
   
}


