
# model: Logistic regression model par1 = beta

Logistic_tilde<-function(y, Z, intercept, sd.z, par1, kernel, bdwth, nsteps, alpha, eps_gd){

	#preparation of the output "res"
  	res<-list()
  	
  	#Initial parameter value
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
   	v<-1
   	n<-length(y)
   	store<-matrix(0, nsteps, length(par))
	K00<-K1d_dist(rep(0,n),kernel=kernel,bdwth=bdwth)
   	K01<-K1d_dist(rep(1,n),kernel=kernel,bdwth=bdwth)
   	K0y<-K1d_dist(rep(0,n)-y,kernel=kernel,bdwth=bdwth)
   	K1y<-K1d_dist(rep(1,n)-y,kernel=kernel,bdwth=bdwth)
   
   	#running time
   	mu<-Z%*%as.matrix(par)
   	p<-1/(1+exp(-mu))
   
   	f1<- mean((p^2+(1-p)^2)*K00+2*p*(1-p)*K01-2*p*K1y-2*(1-p)*K0y)
   	g11<- c(K00*(4*(1-p)*p^2-2*p*(1-p))+ K01*(p*(1-p)-2*(1-p)*p^2))*Z
   	g12<- c(p*(1-p)*K1y-p*(1-p)*K0y)*Z
   	g1<- apply(g11-2*g12,2,mean)
   	ng<-sum(g1^2)
   	
   	res$convergence<-1
   	for(i in 1:nsteps){ 
		mu<-Z%*%as.matrix(par-v*g1)
		p<-1/(1+exp(-mu))
   		f2<- mean((p^2+(1-p)^2)*K00+2*p*(1-p)*K01-2*p*K1y-2*(1-p)*K0y)
   
        	while(f2>f1-0.5*v*ng){
           		v<-alpha*v
           		mu<-Z%*%as.matrix(par-v*g1)
   	   		p<-1/(1+exp(-mu))
   	    		f2<- mean((p^2+(1-p)^2)*K00+2*p*(1-p)*K01-2*p*K1y-2*(1-p)*K0y)
       	 }
   		par<-par-v*g1
   		store[i,]<-par
   	
   		g11<- c(K00*(4*(1-p)*p^2-2*p*(1-p))+ K01*(p*(1-p)-2*(1-p)*p^2))*Z
       	g12<- c(p*(1-p)*K1y-p*(1-p)*K0y)*Z
        	g1<- apply(g11-2*g12,2,mean)
   		ng<-sum(g1^2)
   		if(log(abs(f2-f1))-log(abs(f1))<log(eps_gd)){
			res$convergence<-0
   			break
   		}
   		f1<-f2
  	}
  	nsteps<-i
	store<-store[1:i,]
  	
  	#compute the estimator
  	store<-t( t(store)/sd.z)
	
  	#return the results
  	res$coefficients<-store[nsteps,]
  	res$trajectory<-store
  	res$SG<-FALSE
  	
  	return(res)  
  	
}
