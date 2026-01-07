##function that computes the pair-wise distances and sort them by increasing order
sort_obs<- function(X){ ##TO DO: Make it more efficient
   Z<-X
   if(is.matrix(Z)){
        n <- dim(Z)[1]
        d <- dim(Z)[2]
   }else{
        n<-length(c(Z))
        d<-1
        Z<-as.matrix(Z)
   }
   Zi<-matrix(0,nrow=n^2,ncol=d)
   Zj<- matrix(0,nrow=n^2,ncol=d)

   m<-n*(n+1)/2
   distances<-matrix(0,nrow=m,ncol=1)
   indices<-matrix(0,m,2)
   k<-1
   for(i in 1:n){
     for(j in i:n){
         distances[k]<-sqrt(sum((Z[i,]-Z[j,])^2))
         indices[k,]<-c(i,j)
         k<-k+1
     }
   }
   J<-order(distances)
   return(list(DIST=distances[J], IND=indices[J,])) 
}



K1d_dist <- function(u, kernel, bdwth=1) {
	if(kernel=="Gaussian") {
    		return(exp(-(u/bdwth)^2))
  	}else if(kernel=="Laplace") {
   		return(exp(-abs(u/bdwth)))
 	}else if(kernel=="Cauchy") {
    		return(1/(2+(u/bdwth)^2))
  	}
}


K1d = function(x, y, kernel="Laplace", bdwth=1) {
  u = outer(x, y, FUN="-")/bdwth
  if (kernel=="Gaussian") {
    return(exp(-u^2))
  } else if (kernel=="Laplace") {
    return(exp(-abs(u)))
  } else if (kernel=="Cauchy") {
    return(1/(2+u^2))
  }
}

K1d.diff = function(x, y, kernel="Laplace", bdwth=1) {
  u = outer(x, y, FUN="-")/bdwth
  if (kernel=="Gaussian") {
    return(-2*u*exp(-u^2))
  } else if (kernel=="Laplace") {
    return(-sign(u)*exp(-abs(u)))
  } else if (kernel=="Cauchy") {
    return(-2*u/((2+u^2)^2))
  }
}

Kmd = function(x, y, kernel="Laplace", bdwth=1) {
  n = dim(x)[1]
  u = as.matrix(dist(rbind(x,y), diag=TRUE,upper=TRUE,method="euclidean"))[1:n,(n+1):(2*n)]/bdwth
  if (kernel=="Gaussian") {
    return(exp(-u^2))
  } else if (kernel=="Laplace") {
    return(exp(-abs(u)))
  } else if (kernel=="Cauchy") {
    return(1/(2+u^2))
  }
}

Kmd.diff = function(x, y, kernel="Laplace", bdwth=1) {
  diff = (matrix(data=y,nrow=dim(x)[1],ncol=dim(x)[2],byrow=TRUE)-x)/bdwth
  w = ((diff^2)%*%rep(1,dim(x)[2]))[,1]
  if (kernel=="Gaussian") {
    return(-2*diff*exp(-w))
  } else if (kernel=="Laplace") {
    nrm = sqrt(w)
    dir = diff/nrm
    dir[is.nan(dir)] = 0
    return(-dir*exp(-nrm))
  } else if (kernel=="Cauchy") {
    return(-2*diff/((2+w)^2))
  }
}
