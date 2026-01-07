
#' MMD regression
#'
#' @importFrom Rdpack reprompt 
#' @importFrom stats dist median sd
#' @description Fits a regression model to the data, using the robust procedure based on maximum mean discrepancy (MMD) minimization introduced and studied in \insertCite{MMD;textual}{regMMD}. 
#'
#'
#' @usage  mmd_reg(y, X, model, intercept, par1, par2, kernel.y, kernel.x, bdwth.y, bdwth.x, 
#'         control= list())
#'
#' @param y Response variable. Must be a vector of length  \eqn{n\geq 3}.
#' @param X Design matrix. \code{X} must be either a matrix of dimension \eqn{n\times p} or a vector of size  \eqn{n},  where \eqn{n} is the size of \code{y}.
#' @param model Regression model to be fitted to the data. By default, the linear regression model with \eqn{\mathcal{N}_1(0,\phi^2)} error terms is used.  See details for the list of available models.
#' @param intercept If \code{intercept=TRUE} an intercept is added to the model, while no intercept is added if \code{intercept=FALSE}. By default, \code{intercept=TRUE}.
#' @param par1 Values of the regression coefficients of the model used as starting values to numerically optimize the objective function. \code{par1} must be either a vector of size \eqn{p}, with \eqn{p} the number of columns of \code{X} (or with \eqn{p=1} if \code{X} is a vector), or equal to \code{"auto"}, in which case a non-robust estimate of the regression coefficients of the model is used as starting values. By default, \code{par1="auto"}.
#' @param par2 A value for the auxilliary parameter \eqn{\phi} of the model. \code{par2} needs to be specified only if relevant (see details for the list of models having an  auxilliary parameter \eqn{\phi}).  If the model assumes that \eqn{\phi}  is known (see details) then \code{par2} must be a strictly positive real number and the model is estimated with \eqn{\phi=}\code{par2}. If the model assumes that \eqn{\phi} is unknown (see details)  then the value specified by   \code{par2}  is used as starting value to numerically optimize the objective function. For such models \code{par2} must be either a strictly positive real number or equal to \code{"auto"}, in which case a non-robust estimate of \eqn{\phi} is used as starting value. By default, \code{par2="auto"}.
#' @param kernel.y Kernel applied on the response variable. Available options for  \code{kernel.y} are \code{"Gaussian"} (Gaussian kernel), \code{"Laplace"} (Laplace, or exponential, kernel) and \code{"Cauchy"} (Cauchy kernel). By default, \code{kernel.y="Gaussian"} for the linear regression model and \code{kernel.y="Laplace"} for the other models.
#' @param kernel.x Kernel applied on the explanatory variables. Available options for  \code{kernel.x} are \code{"Gaussian"} (Gaussian kernel), \code{"Laplace"} (Laplace, or exponential, kernel) and \code{"Cauchy"} (Cauchy kernel). By default, \code{kernel.x="Laplace"}. 
#' @param bdwth.y Bandwidth parameter for the kernel \code{kernel.y}. \code{bdwth.y} must be eiter a strictly positive real number or equal to \code{"auto"}, in which case the median heuristic  is used to select the bandwidth parameter of \code{kernel.y} (see details). By default, \code{bdwth.y="auto"}.
#' @param bdwth.x Bandwidth parameter for the kernel \code{kernel.x}. \code{bdwth.x} must be either a non-negative real number or equal to \code{"auto"}, in which case a rescaled version of the median heuristic is used to specify the bandwidth parameter  of \code{kernel.x} (see details). By default,   \code{bdwth.x}=0. Remark: for computational reasons, for large dataset (i.e.~when the sample size is bigger than a few thousands) it is recommended to choose \code{bdwth.x}=0 (see details).
#' @param control A \code{list} of control parameters for the numerical optimization of the objective function. See details.
#' @details  
#'
#' Available options for \code{model} are:
#' \describe{
#' \item{\code{"linearGaussian"}}{Linear regression model with \eqn{\mathcal{N}_1(0,\phi^2)} error terms, with \eqn{\phi} unknown.}
#' \item{\code{"linearGaussian.loc"}}{Linear regression model with \eqn{\mathcal{N}_1(0,\phi^2)} error terms, with \eqn{\phi} known.}
#' \item{\code{"gamma"}}{Gamma regression model with unknown shape parameter \eqn{\phi}. The inverse function is used as link function.}
#' \item{\code{"gamma.loc"}}{Gamma regression model with known shape parameter \eqn{\phi}. The inverse function is used as link function.}
#' \item{\code{"beta"}}{Beta regression model with unknown precision parameter \eqn{\phi}. The logistic function is used as link function.}
#' \item{\code{"beta.loc"}}{Beta regression model with  known precision parameter \eqn{\phi}. The logistic function is used as link function.}
#' \item{\code{"logistic"}}{Logistic regression model.}
#' \item{\code{"exponential"}}{Exponential regression.}
#' \item{\code{"poisson"}}{Poisson regression model.}
#'}
#'
#' When \code{bdwth.x}>0 the function \code{mmd_reg} computes the estimator \eqn{\hat{\theta}_n} introduced in \insertCite{MMD;textual}{regMMD}. When \code{bdwth.x}=0  the function \code{mmd_reg} computes the estimator \eqn{\tilde{\theta}_n} introduced in \insertCite{MMD;textual}{regMMD}. The former estimator has stronger theoretical properties but is more expensive to compute (see below).
#'
#' When \code{bdwth.x}=0 and \code{model} is  \code{"linearGaussian"}, \code{"linearGaussian.loc"} or \code{"logistic"}, the objective function and its gradient can be computed on \eqn{\mathcal{O}(n)} operations, where \eqn{n} is the sample size (i.e. the dimension of \code{y}). In this case, gradient descent with backtraking line search is used to perform the minimizatiom. The algorithm stops  when the maximum number of iterations \code{maxit} is reached, or as soon as the change in the objective function is less than \code{eps_gd} times the current function value. In the former case, a warning message is generated. By defaut, \code{maxit}=\eqn{5\times 10^4} and  \code{eps_gd=sqrt(.Machine$double.eps)}, and the value of these two parameters can be changed using the \code{control} argument of \code{mmd_reg}.
#'
#' When \code{bdwth.x}>0 and \code{model} is  \code{"linearGaussian"}, \code{"linearGaussian.loc"} or \code{"logistic"}, the objective function and its gradient can be computed on \eqn{\mathcal{O}(n^2)} operations. To reduce the computational cost the objective function is minimized using norm adagrad \insertCite{duchi2011adaptive}{regMMD}, an adaptive step size stochastic gradient algorithm. Each iteration of the algorithm requires \eqn{\mathcal{O}(n)} operations. However,  the algorithm has an intialization step that requires \eqn{\mathcal{O}(n^2)} operations and has a memory requirement of size \eqn{\mathcal{O}(n^2)}.
#'
#' When   \code{model} is not in \code{c("linearGaussian", "linearGaussian.loc", "logistic")},  the objective function and its gradient cannot be computed explicitly and the minimization is performed using norm adagrad. The cost per iteration of the algorithm is \eqn{\mathcal{O}(n)} but, for  \code{bdwth.x}>0, the memory requirement and the initialization cost are both of size \eqn{\mathcal{O}(n^2)}.
#'
#' When adagrad is used, \code{burnin} iterations  are performed as a warm-up step.  The algorithm then stops when \code{burnin}+\code{maxit} iterations are performed, or as soon as the norm of the average value of the gradient evaluations computed in all the previous iterations is less than \code{eps_sg}. A warning message is generated if the maximum number of iterations is reached. By default, \code{burnin}=\eqn{10^3}, \code{nsteps}=\eqn{5\times 10^4} and \code{eps_sg}=\eqn{10^{-5}} and  the value of these three parameters can be changed using the \code{control} argument of  \code{mmd_reg}.
#'
#' If \code{bdwth.y="auto"} then the value of the bandwidth parameter of \code{kernel.y} is equal to \eqn{H_n/\sqrt{2}}  with \eqn{H_n}   the median value of the set \eqn{ \{|y_i-y_j|\}_{i,j=1}^n}, where \eqn{y_i} denote the ith component of \code{y}. This definition of \code{bdwth.y} is motivated by the results in \insertCite{garreau2017large;textual}{regMMD}. If \eqn{H_n=0} the bandwidth parameter of \code{kernel.y} is set to 1.
#'
#' If \code{bdwth.x="auto"} then the value of the bandwidth parameter of \code{kernel.x} is equal to \eqn{0.01H_n/\sqrt{2}}  with \eqn{H_n} is the median value of the set \eqn{ \{\|x_i-x_j\|\}_{i,j=1}^n}, where \eqn{x_i} denote the ith row of the design matrix \code{X}. If \eqn{H_n=0} the bandwidth parameter of \code{kernel.x} is set to 1.
#'
#'The \code{control} argument is a list  that can supply any of the following components:
#' \describe{
#' \item{rescale:}{If \code{rescale=TRUE} the (non-constant) columns of \code{X} are rescalled before perfoming the optimization, while if \code{rescale=FLASE} no rescaling is applied. By default \code{rescale=TRUE}.}
#' \item{burnin}{A non-negative integer.}
#' \item{eps_gd}{A non-negative real number.}
#' \item{eps_sg}{A non-negative real number.}
#' \item{maxit}{A integer strictly larger than 2.}
#' \item{stepsize}{Scaling constant for the step-sizes used by adagrad. \code{stepsize} must be a stictly positive number and by default \code{stepsize}=1.}
#' \item{trajectory:}{If \code{trajectory=TRUE} then the parameter value obtained at the end  of  each iteration  (after the burn-in perdiod for adagrad)  is returned. By default, \code{trajectory=TRUE} and   \code{trajectory} is automatically set to \code{TRUE} if the maximum number of iterations is reached.}
#' \item{epsilon}{Parameter used in adagrad to avoid numerical errors in the computation of the step-sizes. \code{epsilon} must be a strictly positive real number and by default \code{epsilon}=\eqn{10^{-4}}.}
#' \item{alpha}{Parameter for the backtraking line search. \code{alpha} must be a real number in \eqn{(0,1)} and by default \code{alpha}=0.8.}
#' \item{c_det}{Parameter used to control the computational cost of the algorithm when \code{gamma.x}\eqn{>0}, see the Suplementary material in \insertCite{MMD;textual}{regMMD} for mode details. \code{c_det} must be a real number in \eqn{(0,1)} and by default \code{c_det}=0.2.}
#' \item{c_rand}{Parameter used to control the computational cost of the algorithm when \code{bdwth.x}\eqn{>0}, see the Suplementary material in \insertCite{MMD;textual}{regMMD} for mode details. \code{c_rand} must be a real number in \eqn{(0,1)} and by default \code{c_rand}=0.1.}
#'}
#'
#' @return  \code{MMD_reg} returns an object of \code{class} \code{"regMMD"}.
#'
#' The function \code{summary}  can be used to print the results.
#'
#' An object of class \code{regMMD} is a list containing the following components:
#'   \item{coefficients}{Estimated regression coefficients.}
#'   \item{intercept}{If \code{intercept}=TRUE an intercept has been added to model, if \code{intercept}=FALSE no intercept has been added.}
#'   \item{phi}{If relevant (see details), either the estimated value of the \eqn{\phi} parameter of model, or the value of \eqn{\phi} used to fit the model if \eqn{\phi} is assumed to be known.}
#'   \item{kernel.y}{Kernel applied on the response variable used to fit the model.}
#'   \item{bdwth.y}{Value of the bandwidth for the kernel applied on the response variable used to fit the model.}
#'   \item{kernel.x}{Kernel applied on the explanatory variables used to fit the model.}
#'   \item{bdwth.x}{Value of the bandwidth for the kernel applied on the explanatory variables used to fit the model.}
#'   \item{par1}{Value of the parameter \code{par1} used to fit the model.}
#'   \item{par2}{Value of  parameter \code{par2} used to fit the model.}
#'   \item{trajectory}{If the control parameter \code{trajectory=TRUE}, \code{trajectory} is a matrix containing the parameter values obtained at the end of each iteration of the optimization algorithm.}
#'
#' @examples
#' #Simulate data
#' n<-1000
#' p<-4
#' beta<-1:p
#' phi<-1
#' X<-matrix(data=rnorm(n*p,0,1),nrow=n,ncol=p)
#' data<-1+X%*%beta+rnorm(n,0,phi)
#'
#' ##Example 1: Linear Gaussian model 
#' y<-data
#' Est<-mmd_reg(y, X)
#' summary(Est)
#'
#' ##Example 2: Logisitic regression model
#' y<-data
#' y[data>5]<-1
#' y[data<=5]<-0
#' Est<-mmd_reg(y, X, model="logistic")
#' summary(Est)
#' Est<-mmd_reg(y, X, model="logistic", bdwth.x="auto")
#' summary(Est)
#' @references
#'   \insertAllCited{}
#' @export



mmd_reg<-function(y, X, model="linearGaussian", intercept=TRUE, par1="auto", par2="auto", kernel.y="auto", kernel.x="auto", bdwth.y="auto", bdwth.x=0,  
		 	control= list()){
		 	

	#list of models that are implemented
	MODEL_LIST<-c("linearGaussian", "linearGaussian.loc", "logistic", "gamma", "gamma.loc", "exponential", "poisson", "beta", "beta.loc")
	#list of models having a fixed par2 parameter
	MODEL_LIST2<-c("linearGaussian.loc", "gamma.loc", "beta.loc")
	#list of models having a par2 parameter to estimate
	MODEL_LIST3<-c("linearGaussian", "gamma", "beta")
	#list of available kernels	
	KERNEL.y_LIST<-c("auto","Gaussian", "Laplace", "Cauchy")
	KERNEL.y_LIST2<-c("Gaussian", "Laplace", "Cauchy")
	KERNEL.x_LIST<-c("auto", "Gaussian", "Laplace", "Cauchy")
	KERNEL.x_LIST2<-c("Gaussian", "Laplace", "Cauchy")

	#default value for the parameters of the optimizations agorithms
	RESCALE<-TRUE
	BURNIN<-10^3
	MAXIT<-50000
	STEPSIZE<-1
	TRAJECTORY<-TRUE
	EPSILON<-10^{-4}
	ALPHA<-0.8
	C_DET<-0.2
	C_RAND<-0.1
	EPS_GD<-sqrt(.Machine$double.eps)
	EPS_SG<-10^{-5}

	res<-NULL
	
	#check that variable "model" is OK 
	if(min(model %in% MODEL_LIST)==0 || length(c(model))>1){
	  warning("Error: 'model' unknown"); return(invisible(res))
	}
	
	#check that variable "y" is OK
	if(is.numeric(y)==FALSE || (is.vector(y)==FALSE && is.matrix(y)==FALSE)){
	  warning("Error: 'y' must be a numerical vector or a numerical matrix"); return(invisible(res))
 	}
 	
 	if(model=="logistic"){
 		n<-length(y)
		I<-rep(0,n)
		I[y==0]<-1
		I[y==1]<-1
		if(sum(I)<n){
		  warning("Error: For Logsitic regression the elements of 'y' must belong to {0,1}"); return(invisible(res))
		}	
	}
	
	if(model=="gamma"){
		if(min(y)<=0){
		  warning("Error: For Gamma regression the elements of 'y' must be strictly positive"); return(invisible(res))
		}
	
	}
	
	if(model=="poisson"){
		if(sum(y%%1)>0 && min(y)<0){
		  warning("Error: For Poisson regression the elements of 'y' must be non-negative integers"); return(invisible(res))
		}
	
	}
	
	if(model=="beta"){
		if(min(y)<=0 && max(y)>=1){
		    warning("Error: For Beta regression the elements of 'y' must be in (0,1)"); return(invisible(res))
		}
	
	}
	
 	#check that variable "X" is OK
 	if( (is.numeric(X)==FALSE || (is.vector(X)==FALSE && is.matrix(X)==FALSE))){
 	  warning("Error: 'X' must  be a numerical vector or a numerical matrix"); return(invisible(res))
 	}
	
	#check that dimensions of "y" and "X" are OK
	if(is.matrix(X)){
		n <- dim(X)[1]
        	d <- dim(X)[2]
   	}else{
		n<-length(c(X))
  		d<-1
		X<-as.matrix(X)
   	}
	y<-c(y)
	if(length(y)!=n){
	  warning("Error: 'X' must be either a matrix of size n times d or a vector of length n, with n=length(c(y))"); return(invisible(res))
	}
	
	if(n<3){
	  warning("Error: the length of 'y' must be at least three"); return(invisible(res))
	}
	
	
	#check that variable "bdwth.y" is OK
	if(bdwth.y!="auto"){
	    if( (is.numeric(bdwth.y)==FALSE)){
	      warning("Error: 'bdwth.y' must  be either a strictly positive real number or equal to 'auto'"); return(invisible(res))
 	    }else if (length(c(bdwth.y))>1){
 	      warning("Error: 'bdwth.y' must  be either a strictly positive real number or equal to 'auto'"); return(invisible(res))
 	    }else if(bdwth.y<=0){
 	      warning("Error: 'bdwth.y' must  be either a strictly positive real number or equal to 'auto'"); return(invisible(res))
 	    }
 	}
 
 	#check that variable "bdwth.x" is OK
	if(bdwth.x!="auto"){
	    if( (is.numeric(bdwth.x)==FALSE)){
	      warning("Error: 'bdwth.x' must  be either non-negative real number or equal to 'auto'"); return(invisible(res))
 	    }else if(length(c(bdwth.x))>1){
 	      warning("Error: 'bdwth.x' must  be either non-negative real number or equal to 'auto'"); return(invisible(res))
 	    }else if(bdwth.x<0){
 	      warning("Error: 'bdwth.x' must  be either non-negative real number or equal to 'auto'"); return(invisible(res))
 	    }
 	}

	
	#check that variable "intercept" is OK
 	if(min(intercept %in% c("TRUE","FALSE"))==0 || length(c(intercept))>1 ){
 	  warning("Error: Possible choices for 'intercept' are  TRUE or FALSE"); return(invisible(res))
	}

	#check that an intercept can be added
	sd.x<-apply(X,2,sd)
	if(min(sd.x)==0 && intercept==TRUE){
		intercept<-FALSE
		warning("Remark: intercept not added since X contains at least one constant variable")
	}

	
	#check that variable "par1" is OK
	if( (length(par1)==1 &&  par1=="auto")==FALSE){
		if( (is.numeric(par1)==FALSE || (length(c(par1))!=d))){
		  warning("Error: 'par1' must be either a numerical vector of size p (with p the number of columns of 'X') or equal to 'auto'"); return(invisible(res))
		}
	}
	
	#check that variable "par2" is OK
	if(model %in% MODEL_LIST3){
		if(par2!="auto"){
			if( is.numeric(par2)==FALSE){
			  warning("Error: 'par2' must  be either a strictly positive real number or equal to 'auto'"); return(invisible(res))
 	    		}else if (length(c(par2))>1){
 	    		  warning("Error: 'par2' must  be either a strictly positive real number or equal to 'auto'"); return(invisible(res))
 	   	 	}else if(par2<=0){
 	   	 	  warning("Error: 'par2' must  be either a strictly positive real number or equal to 'auto'"); return(invisible(res))
 	    		}
		}
	}
	if(model %in% MODEL_LIST2){
		if(is.numeric(par2)==FALSE){
		  warning("Error: 'par2' must  be a strictly positive real number when in this model"); return(invisible(res))
 	    	}else if (length(c(par2))>1){
 	    	  warning("Error: 'par2' must  be a strictly positive real number in this model"); return(invisible(res))
 	   	 }else if(par2<=0){
 	   	   warning("Error: 'par2' must  be a strictly positive real number in this model"); return(invisible(res))
 	    	}
	}

	
	#check that variable "kernel.y" is OK
	if(min(kernel.y %in% KERNEL.y_LIST)==0 || length(c(kernel.y))>1){
	  warning("Error: 'kernel.y' unknown"); return(invisible(res))
	}
	
	#check that variable "kernel.x" is OK
	if(bdwth.x!=0){
		if(min(kernel.x %in% KERNEL.x_LIST)==0 || length(c(kernel.x))>1){
		  warning("Error: 'kernel.x' unknown"); return(invisible(res))
		}
	}
	
	##Defaut values for the control  variables
	rescale<-RESCALE
	burnin<-BURNIN
	maxit<-MAXIT
	stepsize<-STEPSIZE
	trajectory<-TRAJECTORY
	epsilon<-EPSILON
	alpha<-ALPHA
	c_det<-C_DET
	c_rand<-C_RAND
	eps_gd<-EPS_GD
	eps_sg<-EPS_SG
	
	#check that variable "eps_gd" is OK
	if(is.null(control$eps_gd)==FALSE){
		 if(is.numeric(control$eps_gd)==FALSE || length(c(control$eps_gd))!=1 || control$eps_gd<0){
		   warning("Error: 'eps_gd' must be a non-negative real number"); return(invisible(res))
 		 }else{
			eps_gd<-control$eps_gd
		 }
	}
	
	#check that variable "eps_sg" is OK
	if(is.null(control$eps_sg)==FALSE){
		 if(is.numeric(control$eps_sg)==FALSE || length(c(control$eps_sg))!=1 || control$eps_sg<0){
		   warning("Error: 'eps_sg' must be a non-negative real number"); return(invisible(res))
 		 }else{
			eps_sg<-control$eps_sg
		 }
	}
	
	
	#check that variable "rescale" is OK
	if(is.null(control$rescale)==FALSE){
 		if(min(control$rescale%in% c("TRUE","FALSE"))==0 || length(c(control$rescale))>1){
 		  warning("Error: Possible valued for 'rescale' are: TRUE, FALSE"); return(invisible(res))
		}else{
			rescale<-control$rescale
		}
	}
	#check that variable "burnin" is OK
	if(is.null(control$burnin)==FALSE){
		 if(is.numeric(control$burnin)==FALSE || length(c(control$burnin))!=1 || control$burnin<0 || control$burnin%%1!=0){
		   warning("Error: 'burnin' must be an integer greater or equal to 0"); return(invisible(res))		 
 		 }else{
			burnin<-control$burnin
		 }
	}
 	    
 	#check that variable "maxit" is OK
	if(is.null(control$maxit)==FALSE){
		 if(is.numeric(control$maxit)==FALSE || length(c(control$maxit))!=1 || control$maxit<2 || control$maxit%%1!=0){
		   warning("Error: 'maxit' must be an integer greater or equal to 2"); return(invisible(res))		 
 		 }else{
			maxit<-control$maxit
		 }
	}
	
	#check that variable "stepsize" is OK
	if(is.null(control$stepsize)==FALSE){
		 if(is.numeric(control$stepsize)==FALSE || length(c(control$stepsize))!=1 || control$stepsize<=0){
		   warning("Error: 'stepsize' must be a strictly positive real number"); return(invisible(res))		 
 		 }else{
			stepsize<-control$stepsize
		 }
	}

	#check that variable "trajectory" is OK
	if(is.null(control$trajectory)==FALSE){
 		if(min(control$trajectory%in% c("TRUE","FALSE"))==0 || length(c(control$trajectory))>1){
 		  warning("Error: Possible valued for 'trajectory' are: TRUE, FALSE"); return(invisible(res))		
		}else{
			trajectory<-control$trajectory
		}
	}
	
	#check that variable "epsilon" is OK
	if(is.null(control$epsilon)==FALSE){
		 if(is.numeric(control$epsilon)==FALSE || length(c(control$epsilon))!=1 || control$epsilon<=0){
		   	
 		 }else{
			epsilon<-control$epsilon
		 }
	}
	
	#check that variable "alpha" is OK
	if(is.null(control$alpha)==FALSE){
		 if(is.numeric(control$alpha)==FALSE || length(c(control$alpha))!=1 || control$alpha<=0 || control$alpha>=1){
		   warning("Error: 'alpha' must be an element of (0,1)"); return(invisible(res))
 		 }else{
			alpha<-control$alpha
		 }
	}
	
	#check that variable "c_det" is OK
	if(is.null(control$c_det)==FALSE && bdwth.x!=0){
		 if(is.numeric(control$c_det)==FALSE || length(c(control$c_det))!=1 || control$c_det<=0 || control$c_det>=1){
		   warning("Error: 'c_det' must be an element of (0,1)"); return(invisible(res))
 		 }else{
			c_det<-control$c_det
		 }
	}
	
	#check that variable "c_rand" is OK
	if(is.null(control$c_rand)==FALSE && bdwth.x!=0){
		 if(is.numeric(control$c_rand)==FALSE || length(c(control$c_rand))!=1 || control$c_rand<=0 || control$c_rand>=1){
		   warning("Error: 'c_rand' must be an element of (0,1)"); return(invisible(res))
 		 }else{
			c_rand<-control$c_rand
		 }
	}
	
	
	##Preliminary computations
   	
	#rescale variables if needed
	if(rescale){
		if(min(sd.x)==0){
			i.x<-which(sd.x==0)
			X[,-i.x]<-t( t(X[,-i.x])/sd.x[-i.x])
			sd.x[i.x]<-1
		}else{
			X<-t( t(X)/sd.x)
      		}
   	}
   	
   	#add an intercept if needed
   	if(intercept){
		X<-cbind(rep(1,n),X)
		sd.x<-c(1,sd.x)
       	d<-d+1
   	}
   
	#Compute bdwth.y if bdwth.y=auto (median heuristic)
	if(bdwth.y=="auto"){
		bdwth.y<-median(c(dist(y, method="euclidean")))/sqrt(2)
		if(bdwth.y==0) bdwth.y<-1
	}
	
	#preliminary computations for computing theta_hat
	if(bdwth.x!=0){
		#sort observations according to their pairwise distances
		sorted_obs<-sort_obs(X) 
		if(bdwth.x=="auto"){
			bdwth.x<-0.01*median(sorted_obs$DIST)/sqrt(2)
			if(bdwth.x==0) bdwth.x<-1
		}
		if(kernel.x=="auto"){
			kernel.x<-"Laplace"
		}
		KX<-K1d_dist(sorted_obs$DIST, kernel.x, bdwth.x)   
		M.det<-floor(n*c_det)
		M.rand<-floor(n*c_rand)
		l_KX<-length(KX)
		if( (n+M.det+M.rand)>l_KX){
			M.det<-l_KX-n-2
			M.rand<-2
		}
	}
	
	##Linear Gaussian Model
	if(model %in% c("linearGaussian", "linearGaussian.loc")){
		##Estimator theta_tilde
		if(bdwth.x==0){
			##Fixed scale parameter
			if(model=="linearGaussian.loc"){
				if(kernel.y=="Gaussian" || kernel.y=="auto"){
					kernel.y<-"Gaussian"
					res<-LG_GD_loc_tilde(y, X, intercept, sd.x, par1, par2, bdwth.y, maxit, alpha, eps_gd)
				}else{
					res<-LG_SGD_loc_tilde(y, X, intercept, sd.x, par1, par2, kernel.y, bdwth.y, burnin, maxit, stepsize, epsilon, eps_sg) 
				}
			##Also estimate the scale parameter
			}else{
				if(kernel.y=="Gaussian" || kernel.y=="auto"){
					kernel.y<-"Gaussian"
					res<-LG_GD_tilde(y, X, intercept, sd.x, par1, par2, bdwth.y, maxit,  alpha, eps_gd)
				}else{
					res<-LG_SGD_tilde(y, X, intercept, sd.x, par1, par2, kernel.y, bdwth.y, burnin, maxit, stepsize, epsilon, eps_sg) 
				}
			}
		##Estimator theta_hat
		}else{
			##Fixed scale parameter
			if(model=="linearGaussian.loc"){
				if(kernel.y=="Gaussian" || kernel.y=="auto"){
					kernel.y<-"Gaussian"
					res<-LG_PartialSGD_loc_hat(y, X, intercept, sd.x, par1, par2, M.det, M.rand, bdwth.y,
								   		 burnin, maxit, stepsize, epsilon, sorted_obs, KX, eps_sg)
				}else{
					res<-LG_SGD_loc_hat(y, X, intercept, sd.x, par1, par2, kernel.y, M.det, M.rand, bdwth.y,
								 burnin, maxit, stepsize,  epsilon, sorted_obs, KX, eps_sg) 
				}
			##Also estimate the scale parameter
			}else{
				if(kernel.y=="Gaussian" || kernel.y=="auto"){
					kernel.y<-"Gaussian"
					res<-LG_PartialSGD_hat(y, X, intercept, sd.x, par1, par2, M.det, M.rand, bdwth.y,
								   		 burnin, maxit, stepsize, epsilon, sorted_obs, KX, eps_sg)
				}else{
					res<-LG_SGD_hat(y, X, intercept, sd.x, par1, par2, kernel.y, M.det, M.rand, bdwth.y,
								 burnin, maxit, stepsize, epsilon, sorted_obs, KX, eps_sg) 
				}
			
			}
		
		}
	}else if(model=="logistic"){
		if(kernel.y=="auto"){
			kernel.y<-"Laplace"
		}
		##Estimator theta_tilde
		if(bdwth.x==0){
			res<-Logistic_tilde(y, X, intercept, sd.x, par1, kernel.y, bdwth.y, maxit, alpha, eps_gd)
		##Estimator theta_hat
		}else{	
			res<-Logistic_hat(y, X, intercept, sd.x, par1, kernel.y, M.det, M.rand, bdwth.y,
						burnin, maxit, stepsize, epsilon, sorted_obs, KX, eps_sg)
		}
	
	}else if(model %in% c("gamma", "gamma.loc")){
		if(kernel.y=="auto"){
			kernel.y<-"Laplace"
		}
		##Estimator theta_tilde
		if(bdwth.x==0){
			##Fixed shape parameter
			if(model=="gamma.loc"){
				res<-Gamma_loc_tilde(y, X, intercept, sd.x, par1, par2, kernel.y, bdwth.y, burnin, maxit, stepsize, epsilon, eps_sg) 
				
			##Also estimate the shape parameter
			}else{
				res<-Gamma_tilde(y, X, intercept, sd.x, par1, par2, kernel.y, bdwth.y, burnin, maxit, stepsize, epsilon, eps_sg) 
			}
		##Estimator theta_hat
		}else{
			##Fixed shape parameter
			if(model=="gamma.loc"){
				res<-Gamma_loc_hat(y, X, intercept, sd.x, par1, par2, kernel.y, M.det, M.rand, bdwth.y,
								 burnin, maxit, stepsize, epsilon, sorted_obs, KX, eps_sg) 
			##Also estimate the shape parameter
			}else{
				res<-Gamma_hat(y, X, intercept, sd.x, par1, par2, kernel.y, M.det, M.rand, bdwth.y,
								 burnin, maxit, stepsize, epsilon, sorted_obs, KX, eps_sg) 
			}
		}
	}else if(model=="exponential"){
		if(kernel.y=="auto"){
			kernel.y<-"Laplace"
		}
		##Estimator theta_tilde
		if(bdwth.x==0){
			res<-Gamma_loc_tilde(y, X, intercept, sd.x, par1, par2=1, kernel.y, bdwth.y, burnin, maxit, stepsize, epsilon, eps_sg) 
		##Estimator theta_hat
		}else{
			res<-Gamma_loc_hat(y, X, intercept, sd.x, par1, par2=1, kernel.y, M.det, M.rand, bdwth.y,
								 burnin, maxit, stepsize, epsilon, sorted_obs, KX, eps_sg) 
		}
	}else if(model=="poisson"){
		if(kernel.y=="auto"){
			kernel.y<-"Laplace"
		}
		##Estimator theta_tilde
		if(bdwth.x==0){
			res<-Poisson_tilde(y, X, intercept, sd.x, par1, kernel.y, bdwth.y, burnin, maxit, stepsize, epsilon, eps_sg) 
		##Estimator theta_hat
		}else{
			res<-Poisson_hat(y, X, intercept, sd.x, par1, kernel.y, M.det, M.rand, bdwth.y,
								 burnin, maxit, stepsize, epsilon, sorted_obs, KX, eps_sg)  
		}
	
	}else  if(model %in% c("beta", "beta.loc")){
		if(kernel.y=="auto"){
			kernel.y<-"Laplace"
		}
		##Estimator theta_tilde
		if(bdwth.x==0){
			if(model=="beta.loc"){
				res<-Beta_loc_tilde(y, X, intercept, sd.x, par1, par2, kernel.y, bdwth.y, burnin, maxit, stepsize, epsilon, eps_sg) 
			##Also estimate the precision parameter
			}else{
				res<-Beta_tilde(y, X, intercept, sd.x, par1, par2, kernel.y, bdwth.y, burnin, maxit, stepsize, epsilon, eps_sg) 
			}
		##Estimator theta_hat
		}else{
			if(model=="beta.loc"){
				res<-Beta_loc_hat(y, X, intercept, sd.x, par1, par2, kernel.y, M.det, M.rand, bdwth.y,
						   burnin, maxit, stepsize, epsilon, sorted_obs, KX, eps_sg) 
			##Also estimate the precision parameter
			}else{
				res<-Beta_hat(y, X, intercept, sd.x, par1, par2, kernel.y, M.det, M.rand, bdwth.y,
					      burnin, maxit, stepsize, epsilon, sorted_obs, KX, eps_sg) 
			}						 
		}
	
	}
	
	
	#check convergence 
	if(res$convergence==1){
	  warning("Warning: The maximum number of iterations  has been reached.")
		trajectory<-TRUE
	}
	if(res$convergence==-1){
		res<-NULL
		warning("Error: Optimization of the objective function has failed"); return(invisible(res))
	}
	
	if(trajectory==FALSE) res$trajectory<-NULL
	res$convergence<-NULL
	res$model<-model
	res$intercept<-intercept
	
	res$kernel.y<-kernel.y
	res$bdwth.y<-bdwth.y
	res$kernel.x<-kernel.x
	res$bdwth.x<-bdwth.x
	class(res) <- "regMMD"
	return(res)
	
}
