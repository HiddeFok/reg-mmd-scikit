#' MMD estimation
#'
#' @importFrom stats dist median sd
#' @description Fits a statistical models to the data, using the robust procedure based on maximum mean discrepancy (MMD) minimization introduced and studied in \insertCite{briol2019statistical,MMD2;textual}{regMMD}.
#' 
#' @usage mmd_est(x, model, par1, par2, kernel, bdwth, control= list())
#'
#' @param x Data. Must be a vector for univariate models, a matrix of dimension n by d, where n is the sample size and d the dimension of the model.
#' @param model Parametric model to be fitted to the data. No default. See details for the list of available models.
#' @param par1 First parameter of the model. In models where the first parameter is fixed, it is necessary to provide a value for \code{par1}. In models where the first parameter is estimated, \code{par1} can be used to provide an alternative to the default initialization of the optimization algorithms.
#' @param par2 Second parameter of the model (if any). In models where the second parameter is fixed, it is necessary to provide a value for \code{par2}. In models where the first parameter is estimated,\code{par2} can be used to provide an alternative to the default initialization of the optimization algorithms.
#' @param kernel Kernel to be used in the MMD. Available options for \code{kernel} are "Gaussian" (Gaussian kernel), "Laplace" (Laplace, or exponential, kernel) and "Cauchy" (Cauchy kernel). By default, \code{kernel}="Gaussian"
#' @param bdwth Bandwidth parameter for the kernel. \code{bdwth} must be a strictly positive real number. By default, the value of \code{bdwth} is chosen using the  median heuristic \insertCite{garreau2017large}{regMMD}.
#' @param control A \code{list} of control parameters for the numerical optimization of the objective function. See details.
#'
#' @details
#' Available options for \code{model} are:
#' \describe{
#' \item{"\code{beta}"}{Beta distribution with pdf \eqn{~x^{a-1}(1-x)^(b-1)} on \eqn{[0,1]}, \code{par1}\eqn{=a} and \code{par2}\eqn{=b} are both estimated.}
#' \item{"\code{binomial}"}{Binomial distribution with pmf \eqn{~p^{x}(1-p)^{N-x}} on \eqn{\{0,1,...,N\}}, \code{par1}\eqn{=N} and \code{par2}\eqn{=p} are both estimated. Note that in this case, if the user specifies a value for \eqn{N}, it is used as an upper bound rather than an initialization.}
#' \item{"\code{binomial.prob}"}{Binomial distribution with pmf \eqn{~p^{x}(1-p)^{N-x}} on \eqn{\{0,1,...,N\}}, \code{par1}\eqn{=N} is fixed and must be specified by the user while \code{par2}\eqn{=p} is estimated.}
#' \item{"\code{binomial.size}"}{Binomial distribution with pmf \eqn{~p^{x}(1-p)^{N-x}} on \eqn{\{0,1,...,N\}}, \code{par1}\eqn{=N} is estimated while \code{par2}\eqn{=p} fixed and must be specified by the user. Note that in this case, if the user specifies a value for \eqn{N}, it is used as an upper bound rather than an initialization.}
#' \item{"\code{Cauchy}"}{Cauchy distribution with pdf \eqn{~1/(1+(x-m)^2)}, \code{par1}\eqn{=m} is estimated.}
#' \item{"\code{continuous.uniform.loc}"}{Uniform distribution with pdf \eqn{~1} on \eqn{[m-L/2,m+L/2]}, \code{par1}\eqn{=m} is estimated while \code{par2}\eqn{=L}  is fixed and must be specified by the user.}
#' \item{"\code{continuous.uniform.upper}"}{Uniform distribution with pdf \eqn{~1} on \eqn{[a,b]}, \code{par1}\eqn{=a} is fixed and must be specified by the user while \code{par2}\eqn{=b} is estimated.}
#' \item{"\code{continuous.uniform.lower.upper}"}{Uniform distribution with pdf \eqn{~1} on \eqn{[a,b]}, \code{par1}\eqn{=a} and \code{par2}\eqn{=b} are estimated.}
#' \item{"\code{Dirac}"}{Dirac mass at point \eqn{a} on the reals, \code{par1}\eqn{=a} is estimated.}
#' \item{"\code{discrete.uniform}"}{Uniform distribution with pmf \eqn{~1} on \eqn{\{1,2,..,M\}}, \code{par1}\eqn{=M} is estimated. Note that in this case, if the user specifies a value for \eqn{M}, it is used as an upper bound rather than an initialization.}
#' \item{"\code{exponential}"}{Exponential distribution with pdf \eqn{~\exp(-b x)} on positive reals \eqn{R_+}, \code{par1}\eqn{=b} is estimated.}
#' \item{"\code{gamma}"}{Gamma distribution with pdf \eqn{~x^{a-1}\exp(-b x)} on positive reals \eqn{R_+}, \code{par1}\eqn{=a>=0.5} and \code{par2}\eqn{=b} are estimated.}
#' \item{"\code{gamma.shape}"}{Gamma distribution with pdf \eqn{~x^{a-1}\exp(-b x)} on positive reals \eqn{R_+}, \code{par1}\eqn{=a>=0.5} is estimated while \code{par2}\eqn{=b} is fixed and must be specified by the user.}
#' \item{"\code{gamma.rate}"}{Gamma distribution with pdf \eqn{~x^{a-1}\exp(-b x)} on positive reals \eqn{R_+}, \code{par1}\eqn{=a>=0.5} is fixed and must be specified by the user while \code{par2}\eqn{=b} is estimated.}
#' \item{"\code{Gaussian}"}{Gaussian distribution with pdf\eqn{~\exp(-(x-m)^2/2s^2)} on reals \eqn{R}, \code{par1}\eqn{=m} and \code{par2}\eqn{=s} are estimated.}
#' \item{"\code{Gaussian.loc}"}{Gaussian distribution with pdf \eqn{~\exp(-(x-m)^2/2s^2)} on reals \eqn{R}, \code{par1}\eqn{=m} is estimated while \code{par2}\eqn{=s} is fixed and must be specified by the user.}
#' \item{"\code{Gaussian.scale}"}{Gaussian distribution with pdf \eqn{~\exp(-(x-m)^2/2s^2)} on reals \eqn{R}, \code{par1}\eqn{=m} is fixed and must be specified by the user while \code{par2}\eqn{=s} is estimated.}
#' \item{"\code{geometric}"}{Geometric distribution with pmf \eqn{~p(1-p)^x} on \eqn{\{0,1,2,...\}}, \code{par1}\eqn{=p} is estimated.}
#' \item{"\code{multidim.Dirac}"}{Dirac mass at point \eqn{a} on \eqn{R^d}, \code{par1}\eqn{=a} (\eqn{d}-dimensional vector) is estimated.}
#' \item{"\code{multidim.Gaussian}"}{Gaussian distribution with pdf \eqn{~\exp(-(x-m)'U'U(x-m)} on \eqn{R^d}, \code{par1}\eqn{=m} (\eqn{d}-dimensional vector) and \code{par2}\eqn{=U} (\eqn{d}-\eqn{d} matrix) are estimated.}
#' \item{"\code{multidim.Gaussian.loc}"}{Gaussian distribution with pdf \eqn{~\exp(-\|x-m\|^2/2s^2)} on \eqn{R^d}, \code{par1}\eqn{=m} (\eqn{d}-dimensional vector) is estimated while \code{par2}\eqn{=s} is fixed.}
#' \item{"\code{multidim.Gaussian.scale}"}{Gaussian distribution with pdf \eqn{~\exp(-(x-m)'U'U(x-m)} on \eqn{R^d}, \code{par1}\eqn{=m} (\eqn{d}-dimensional vector) is fixed and must be specified by the user while and \code{par2}\eqn{=U} (\eqn{d}-\eqn{d} matrix) is estimated.}
#' \item{"\code{Pareto}"}{Pareto distribution with pmf \eqn{~1/x^{a+1}} on the reals \eqn{>1}, \code{par1}\eqn{=a} is estimated.}
#' \item{"\code{Poisson}"}{Poisson distribution with pmf \eqn{~b^x/x!} on \eqn{\{0,1,2,...\}}, \code{par1}\eqn{=b} is estimated.}
#' }
#' 
#' The \code{control} argument is a list  that can supply any of the following components:
#' \describe{
#' \item{burnin}{Length of the burn-in period in GD or SGD. \code{burnin} must be a non-negative integer and defaut \code{burnin}==\eqn{500}.}
#' \item{nsteps}{Number of iterations performed after the burn-in period in GD or SGD. \code{nsetps} must be an integer strictly larger than 2 and by default \code{nsteps}=\eqn{1000}}
#' \item{stepsize}{Stepsize parameter. An adaptive gradient step is used (adagrad), but it is possible to pre-multiply it by \code{stepsize}. It must be strictly positive number and by default \code{stepsize}=\eqn{1}}
#' \item{epsilon}{Parameter used in adagrad to avoid numerical errors in the computation of the step-size. \code{epsilon} must be a strictly positive real number and by default \code{epsilon}=\eqn{10^{-4}}.}
#' \item{method}{Optimization method to be used: \code{"EXACT"} for exact, \code{"GD"} for gradient descent and \code{"SGD"} for stochastic gradient descent. Not all methods are available for all models. By default, exact is preferred to GD which is prefered to SGD.}
#' }
#'
#' @return  \code{MMD_est} returns an object of \code{class} \code{"estMMD"}.
#'
#' The functions \code{summary}  can be used to print the results.
#'
#' An object of class \code{estMMD} is a list containing the following components:
#'   \item{model}{Model estimated}
#'   \item{par1}{In models where the first parameter is fixed, this is the value \code{par1} fixed by the user. In models where the first parameter is estimated, this is the initialization of the optimization procedure}
#'   \item{par2}{In models where the second parameter is fixed, this is the value \code{par2} fixed by the user. In models where the second parameter is estimated, this is the initialization of the optimization procedure}
#'   \item{kernel}{Kernel used in the MMD}
#'   \item{bdwth}{Bandwidth used. That is, either the value specified by the user, either the bandwidth computedby the median heuristic}
#'   \item{burnin}{Number of steps in the burnin of the GD or SGD algorithm}
#'   \item{nstep}{Number of steps in the GD or SGD algorithm}
#'   \item{stepsize}{Stepize parameter in GD or SGD}
#'   \item{epsilon}{Parameter used in adagrad to avoid numerical errors in the computation of the step-size}
#'   \item{method}{Optimization method used}
#'   \item{error}{Error message (if any)}
#'   \item{estimator}{Estimated parameter(s)}
#'   \item{type}{Takes the value "\code{est}"}
#' @export
#'
#' @examples
#' #simulate data
#' x = rnorm(50,0,1.5)
#' 
#' # estimate the mean and variance (assuming the data is Gaussian)
#' Est = mmd_est(x, model="Gaussian")
#' 
#' # print a summary
#' summary(Est)
#' 
#' # estimate the mean (assuming the data is Gaussian with known standard deviation =1.5)
#' Est2 = mmd_est(x, model="Gaussian.loc", par2=1.5)
#' 
#' # print a summary
#' summary(Est2)
#' 
#' # estimate the standard deviation (assuming the data is Gaussian with known mean = 0)
#' Est3 = mmd_est(x, model="Gaussian.scale", par1=0)
#' 
#' # print a summary
#' summary(Est3)
#' 
#' # test of the robustness
#' x[42] = 100
#' 
#' mean(x)
#' 
#' # estimate the mean and variance (assuming the data is Gaussian)
#' Est4 = mmd_est(x, model="Gaussian")
#' summary(Est4)
#' 
#' @references
#'   \insertAllCited{}
#' @export

mmd_est = function(x, model, par1=NULL, par2=NULL, kernel="Gaussian", bdwth="median", control= list()) {
  
  # useful lists
  
  list.models.multidim = c("multidim.Gaussian.loc",
                           "multidim.Gaussian.scale",
                           "multidim.Gaussian",
                           "multidim.Dirac")
  
  list.models.uni = c("Gaussian.loc",
                      "Gaussian.scale",
                      "Gaussian",
                      "continuous.uniform.loc",
                      "continuous.uniform.upper",
                      "continuous.uniform.lower.upper",
                      "exponential",
                      "gamma.shape",
                      "gamma.rate",
                      "gamma",
                      "Cauchy",
                      "Pareto",
                      "beta",
                      "Poisson",
                      "geometric",
                      "Dirac",
                      "binomial.prob",
                      "binomial.size",
                      "binomial",
                      "discrete.uniform")
  
  list.kernels = c("Gaussian",
                   "Laplace",
                   "Cauchy")
  
  list.methods = c("GD",
                   "SGD",
                   "EXACT")
  
  # check the control options and fill with default
  
  if (is.null(control$burnin)==FALSE) {
    if ((is.double(control$burnin)==FALSE)||(length(control$burnin)!=1)) {
      res$error = c(res$error,"control$burnin must be numerical")
    } else if (floor(control$burnin)!=control$burnin) {
      res$error = c(res$error,"control$burnin must be an integer")
    } else burnin = control$burnin
  } else burnin = 500
  
  if (is.null(control$nstep)==FALSE) {
    if ((is.double(control$nstep)==FALSE)||(length(control$nstep)!=1)) {
      res$error = c(res$error,"control$nstep must be numerical")
    } else if (floor(control$nstep)!=control$nstep) {
      res$error = c(res$error,"control$nstep must be an integer")
    } else nstep = control$nstep
  } else nstep = 1000
  
  if (is.null(control$epsilon)==FALSE) {
    if ((is.double(control$epsilon)==FALSE)||(length(control$epsilon)!=1)) {
      res$error = c(res$error,"control$epsilon must be numerical")
    } else epsilon = control$epsilon
  } else epsilon = 10^(-5)
  
  if (is.null(control$stepsize)==TRUE) stepsize = "auto" else stepsize = control$stepsize
  
  if (is.null(control$method)==TRUE) method = "default" else method = control$method
  
  # prepare the output
  
  out = list(model=model, par1init=par1, par2init=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon, method=method, trajectory=NULL, error=NULL, estimator=NULL, type="est")
  
  # sanity test for the inputs, and set some of the default values
  
  if (model%in%list.models.uni) {
    if ((is.vector(x)==FALSE)||(is.numeric(x)==FALSE)) out$error = c(out$error,"Data x should be a numeric vector for this model")
  } else if (model%in%list.models.multidim) {
    if (is.matrix(x)==FALSE) out$error = c(out$error,"Data x should be an (n,p) matrix for this model (n: sample size)")
  } else out$error = c(out$error,"Model unknown")
  
 # if there was an error so far, don't go further
  
  if (is.null(out$error)==FALSE) {for (i in 1:length(out$error)) warning(out$error[i]); return(invisible(out))}
  
  if (bdwth=="median") {
    bdwth = median(c(dist(x, method="euclidean")))
    if (bdwth==0) bdwth=1
  } else if (is.double(bdwth)==FALSE) out$error = c(out$error,"bdwth must be numeric or 'median'") else if (bdwth<=0) out$error = c(out$error,"bdwth must be positive")
  
  if (kernel%in%list.kernels==FALSE) out$error = c(out$error,"Kernel unknown")
  
  if (is.double(burnin)==FALSE) out$error = c(out$error,"burnin must be numeric")
  if (is.double(nstep)==FALSE) out$error = c(out$error,"nstep must be numeric")
  if (is.double(epsilon)==FALSE) out$error = c(out$error,"epsilon must be numeric")
  if (is.double(stepsize)==FALSE) if (stepsize!="auto") out$error = c(out$error,"stepsize must be numeric or 'auto'")
  
  # if there was an error so far, don't go further
  
  if (is.null(out$error)==FALSE) {for (i in 1:length(out$error)) warning(out$error[i]); return(invisible(out))}
  
  # CALL FOR MODELS WITH MANY OPTIMIZATION METHODS AVAILABLE
  
  if (model=="Gaussian.loc") {
    if (((kernel!="Gaussian")&&(method=="GD"))||(method== "EXACT")) out$error = c(out$error,"This method is not possible with this kernel and model")
    if (((kernel=="Gaussian")&&(method=="GD"))||((kernel=="Gaussian")&&(method=="default"))) { method="GD"; resultat = GD.MMD.Gaussian.loc(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)}
    if ((method=="SGD")||((kernel!="Gaussian")&&(method=="default"))) { method="GD"; resultat = SGD.MMD.Gaussian.loc(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)}
  }
  if (model=="continuous.uniform.loc") {
    if (method=="EXACT") out$error = c(out$error,"This method is not possible with this kernel and model")
    if ((method=="GD")||(method=="default")) {method="GD";  resultat = GD.MMD.continuous.uniform.loc(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)}
    if (method=="SGD") resultat = SGD.MMD.continuous.uniform.loc(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
  }
  if (model=="binomial.prob") {
    if (method=="EXACT") out$error = c(out$error,"This method is not possible with this kernel and model")
    if ((method=="GD")||(method=="default")) {method="GD"; resultat = GD.MMD.binomial.prob(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)}
    if (method=="SGD") resultat = SGD.MMD.binomial.prob(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
  }
  if (model=="multidim.Gaussian.loc") {
    if (((kernel!="Gaussian")&&(method=="GD"))||(method== "EXACT")) out$error = c(out$error,"This method is not possible with this kernel and model")
    if (((kernel=="Gaussian")&&(method=="GD"))||((kernel=="Gaussian")&(method=="default"))) { method="GD"; resultat = GD.MMD.multidim.Gaussian.loc(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)}
    if ((method=="SGD")||((kernel!="Gaussian")&&(method=="default"))) { method="SGD"; resultat = SGD.MMD.multidim.Gaussian.loc(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)}
  }
  
  # CALL FOR MODELS WITH ONLY SGD AVAILABLE
  
  if (model=="Gaussian.scale") {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.Gaussian.scale(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="Gaussian") {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.Gaussian(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="multidim.Gaussian") {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.multidim.Gaussian(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="multidim.Gaussian.scale")  {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.multidim.Gaussian.scale(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="continuous.uniform.upper") {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.continuous.uniform.upper(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="continuous.uniform.lower.upper") {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.continuous.uniform.lower.upper(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="exponential") {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.gamma.rate(x=x, par1=1, par2=par1, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
      resultat$par1 = resultat$par2
      resultat$par2 = NULL
    }
  }
  if (model=="gamma.shape") {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.gamma.shape(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="gamma.rate") {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.gamma.rate(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="gamma") {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.gamma(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="Cauchy") {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.Cauchy(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="Pareto") {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.Pareto(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  } 
  if (model=="beta") {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.beta(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="Poisson") {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.Poisson(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="geometric") {
    if (method=="default") method="SGD"
    if (method!="SGD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = SGD.MMD.geometric(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="Dirac") {
    if (method=="default") method="GD"
    if (method!="GD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = GD.MMD.Dirac(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  
  # CALL FOR MODELS WITH ONLY GD AVAILABLE
  
  if (model=="multidim.Dirac") {
    if (method=="default") method="GD"
    if (method!="GD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = GD.MMD.multidim.Dirac(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="binomial") {
    if (method=="default") method="GD"
    if (method!="GD") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = GD.MMD.binomial(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  
  # CALL FOR MODELS WITH ONLY EXACT AVAILABLE
  
  if (model=="binomial.size") {
    if (method=="default") method="EXACT"
    if (method!="EXACT") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = EXACT.MMD.binomial.size(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  if (model=="discrete.uniform") {
    if (method=="default") method="EXACT"
    if (method!="EXACT") out$error = c(out$error,"This method is not possible with this kernel and model") else {
      resultat = EXACT.MMD.discrete.uniform(x=x, par1=par1, par2=par2, kernel=kernel, bdwth=bdwth, burnin=burnin, nstep=nstep, stepsize=stepsize, epsilon=epsilon)
    }
  }
  
  # check for ultimate errors
  
  if (is.null(out$error)==FALSE) {for (i in 1:length(out$error)) warning(out$error[i]); return(invisible(out))}
  
  # return the final result 
  
  if ((is.null(resultat$error)==TRUE)||((length(resultat$error)==1)&&(startsWith(resultat$error[1],"Attention"))))
  {
    out$par1init = resultat$par1
    out$par2init = resultat$par2
    out$stepsize = resultat$stepsize
    out$bdwth = resultat$bdwth
    out$error = resultat$error
    out$method = method
    out$estimator = resultat$estimator
    out$trajectory = resultat$trajectory
    if (is.null(out$error)==FALSE)  warning(out$error)
    class(out) <- "estMMD"
    return(invisible(out))
  } else {out$error= resultat$error; for (i in 1:length(out$error)) warning(out$error[i]); return(invisible(out))}
  
}
