#' Summary method for the \code{class} \code{"estMMD"}
#' 
#' @param object An object of \code{class} \code{"estMMD"}.
#' @param ... Additional arguments (not used).
#' @return No return value, called only to print information on the output of \code{"estMMD"}.
#' @export
summary.estMMD <- function(object, ...) {
  summary_estMMD(object)
}
