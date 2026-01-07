#' Summary method for the \code{class} \code{"regMMD"}
#' 
#' @param object An object of \code{class} \code{"regMMD"}.
#' @param ... Additional arguments (not used).
#' @return No return value, called only to print information on the output of \code{"regMMD"}.
#' @export
summary.regMMD <- function(object, ...) {
  summary_regMMD(object)
}
