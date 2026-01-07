

summary_regMMD <- function(object) {
  width <- 40
  estimator <- ifelse(object$bdwth.x == 0, "theta tilde (bdwth.x=0)", "theta hat  (bdwth.x>0)")
  
  # Create a header with separators
  lines <- c(
    "======================== Summary ========================",
    sprintf("%-20s %s", "Model:", object$model),
    sprintf("%-20s %s", "Estimator:", estimator),
    "---------------------------------------------------------",
    sprintf("  %-20s %10s", "Coefficients", "Estimate"), # Adding "Estimate" header
    "---------------------------------------------------------"
  )
  
  d <- length(object$coefficients)
  betas <- paste0("X", 1:d)
  
  # Add Intercept and coefficients
  if (object$intercept) {
    lines <- c(lines, sprintf("  %-20s %10s", "(Intercept)", round(object$coefficients[1], 4)))
    for (i in 1:(d - 1)) {
      lines <- c(lines, sprintf("  %-20s %10s", betas[i], round(object$coefficients[i + 1], 4)))
    }
  } else {
    lines <- c(lines, sprintf("  %-20s %10s", "(Intercept)", "NA"))
    for (i in 1:d) {
      lines <- c(lines, sprintf("  %-20s %10s", betas[i], round(object$coefficients[i], 4)))
    }
  }
  
  lines <- c(lines, "---------------------------------------------------------")
  
  # Function to add parameter info
  add_parameter_info <- function(param_name, phi_value, is_fixed) {
    line <- paste(param_name, ":", round(phi_value, 4), ifelse(is_fixed, "(fixed)", "(estimated)"))
    c(sprintf("  %-20s", line))
  }
  
  # Add model-specific information
  if (object$model %in% c("linearGaussian", "linearGaussian.loc")) {
    lines <- c(lines, add_parameter_info("Std. dev. of Gaussian noise", object$phi, object$model == "linearGaussian.loc"))
  } else if (object$model %in% c("gamma", "gamma.loc")) {
    lines <- c(lines, add_parameter_info("Shape parameter", object$phi, object$model == "gamma.loc"))
  } else if (object$model %in% c("beta", "beta.loc")) {
    lines <- c(lines, add_parameter_info("Precision parameter", object$phi, object$model == "beta.loc"))
  }
  
  # Add kernel information
  lines <- c(
    lines,
    "---------------------------------------------------------",
    sprintf("  %-20s", paste("Kernel for y:", object$kernel.y, "with bandwidth", round(object$bdwth.y, 4)))
  )
  
  if (object$bdwth.x != 0) {
    lines <- c(lines, sprintf("  %-20s", paste("Kernel for x:", object$kernel.x, "with bandwidth", round(object$bdwth.x, 4))))
  }
  
  lines <- c(lines, "=========================================================")
  
  # Print all lines at once
  cat(paste(lines, collapse = "\n"), "\n")
}

