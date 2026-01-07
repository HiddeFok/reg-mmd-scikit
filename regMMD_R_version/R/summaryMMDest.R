

summary_estMMD<- function(Est) {
  width <- 30
  
  # useful lists
  
  list.par1.uni = c("mean",
                    "mean",
                    "mean",
                    "center of interval",
                    "lower bound",
                    "lower bound",
                    "rate",
                    "shape",
                    "shape",
                    "shape",
                    "location parameter",
                    "scale",
                    "shape 1",
                    "rate",
                    "probability of success",
                    "location",
                    "number of trials (size)",
                    "number of trials (size)",
                    "number of trials (size)",
                    "size"
  )
  
  list.par2.uni = c("standard deviation",
                    "standard deviation",
                    "standard deviation",
                    "lenght of interval",
                    "upper bound",
                    "upper bound",
                    "",
                    "rate",
                    "rate",
                    "rate",
                    "",
                    "",
                    "shape 2",
                    "",
                    "",
                    "",
                    "probability of success",
                    "probability of success",
                    "probability of success",
                    ""
  )
  
  list.action1.uni = c("initialized at",
                       "fixed by user:",
                       "initialized at",
                       "initialized at",
                       "fixed by user:",
                       "initialized at",
                       "initialized at",
                       "initialized at",
                       "fixed by user:",
                       "initialized at",
                       "initialized at",
                       "initialized at",
                       "initialized at",
                       "initialized at",
                       "initialized at",
                       "initialized at",
                       "fixed by user:",
                       "initialized at",
                       "initialized at",
                       "initialized at"
  )
  
  list.action2.uni = c("fixed by user:",
                       "initialized at",
                       "initialized at",
                       "fixed by user:",
                       "initialized at",
                       "initialized at",
                       "",
                       "fixed by user:",
                       "initialized at",
                       "initialized at",
                       "",
                       "",
                       "initialized at",
                       "",
                       "",
                       "",
                       "initialized at",
                       "fixed by user:",
                       "initialized at",
                       ""
  )
  
  
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
  
  # summary
  
  lines <- c(
    "======================== Summary ========================",
    sprintf("%-20s %s", "Model:", Est$model),
    "---------------------------------------------------------"
  )
  
  if (is.null(Est$error)==FALSE) {
    lines = c(lines,
      "Error(s):"
    )
    for (i in length(Est$error)) {
      lines = c(lines,
        Est$error[i]
      )
    }
    cat(paste(lines, collapse = "\n"), "\n")
    if ((length(Est$error)!=1)||(startsWith(Est$error[1],"Attention"))) return(invisible(NULL))
  }

  lines <- c(lines,
    sprintf("%-20s %s", "Algorithm:", Est$method),
    sprintf("%-20s %s", "Kernel:", Est$kernel),
    sprintf("%-20s %s", "Bandwidth:", Est$bdwth),
    "---------------------------------------------------------"
  )

  lines <-c(lines,
    sprintf("%-*s", width, "Parameters:"),
    " "
  )
  
  for (i in list.models.uni) if (Est$model==i) {
    lines <-c(lines,
      sprintf("%-*s", width, paste("par1:", list.par1.uni[list.models.uni==i], "--", list.action1.uni[list.models.uni==i], round(Est$par1init, digits=4)))
              )
    if (list.action1.uni[list.models.uni==i]!="fixed by user:") {
      lines <-c(lines,
        sprintf("%-*s", width, paste("      estimated value:",round(Est$estimator[1], digits=4)))
      )
    }
    
    if (list.par2.uni[list.models.uni==i]!="") {
      lines <-c(lines,
        " ",
        sprintf("%-*s", width, paste("par2:", list.par2.uni[list.models.uni==i], "--", list.action2.uni[list.models.uni==i], round(Est$par2init, digits=4)))
      )
      if (list.action1.uni[list.models.uni==i]=="fixed by user:") {
        lines <-c(lines,
          sprintf("%-*s", width, paste("      estimated value:",round(Est$estimator[1], digits=4)))
        )
      } else if (list.action2.uni[list.models.uni==i]!="fixed by user:") {
        lines <-c(lines,
          sprintf("%-*s", width, paste("      estimated value:",round(Est$estimator[2], digits=4)))
        )
      }
    }
  }
  
  if (Est$model=="multidim.Dirac") {
    lines <-c(lines,
      sprintf("%-*s", width, paste("par1:", "location", "--", "initialized at"))
    )
    for (j in 1:length(Est$estimator)) {
      lines <-c(lines,
        sprintf("%-*s", width, paste("     ",round(Est$par1init[j], digits=4)))
      )
    }
    lines <-c(lines,
      sprintf("%-*s", width, "      estimated value:")
    )
    for (j in 1:length(Est$par1init)){
      lines <-c(lines,
        sprintf("%-*s", width, paste("     ",round(Est$estimator[j], digits=4)))
      )
    }
  }
  
  if (Est$model=="multidim.Gaussian.loc") {
    lines <-c(lines,
      sprintf("%-*s", width, paste("par1:", "mean", "--", "initialized at"))
    )
    for (j in 1:length(Est$par1init)) {
      lines <-c(lines,
        sprintf("%-*s", width, paste("     ",round(Est$par1init[j], digits=4)))
      )
    }
    lines <-c(lines,
      " ",
      sprintf("%-*s", width, "      estimated value:")
    )
    for (j in 1:length(Est$par1init)) {
      lines <-c(lines,
        sprintf("%-*s", width, paste("     ",round(Est$estimator[j], digits=4)))
      )
    }
    lines <-c(lines,
      " ",
      sprintf("%-*s", width,paste("par2:", "scalar standard deviation", "--", "initialized at", round(Est$par2init, digits=4)))
    )
  }
  
  if (Est$model=="multidim.Gaussian.scale") {
    lines <-c(lines,
      sprintf("%-*s", width, paste("par1:", "mean", "--", "fixed by user:"))
    )
    for (j in 1:length(Est$par1init)){
      lines <-c(lines,
        sprintf("%-*s", width, paste("     ",round(Est$par1init[j], digits=4)))
      )
    }
    lines <-c(lines,
      " ",
      sprintf("%-*s", width,paste("par2:", "U, where U'U = var-cov matrix", "--", "initialized at"))
    )
    for (j in 1:length(Est$par1init)) {
      line1="     "
      for (k in 1:length(Est$par1init)) line1<-paste(line1,round(Est$par2init[j,k], digits=4))
      lines = c(lines,
        sprintf("%-*s", width, line1)
      )
    }
    lines = c(lines,
      sprintf("%-*s", width, "      estimated value:")
    )
    for (j in 1:length(Est$par1init)) {
      line1="     "
      for (k in 1:length(Est$par1init)) line1<-paste(line1,round(Est$estimator[j,k], digits=4))
      lines = c(lines,
        sprintf("%-*s", width, line1)
      )
    }
  }
  
  if (Est$model=="multidim.Gaussian") {
    lines = c(lines,
      sprintf("%-*s", width, paste("par1:", "mean", "--", "initialized at"))
    )
    for (j in 1:length(Est$par1init)) {
      lines = c(lines,
                sprintf("%-*s", width, paste("     ",round(Est$par1init[j], digits=4)))
      )
    }
    lines = c(lines,
              sprintf("%-*s", width, "      estimated value:")
    )
    for (j in 1:length(Est$par1init)) {
      lines = c(lines,
                sprintf("%-*s", width, paste("     ",round(Est$estimator$par1[j], digits=4)))
      )
    }
    
    lines = c(lines,
      " ",
      sprintf("%-*s", width, paste("par2:", "U, where U'U = var-cov matrix", "--", "initialized at"))
    )
    for (j in 1:length(Est$par1init)) {
      line1<-c("     ")
      for (k in 1:length(Est$par1init)) line1<-paste(line1,round(Est$par2init[j,k], digits=4))
      lines = c(lines,
        sprintf("%-*s", width, line1)
      )
    }
    lines = c(lines,
      sprintf("%-*s", width, "      estimated value:")
    )
    for (j in 1:length(Est$par1init)) {
      line1<-c("     ")
      for (k in 1:length(Est$par1init)) line1<-paste(line1,round(Est$estimator$par2[j,k], digits=4))
      lines = c(lines,
        sprintf("%-*s", width, line1)
      )
    }
  }
  
  lines <- c(lines, "=========================================================")
  
  # Print all lines at once
  cat(paste(lines, collapse = "\n"), "\n")
}
