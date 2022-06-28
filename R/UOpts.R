#' MCMC options for skewBART
#'
#' Creates a list which provides the parameters for running the Markov chain.
#'
#' @param num_burn Number of warmup iterations for the chain.
#' @param num_thin Thinning interval for the chain.
#' @param num_save The number of samples to collect; in total, num_burn + num_save * num_thin iterations are run
#' @param num_print Interval for how often to print the chain's progress
#' @param update_sigma_mu If true, sigma_mu/k are updated, with a half-Cauchy prior on sigma_mu centered at the initial guess
#' @param update_s If true, s is updated using the Dirichlet prior.
#' @param update_alpha If true, alpha is updated using a scaled beta prime prior
#' @param update_beta If true, beta is updated using a Normal(0,2^2) prior
#' @param update_gamma If true, gamma is updated using a Uniform(0.5, 1) prior
#' @param update_tau If true, tau is updated for each tree
#' @param update_tau_mean If true, the mean of tau is updated
#'
#' @return Returns a list containing the function arguments
#'
UOpts <- function(num_burn = 2500, num_thin = 1, num_save = 2500, num_print = 100,
                 update_sigma_mu = TRUE, update_s = FALSE, update_alpha = TRUE,
                 update_beta = FALSE, update_gamma = FALSE, update_tau = TRUE,
                 update_tau_mean = FALSE) {
  out <- list()
  out$num_burn        <- num_burn
  out$num_thin        <- num_thin
  out$num_save        <- num_save
  out$num_print       <- num_print
  out$update_sigma_mu <- update_sigma_mu
  out$update_s         <- update_s
  out$update_alpha    <- update_alpha
  out$update_beta     <- update_beta
  out$update_gamma    <- update_gamma
  out$update_tau      <- update_tau
  out$update_tau_mean <- update_tau_mean
  out$update_sigma    <- TRUE
  out$update_num_tree <- FALSE

  return(out)

}

GetSigma <- function(X,Y) {

  stopifnot(is.matrix(X) | is.data.frame(X))

  if(is.data.frame(X)) {
    X <- model.matrix(~.-1, data = X)
  }


  fit <- cv.glmnet(x = X, y = Y)
  fitted <- predict(fit, X)
  sigma_hat <- sqrt(mean((fitted - Y)^2))

  return(sigma_hat)

}

MakeForest <- function(hypers, opts) {
  mf <- Module(module = "mod_forest", PACKAGE = "skewBART")
  return(new(mf$UForest, hypers, opts))
}
