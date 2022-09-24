#' Title
#' Fit skewBART with partial dependence
#'
#' Fits the skewBART model and returns a list with function estimates and
#' estimates of the parameters of the skew-normal distribution. Also computes
#' Friedman's partial dependence function for a specified variable.
#' See the real_data_GAAD vignette by running browseVignettes("skewBART") for
#' an illustration.
#' 
#' @param X NxP matrix of training data covariates.
#' @param Y Nx1 vector of training data response.
#' @param x_grid grid of values along which to compute the partial dependence
#'   function. Must be a data.frame with columns corresponding to the different
#'   columns of X the partial dependence is to be computed on. A default is
#'   provided for continuous variables provided that only one variable is used.
#' @param vars the variables to compute the partial dependence function of; must
#'   be a string.
#' @param hypers A list of hyperparameters, typically constructed with the
#'   UHypers function
#' @param opts A list of options for running the Markov chain, typically
#'   constructed with the UOpts function.
#'
#' @return Returns a list with the following components:
#' \itemize{
#'   \item f_hat_train: posterior samples of function fit to the training data for each iteration of the chain
#'   \item f_hat_test: posterior samples of function fit to the testing data for each iteration of the chain
#'   \item y_hat_train: posterior samples of predicted outcome on the training data for each iteration of the chain; not equal to f_hat_train because the errors are not mean 0
#'   \item y_hat_test: posterior samples of function fit to the testing data for each iteration of the chain; not equal to f_hat_test because the errors are not mean 0
#'   \item sigma: posterior samples of the scale of the skew-normal distribution
#'   \item alpha: posterior samples of the skewness parameter of the skew-normal distribution
#'   \item tau: posterior samples of the standard deviation of the latent normal distribution in the parameter-expanded representation of the skew-normal; usually not of direct interest.
#'   \item lambda: posterior samples of the regression coefficient of the truncated normal variables in the parameter-expanded representation of the skew-normal distribution; usually not of direct interest.
#'   \item likelihood_mat: (N x num_save) matrix of log-likelihood values to calculate the log pseudo marginal likelihood (LPML)
#'   \item partial_dependence_samples: Samples of the partial dependence function for both y_hat and f_hat. Stored as a data frame with each row corresponding to an iteration of the chain and combination of variables
#'   \item partial_dependence_summary: Summary of the partial dependence function. Each row corresponds to a combination of variables, and includes the posterior mean and a 95 percent posterior credible interval for f_hat and y_hat.
#' }
#' @export
skewBART_pd <- function(X, Y, vars, x_grid = NULL, hypers = NULL, opts = NULL) {

  y_hat <- f_hat <- NULL
  
  if(is.null(colnames(X))) {
    stop("Need to have named columns of X to make a partial dependence plot")
  }
  
  f_expand <- function(x) {
    XX <- X
    for(v in vars) {
      XX[,v] <- x[[v]]
    }
    return(XX)
  }
  
  if(is.null(x_grid)) {
    stopifnot(length(vars) == 1)
    x_grid <- data.frame(n = 1:20)
    v <- vars[[1]]
    m <- min(X[,vars[[1]]])
    M <- max(X[,vars[[1]]])
    x_grid[[v]] <- seq(from = m, to = M, length = 20)
  }

  test_X <- do.call(rbind, lapply(1:nrow(x_grid), function(i) f_expand(x_grid[i,])))
  
  fitted_skewbart <- skewBART(X, Y, test_X, hypers, opts)
  n_save <- length(fitted_skewbart$alpha)
  n_obs  <- nrow(X)
  n_grid <- nrow(x_grid)

  f_hat_pd_array <- array(fitted_skewbart$f_hat_test, c(n_save, n_obs, n_grid))
  f_hat_pd <- apply(f_hat_pd_array, c(1,3), mean)
  y_hat_pd_array <- array(fitted_skewbart$y_hat_test, c(n_save, n_obs, n_grid))
  y_hat_pd <- apply(y_hat_pd_array, c(1,3), mean)
  
  f_dfs <- list()
  for(i in 1:nrow(x_grid)) {
    f_dfs[[i]] <- data.frame(Iteration = 1:n_save)
    for(v in vars) f_dfs[[i]][[v]] <- x_grid[i,v]
    f_dfs[[i]]$f_hat <- f_hat_pd[,i]
    f_dfs[[i]]$y_hat <- y_hat_pd[,i]
  }
  f_df <- do.call(rbind, f_dfs)
  f_df_summary <- f_df %>% group_by_at(vars) %>% 
    summarise(f_hat_mean = mean(f_hat), f_hat_025 = quantile(f_hat, 0.025), 
              f_hat_975 = quantile(f_hat, 0.975), y_hat_mean = mean(y_hat), 
              y_hat_025 = quantile(y_hat, 0.025), 
              y_hat_975 = quantile(y_hat, 0.975), .groups = 'drop')

  fitted_skewbart$partial_dependence_samples <- f_df
  fitted_skewbart$partial_dependence_summary <- f_df_summary
  
  # fitted_skewbart$v           <- v
  # fitted_skewbart$f_hat_pd      <- f_hat_pd
  # fitted_skewbart$f_hat_pd_mean <- colMeans(f_hat_pd)
  # fitted_skewbart$x_grid        <- x_grid
  
  return(fitted_skewbart)
}
