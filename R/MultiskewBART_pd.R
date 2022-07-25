#' Title
#' Fit MultiskewBART with partial dependence
#'
#' Fits the MultiskewBART model and returns a list with function estimates and
#' estimates of the parameters of the skew-normal distribution. Also computes
#' Friedman's partial dependence function for a specified variable.
#' See the real_data_GAAD vignette by running browseVignettes("skewBART") for
#' an illustration.
#' 
#' @param X NxP matrix of training data covariates.
#' @param Y Nx2 vector of training data responses.
#' @param x_grid grid of values along which to compute the partial dependence
#'   function. Must be a data.frame with columns corresponding to the different
#'   columns of X the partial dependence is to be computed on. A default is
#'   provided for continuous variables provided that only one variable is used.
#' @param vars the variables to compute the partial dependence function of; must
#'   be a string.
#' @param hypers A list of hyperparameters, typically constructed with the
#'   Hypers function
#' @param opts A list of options for running the Markov chain, typically
#'   constructed with the Opts function.
#'
#' @return Returns a list with the following components:
#' \itemize{
#'   \item f_hat_train: posterior samples of function fit to the training data
#'      for each iteration of the chain
#'   \item f_hat_test: posterior samples of function fit to the testing data for
#'      each iteration of the chain
#'   \item y_hat_train: posterior samples of predicted outcome on the training
#'      data for each iteration of the chain; not equal to f_hat_train because the errors are not mean 0
#'   \item y_hat_test: posterior samples of function fit to the testing data for
#'     each iteration of the chain; not equal to f_hat_test because the errors are not mean 0
#'   \item sigma: posterior samples of the scale of the skew-normal distribution
#'   \item alpha: posterior samples of the skewness parameter of the skew-normal
#'      distribution
#'   \item tau: posterior samples of the standard deviation of the latent normal
#'      distribution in the parameter-expanded representation of the skew-normal; usually not of direct interest.
#'   \item lambda: posterior samples of the regression coefficient of the
#'      truncated normal variables in the parameter-expanded representation of the skew-normal distribution; usually not of direct interest.
#'   \item loo: the PSIS-loo computed with loo function in the loo package.
#'   \item partial_dependence_samples: Samples of the partial dependence
#'      function for both y_hat and f_hat. Stored as a data frame with each row
#'     corresponding to an iteration of the chain, a particular outcome,
#'     and combination of the predictors
#'   \item partial_dependence_summary: Summary of the partial dependence
#'      function. Each row corresponds to a combination of variables, and
#'      includes the posterior mean and a 95% posterior credible interval for
#'      f_hat and y_hat.
#' }
#' @export
MultiskewBART_pd <- function(X, Y, vars, x_grid = NULL, hypers = NULL, opts = NULL) {
  y_hat <- f_hat <- NULL
  if(is.null(colnames(X))) {
    stop("Need to have named columns of X to make a partial dependence plot")
  }
  
  if(is.null(colnames(Y))) {
    stopifnot(ncol(Y) == 2)
    colnames(Y) <- c("Var1", "Var2")
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
  
  fitted_mskewbart <- MultiskewBART(X, Y, test_X, hypers, opts)
  n_save <- nrow(fitted_mskewbart$lambda)
  n_obs  <- nrow(X)
  n_grid <- nrow(x_grid)
  
  f_hat_pd_arrays <- array(fitted_mskewbart$f_hat_test, c(n_obs, n_grid, 2, n_save))
  y_hat_pd_arrays <- array(fitted_mskewbart$y_hat_test, c(n_obs, n_grid, 2, n_save))
  process_array <- function(f_hat_pd_array, y_hat_pd_array) {
    f_hat_pd <- apply(f_hat_pd_array, c(1,3), mean)
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
    
    return(list(f_df = f_df, f_df_summary = f_df_summary))
    
  }
  
  f_hat_pd_array_1 <- aperm(f_hat_pd_arrays[,,1,], c(3, 1, 2))
  f_hat_pd_array_2 <- aperm(f_hat_pd_arrays[,,2,], c(3, 1, 2))
  y_hat_pd_array_1 <- aperm(y_hat_pd_arrays[,,1,], c(3, 1, 2))
  y_hat_pd_array_2 <- aperm(y_hat_pd_arrays[,,2,], c(3, 1, 2))
  
  tmp_1 <- process_array(f_hat_pd_array_1, y_hat_pd_array_1)
  tmp_2 <- process_array(f_hat_pd_array_2, y_hat_pd_array_2)
  
  tmp_1$f_df$outcome <- colnames(Y)[1]
  tmp_1$f_df_summary$outcome <- colnames(Y)[1]
  tmp_2$f_df$outcome <- colnames(Y)[2]
  tmp_2$f_df_summary$outcome <- colnames(Y)[2]

  fitted_mskewbart$partial_dependence_samples <- rbind(tmp_1$f_df, tmp_2$f_df)
  fitted_mskewbart$partial_dependence_summary <- rbind(tmp_1$f_df_summary, 
                                                       tmp_2$f_df_summary)
  
  return(fitted_mskewbart)

}
