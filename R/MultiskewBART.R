#' Create hyperparameter object for MultiskewBART
#'
#' Creates a list which holds all the hyperparameters for use with the MultiskewBART.
#'
#' @param X NxP matrix of training data covariates.
#' @param Y Nxk matrix of training data response.
#' @param group For each column of X, gives the associated group
#' @param alpha Positive constant controlling the sparsity level
#' @param beta Parameter penalizing tree depth in the branching process prior
#' @param gamma Parameter penalizing new nodes in the branching process prior
#' @param k Related to the signal-to-noise ratio, sigma_mu = 0.5 / (sqrt(num_tree) * k). BART defaults to k = 2.
#' @param num_tree Number of trees in the ensemble
#'
#' @return Returns a list containing the function arguments.

Hypers <- function(X, Y, group = NULL, alpha = 1, beta = 2, gamma = 0.95, k = 2,
                   num_tree = 20) {


  J <- ncol(Y)
  P <- ncol(X)
  Y <- scale(Y)

  if(is.null(group)) group <- Matrix(diag(P), sparse = TRUE)
  if(J != 2) stop("CURRENTLY ONLY SUPPORT TWO DIMENSIONAL RESPONSE")

  Sigma_mu_hat <- diag(2) * (3 / k / sqrt(num_tree))^2

  ## Get Sigma hyperparameters
  fit_lm_Y_1 <- lm(Y[,1] ~ X)
  fit_lm_Y_2 <- lm(Y[,2] ~ X + Y[,1])
  fit_lm_Y_2_x <- lm(Y[,2] ~ X)

  sigma_1_hat <- summary(fit_lm_Y_1)$sigma
  sigma_2_hat <- summary(fit_lm_Y_2)$sigma
  r_hat       <- sigma_2_hat / sigma_1_hat
  Sigma_hat   <- cov(cbind(residuals(fit_lm_Y_1), residuals(fit_lm_Y_2_x)))

  S0 <- Sigma_hat
  nu <- ncol(X)

  out <- list(alpha = alpha, beta = beta, gamma = gamma,
              Sigma_mu_hat = Sigma_mu_hat, k = k, num_tree = num_tree,
              Sigma_hat = Sigma_hat, temperature = 1,
              sigma_1_hat = sigma_1_hat, sigma_2_hat = sigma_2_hat,
              r_hat = r_hat, group = group, nu = nu, S0 = S0)

  return(out)

}


#' Fit the MultiskewBART model
#'
#' Fits the MultiskewBART model of Um et al. (2021+). The model is of the form \deqn{Y_i = \mu(X_i) + \epsilon_i} where \eqn{\epsilon_i} has a multivariate skew-normal distribution.
#'
#' @param X NxP matrix of training data covariates.
#' @param Y Nxk matrix of training data response.
#' @param test_X MxP matrix of test data covariates.
#' @param hypers a list containing the hyperparameters of the model, usually constructed using the function Hypers().
#' @param opts a list containing options for running the chain, usually constructed using the function Opts().
#' @param do_skew logical, if true fit the skew-normal model; otherwise, fit a multivariate normal.
#' @param Wishart logical, if true use Wishart prior distribution for covariance matrix; otherwise, update the standard deviations and correlations separately.
#'
#' @return Returns a list with the following components:
#' \itemize{
#'   \item f_hat_train: fit of the regression function to the training data for
#'         each iteration of the chain; note that the errors are _not_ mean 0,
#'         so this does not give the expected value.
#'   \item f_hat_test: fit of the regression function to the testing data for each
#'         iteration of the chain; note that the errors are _not_ mean 0, so
#'         this does not give the expected value.
#'   \item y_hat_train: predicted values for the training data for
#'         each iteration of the chain
#'   \item y_hat_test: predicted values for the testing data for each
#'         iteration of the chain
#'   \item Sigma: posterior samples of the covariance matrix
#'   \item lambda: posterior samples of skewness parameters
#'   \item loo: the PSIS-loo computed with loo function in the loo package
#' }
#' @export
#'
MultiskewBART <- function(X, Y, test_X, hypers = NULL, opts = NULL, do_skew = TRUE, Wishart = FALSE){

  mean_Z <- Sigma_Z <- NULL

  if(is.null(hypers)) hypers <- Hypers(X,Y)
  if(is.null(opts)) opts <- Opts()
  iter <- opts$num_burn + opts$num_save
  burn <- opts$num_burn

  ## Used to get the log likelihood and loo
  log_like_fun<- function(yy, Sig, D){
    cov_d <- Sig + D^2
    obj_p <- drop(D %*% solve(cov_d) %*% yy)
    cov_p <- solve(diag(2) + D %*% solve(Sig) %*% D)
    ## pdf <- 2*log(2) + dmvnorm(yy, mean=c(0,0), sigma = cov_d,log=TRUE) + 
    ##   log(TruncatedNormal::pmvnorm(lb= -Inf, ub = obj_p, mu = c(0,0), sigma=cov_p)[1])
    pdf <- 2*log(2) + dmvnorm(yy, mean=c(0,0), sigma = cov_d,log=TRUE) + 
      log(mvtnorm::pmvnorm(lower= -Inf, upper = obj_p, mean = c(0,0), sigma=cov_p)[1])
    return(pdf)
  }


  my_forest <- multi_MakeForest(hypers = hypers, opts = opts)
  Lam_est <- NULL
  Sigma_out <- array(NA, c(2,2,iter))
  mu_out <- array(NA, c(nrow(Y), 2, iter))
  mu_test_out <- array(NA, c(nrow(test_X), 2, iter))
  Sigma_chain <- hypers$Sigma_hat
  l <- c(0, 0);  u <- c(Inf, Inf)
  if(Wishart) {
    S0 <- hypers$S0
    nu <- hypers$nu
   }

  XXtest <- quantile_normalize(X, test_X)
  X <- XXtest$X
  test_X <- XXtest$test_X

  Y_scaled <- scale(Y)
  center_Y <- attributes(Y_scaled)$`scaled:center`
  scale_Y <- attributes(Y_scaled)$`scaled:scale`
  Y <- Y_scaled

  Z <- cbind(rep(1,nrow(Y)), rep(1,nrow(Y))) * do_skew
  Lambda <- c(1,1)
  R <- Y_scaled - Z %*% diag(Lambda)

  for(j in 1:iter){
    R <- Y - Z %*% diag(Lambda)
    my_forest$set_sigma(Sigma_chain)
    mu_hat_chain <- my_forest$do_gibbs(X, R, X, 1)[,,1]
    delta <- R - mu_hat_chain
    if(Wishart) {
      B <- delta - Z %*% diag(Lambda)
      Sigma_chain <- solve(rwish( nrow(Y_scaled) + nu , solve(t(B) %*% B  + S0) ))
    }
    Sigma_chain <- update_sigma(delta, Sigma_chain, hypers)
    mu_hat_test <- my_forest$predict(test_X)
    if(do_skew) {
      c(Lambda, mean_Z, Sigma_Z) %<-% update_z_multi_2(Y, mu_hat_chain, Sigma_chain, Z)
      Lambda <- as.numeric(Lambda)
      Lam_est <- rbind(Lam_est, Lambda)
      Z <- t(sapply(1:nrow(Z), function(i) rtmvnorm(1, mu = mean_Z[i,], sigma=Sigma_Z, lb=l, ub=u)))
    }
    Sigma_out[,,j] <- Sigma_chain
    mu_out[,,j] <- mu_hat_chain
    mu_test_out[,,j] <- mu_hat_test
    if(j %% opts$num_print == 0) {
      cat("\rFinishing iteration", j, "of", iter)
    }
    
  }

  if(do_skew) {
    lambda <- t(t(Lam_est[(burn+1):iter,]) * scale_Y)
  } else {
    lambda <- matrix(0, nrow = iter - burn, ncol = 2)
  }
  mu <- mu_out[,,(burn+1):iter]
  mu[,1,] <- mu[,1,] * scale_Y[1] + center_Y[1]
  mu[,2,] <- mu[,2,] * scale_Y[2] + center_Y[2]
  y_hat_train <- mu
  y_hat_train[,1,] <- y_hat_train[,1,] + lambda[,1] * sqrt(2 / pi)
  y_hat_train[,2,] <- y_hat_train[,2,] + lambda[,2] * sqrt(2 / pi)

  mu_test <- mu_test_out[,,(burn+1):iter]
  mu_test[,1,] <- mu_test[,1,] * scale_Y[1] + center_Y[1]
  mu_test[,2,] <- mu_test[,2,] * scale_Y[2] + center_Y[2]
  y_hat_test <- mu_test
  y_hat_test[,1,] <- y_hat_test[,1,] + lambda[,1] * sqrt(2 / pi)
  y_hat_test[,2,] <- y_hat_test[,2,] + lambda[,2] * sqrt(2 / pi) 
  Sigma <- Sigma_out[,,(burn+1):iter]
  for(j in 1:2) {
    for(k in 1:2) {
      Sigma[j,k,] <- Sigma[j,k,] * scale_Y[j] * scale_Y[k]
    }
  }

  y_hat_train_mean <- apply(y_hat_train, c(1,2), mean)
  y_hat_test_mean <- apply(y_hat_test, c(1,2), mean)
  mu_train_mean <- apply(mu, c(1,2), mean)
  mu_test_mean <- apply(mu_test, c(1,2), mean)

  ## Doing the log-likelihood
  num_save <- iter - burn
  like_skew <- matrix(NA, nrow = num_save, ncol = nrow(Y))
  Y <- Y_scaled
  Y[,1] <- Y_scaled[,1] * scale_Y[1] + center_Y[1]
  Y[,2] <- Y_scaled[,2] * scale_Y[2] + center_Y[2]

  for(k in 1:num_save) {
    Y_res <- Y - mu[,,k]
    f_apply <- function(i) {
      log_like_fun(yy = Y_res[i,], Sig = Sigma[,,k], D = diag(lambda[k,]))
    }
    like_skew[k,] <- sapply(1:nrow(Y_res), f_apply)
  }
  loo_out <- loo(like_skew)


  return(list(y_hat_train = y_hat_train, 
              y_hat_test = y_hat_test, 
              y_hat_train_mean = y_hat_train_mean, 
              y_hat_test_mean = y_hat_test_mean, 
              f_hat_train = mu,
              f_hat_test = mu_test,
              f_hat_train_mean = mu_train_mean,
              f_hat_test_mean = mu_test_mean,
              lambda = lambda, Sigma = Sigma, loo = loo_out))

  

}


