#' Create hyperparameter object for skewBART
#'
#' Creates a list which holds all the hyperparameters for use with the skewBART.
#'
#'
#' @param X NxP matrix of training data covariates.
#' @param Y Nx1 vector of training data response.
#' @param group For each column of X, gives the associated group; mostly used for categorical variables with more than 2 levels.
#' @param alpha Positive constant controlling the sparsity level
#' @param beta Parameter penalizing tree depth in the branching process prior
#' @param gamma Parameter penalizing new nodes in the branching process prior
#' @param k Related to the signal-to-noise ratio, sigma_mu = 3.5 / (sqrt(num_tree) * k). BART defaults to k = 2.
#' @param sigma_hat A prior guess at the conditional variance of Y. If not provided, this is estimated empirically by linear regression.
#' @param shape Shape parameter for gating probabilities
#' @param width Bandwidth of gating probabilities
#' @param num_tree Number of trees in the ensemble
#' @param alpha_scale Scale of the prior for alpha; if not provided, defaults to P
#' @param alpha_shape_1 Shape parameter for prior on alpha; if not provided, defaults to 0.5
#' @param alpha_shape_2 Shape parameter for prior on alpha; if not provided, defaults to 1.0
#' @param tau_rate The rate parameter for an exponential prior on the gating bandwidths.
#'
#' @return Returns a list containing the function arguments.
UHypers <- function(X,Y, group = NULL, alpha = 1, beta = 2, gamma = 0.95, k = 2,
                   sigma_hat = NULL, shape = 1, width = 0.1, num_tree = 20,
                   alpha_scale = NULL, alpha_shape_1 = 0.5,
                   alpha_shape_2 = 1, tau_rate = 10) {

  if(is.null(alpha_scale)) alpha_scale <- ncol(X)

  out                                  <- list()

  out$alpha                            <- alpha
  out$beta                             <- beta
  out$gamma                            <- gamma
  out$sigma_mu_hat                     <- 3.5 / (k * sqrt(num_tree))
  out$k                                <- k
  out$num_tree                         <- num_tree
  out$shape                            <- shape
  if(is.null(group)) {
    out$group                          <- 1:ncol(X) - 1
  } else {
    out$group                          <- group - 1
  }

  Y                                    <- scale(Y)
  if(is.null(sigma_hat))
    sigma_hat                          <- GetSigma(X,Y)

  out$sigma_hat                        <- sigma_hat

  out$alpha_scale                      <- alpha_scale
  out$alpha_shape_1                    <- alpha_shape_1
  out$alpha_shape_2                    <- alpha_shape_2
  out$tau_rate                         <- tau_rate
  out$temperature                      <- 1

  return(out)

}



#' Fit the skewBART model
#'
#' Fits the skewBART model and returns a list with function estimates and
#' estimates of the parameters of the skew-normal error distribution
#'
#' @param X NxP matrix of training data covariates.
#' @param Y Nx1 vector of training data response.
#' @param test_X MxP matrix of test data covariates
#' @param hypers A list of hyperparameters, typically constructed with the UHypers function
#' @param opts A list of options for running the Markov chain, typically constructed with the UOpts function.
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
#' }
#'
#' @examples
#' \donttest{
#'
#' library(zeallot)
#' sim_fried <- function(N, P, alpha, sigma) {
#'   lambda <- alpha * sigma/sqrt(1+alpha^2)
#'   tau <- sigma/sqrt(1+alpha^2)
#'   X <- matrix(runif(N * P), nrow = N)
#'   mu <- 10 * sin(pi * X[,1] * X[,2]) + 20 * (X[,3] - 0.5)^2 + 10 * X[,4] + 5 * X[,5]
#'   Z <- abs(rnorm(N, mean=0, sd=1) )
#'   Y <- mu + lambda * Z + rnorm(N, mean=0, sd=sqrt(tau))
#'   EY <- mu + lambda * sqrt(2/pi)
#'   return(list(X = X, Y = Y, EY = EY, mu = mu, Z=Z, tau = tau, lambda = lambda))
#' }
#'
#' ## Traning dataset : n = 250 observations, P = 5 covariates, sigma = 2, alpha = 5 ----
#'
#' set.seed(12345)
#' c(X,Y,EY,mu,Z,tau,lambda) %<-% sim_fried(250, 5, 5, 2)
#'
#' ## Test dataset : n = 100 observations, P = 5 covariates, sigma = 2, alpha = 5 ----
#'
#' c(test_X,test_Y,test_EY,test_mu,test_Z,test_tau,test_lambda)  %<-% sim_fried(100, 5, 5 ,2)
#'
#' ## Fit ----
#'
#' hypers <- UHypers(X, Y)
#' opts <- UOpts(num_burn = 5000, num_save = 5000)
#' fitted_skewbart <- skewBART(X, Y, test_X, hypers, opts)
#'
#' ## Traceplot of alpha samples and assessment of how well we recover the nonparametric function ----
#'
#' par(mfrow = c(1,2))
#' plot(fitted_skewbart$alpha)
#' plot(colMeans(fitted_skewbart$f_hat_test), test_mu, pch = 2)
#' abline(a=0,b=1, col = 'green', lwd = 3)
#' }
#'
skewBART <- function(X, Y, test_X, hypers = NULL, opts = NULL){

  if(is.null(opts)) opts <- UOpts()
  if(is.null(hypers)) hypers <- UHypers(X, Y)
  iter <- opts$num_burn + opts$num_save
  burn <- opts$num_burn

  n_train <- nrow(X)
  mu_hat <- matrix(NA, iter, n_train)
  mu_hat_test <- matrix(NA, iter, nrow(test_X))
  opts <- UOpts()
  forest <- MakeForest(hypers, opts)
  EST_Tau <- EST_Lambda <-  NULL
  Z <- rep(1,n_train);  Lambda <- 1
  XXtest <- quantile_normalize(X, test_X)
  X <- XXtest$X
  test_X <- XXtest$test_X

  scale_Y <- sd(Y)
  center_Y <- mean(Y)
  Y <- (Y - center_Y) / scale_Y
  #like_skew <- matrix(NA, n_train, iter)

  for(i in 1:iter){
    R <- Y - Lambda*Z
    mu_hat[i,] <- forest$do_gibbs(X, R, X, 1)
    mu_hat_test[i,] <- forest$predict(test_X)
    Tau <- forest$get_params()[["sigma"]]^2
    delta <- Y - mu_hat[i,]
    Z <- rtruncnorm(1, a=0, b=Inf, mean = delta*Lambda/(Lambda^2 + Tau), sd = sqrt(Tau/(Lambda^2+Tau)))
    Lambda <- rnorm(1, mean = (t(Z) %*% delta)/(t(Z)%*%Z), sd = 1/sqrt(t(Z)%*%Z/Tau ))
    EST_Tau <- c(EST_Tau, Tau)
    EST_Lambda <- c(EST_Lambda, Lambda)
    #like_skew[,i] <- dsn(Y_scaled - mu_hat[i,], tau=0, omega= Omega, alpha = Alpha , dp=NULL, log=TRUE)
    if(i %% 100 == 0) (cat("\rFinishing iteration", i, "of",iter,"\t"))
  }


  ## Rescale stuff: f_hat
  f_hat_train      <- mu_hat[-c(1:burn), ] * scale_Y + center_Y
  f_hat_test       <- mu_hat_test[-c(1:burn),] * scale_Y + center_Y 
  f_hat_train_mean <- colMeans(f_hat_train)
  f_hat_test_mean  <- colMeans(f_hat_test)
  
  ## Rescale stuff: tau and lambda
  lambda <- EST_Lambda[-c(1:burn)] * scale_Y
  tau <- sqrt(EST_Tau[-c(1:burn)]  * scale_Y^2)
  
  ## Compute alpha and sigma
  alpha <- lambda / tau
  sigma <- tau * sqrt(1 + alpha^2)
  
  ## Compute means
  y_hat_train <- f_hat_train + lambda * sqrt(2 / pi)
  y_hat_test  <- f_hat_test + lambda * sqrt(2 / pi)
  y_hat_train_mean <- colMeans(y_hat_train)
  y_hat_test_mean <- colMeans(y_hat_test)
  
  ## Compute the loo
  Y <- Y * scale_Y + center_Y
  like_iter <- function(t) {
    dsn(Y, xi = f_hat_train[t,], omega = sigma[t], alpha = alpha[t], log = TRUE)
  }
  likelihood_mat <- t(sapply(1:length(alpha), like_iter))
  #loo_out <- loo(t(like_skew))
  #likelihood <- rowSums(likelihood_mat)
  
  return(list(y_hat_train = y_hat_train, y_hat_test = y_hat_test,
              y_hat_train_mean = y_hat_train_mean, y_hat_test_mean = y_hat_test_mean,
              f_hat_train = f_hat_train, f_hat_test = f_hat_test, 
              f_hat_train_mean = f_hat_train_mean, 
              f_hat_test_mean = f_hat_test_mean,
              sigma = sigma, tau = tau, alpha = alpha, lambda = lambda, 
              likelihood_mat = likelihood_mat))
}



