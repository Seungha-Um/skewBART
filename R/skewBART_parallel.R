#' Fit the skewBART model with parallel computation
#'
#'
#' @param X NxP matrix of training data covariates.
#' @param Y Nx1 vector of training data response.
#' @param test_X MxP matrix of test data covariates
#' @param hypers A list of hyperparameters, typically constructed with the UHypers function
#' @param opts A list of options for running the Markov chain, typically constructed with the UOpts function.
#' @param cores Number of cores to run skewBART in parallel.
#' @param nice Job niceness. The niceness scale ranges from 0 (highest) to 19 (lowest);; if not provided, defaults to 19
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

skewBART_parallel <- function(X, Y, test_X, hypers = NULL, opts = NULL, 
                              cores = 2L, nice = 19L){
  
  if(.Platform$OS.type!='unix')
    stop('parallel::mcparallel/mccollect do not exist on windows')
  
  if(is.null(opts)) opts <- UOpts()
  if(is.null(hypers)) hypers <- UHypers(X, Y)
  
  parallel::mc.reset.stream()
  
  cores.detected <- parallel::detectCores()
  
  if(cores > cores.detected) cores <- cores.detected
  mc_num_save <- ceiling(opts$num_save/cores)
  opts$num_save <- mc_num_save
  
  for(i in 1:cores) {
    parallel::mcparallel({tools::psnice(value=nice);
                skewBART(X = X, Y = Y, test_X = test_X, 
                hypers = hypers, opts = opts)},
      silent=(i!=1))
  }
  
  post.list <- parallel::mccollect()

  post <- post.list[[1]]
  
  if(cores==1) return(post)
  else {
    post$num_save <- cores * mc_num_save
      
    for(i in 2:cores) {
      post$yhat.train <- rbind(post$y_hat_train,
                               post.list[[i]]$y_hat_train)

      post$y_hat_test <- rbind(post$y_hat_test,
                                post.list[[i]]$y_hat_test)

      post$f_hat_train <- rbind(post$f_hat_train,
                                post.list[[i]]$f_hat_train)
      
      post$f_hat_test <-  rbind(post$f_hat_test,
                                post.list[[i]]$f_hat_test)
      
      #post$likelihood <- c(post$likelihood, post.list[[i]]$likelihood)
      post$likelihood_mat <- rbind(post$likelihood_mat, post.list[[i]]$likelihood_mat)

      post$sigma <- c(post$sigma, post.list[[i]]$sigma)
      post$tau <-  c(post$tau, post.list[[i]]$tau)
      post$alpha <-  c(post$alpha, post.list[[i]]$alpha)
      post$lambda <-  c(post$lambda, post.list[[i]]$lambda)

    }
        
    post$f_hat_train_mean <- apply(post$f_hat_train, 2, mean)
    post$f_hat_test_mean  <- apply(post$f_hat_test, 2, mean)
    post$y_hat_train_mean <- apply(post$y_hat_train, 2, mean)
    post$y_hat_test_mean <- apply(post$y_hat_test, 2, mean)
    
    return(post)
    
  }
}  
  
