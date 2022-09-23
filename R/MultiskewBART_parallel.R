#' Fit the MultiskewBART model with parallel computation
#'
#'
#' @param X NxP matrix of training data covariates.
#' @param Y Nxk matrix of training data response.
#' @param test_X MxP matrix of test data covariates.
#' @param hypers a list containing the hyperparameters of the model, usually constructed using the function Hypers().
#' @param opts a list containing options for running the chain, usually constructed using the function Opts().
#' @param do_skew logical, if true fit the skew-normal model; otherwise, fit a multivariate normal.
#' @param Wishart logical, if true use Wishart prior distribution for covariance matrix; otherwise, update the standard deviations and correlations separately.
#' @param cores Number of cores to run MultiskewBART in parallel.
#' @param nice Job niceness. The niceness scale ranges from 0 (highest) to 19 (lowest);; if not provided, defaults to 19

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
#'   \item likelihood_mat: (N x num_save) matrix of log-likelihood values to calculate the log pseudo marginal likelihood (LPML)
#' }
#' @export
#'
MultiskewBART_parallel <- function(X, Y, test_X, hypers = NULL, opts = NULL, 
                                   do_skew = TRUE, Wishart = FALSE, 
                                   cores = 2L, nice = 19L){
  
  if(.Platform$OS.type!='unix')
    stop('parallel::mcparallel/mccollect do not exist on windows')
  
  if(is.null(hypers)) hypers <- Hypers(X,Y)
  if(is.null(opts)) opts <- Opts()
  
  parallel::mc.reset.stream()
  cores.detected <- parallel::detectCores()
  
  if(cores > cores.detected) cores <- cores.detected
  mc_num_save <- ceiling(opts$num_save/cores)
  opts$num_save <- mc_num_save
  
  for(i in 1:cores) {
    parallel::mcparallel({tools::psnice(value=nice);
      MultiskewBART(X = X, Y = Y, test_X = test_X, 
                     hypers = hypers, opts = opts, 
                     do_skew = do_skew, Wishart = Wishart)},
      silent=(i!=1))
  }
  
  post.list <- parallel::mccollect()
  
  post <- post.list[[1]]

  if(cores==1) return(post)
  else {
    post$num_save <- cores * mc_num_save
    
    for(i in 2:cores) {
      
      post$y_hat_train <- cbind(post$y_hat_train,
                               post.list[[i]]$y_hat_train)
      
      post$y_hat_test <- cbind(post$y_hat_test,
                               post.list[[i]]$y_hat_test)
      
      post$f_hat_train <- cbind(post$f_hat_train,
                               post.list[[i]]$f_hat_train)
      
      post$f_hat_test <- cbind(post$f_hat_test,
                               post.list[[i]]$f_hat_test)
       
      post$lambda <- rbind(post$lambda,
                           post.list[[i]]$lambda)
      
      post$Sigma <- cbind(post$Sigma,
                          post.list[[i]]$Sigma)
      
      #post$likelihood <- c(post$likelihood, post.list[[i]]$likelihood)
      post$likelihood_mat <- rbind(post$likelihood_mat, post.list[[i]]$likelihood_mat)
      
    }
    # rearrange using array 
    post$y_hat_train <- array(post$y_hat_train, c(nrow(Y), ncol(Y), post$num_save))
    post$y_hat_test <- array(post$y_hat_test, c(nrow(test_X), ncol(Y), post$num_save))
    post$f_hat_train <- array(post$f_hat_train, c(nrow(Y), ncol(Y), post$num_save))
    post$f_hat_test <- array(post$f_hat_test, c(nrow(test_X), ncol(Y), post$num_save))
    post$Sigma <- array(post$Sigma, c(2, 2, post$num_save))
    
    # calculate mean values
    post$y_hat_train_mean <- apply(post$y_hat_train, c(1,2), mean)
    post$y_hat_test_mean <- apply(post$y_hat_test, c(1,2), mean)
    post$f_hat_train_mean  <- apply(post$f_hat_train, c(1,2), mean)
    post$f_hat_test_mean  <- apply(post$f_hat_test, c(1,2), mean)
    
    return(post)
  }
}  




  