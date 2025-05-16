#' MCMC options for MultiskewBART 
#'
#' Creates a list which provides the parameters for running the Markov chain.
#'
#' @param num_burn Number of warmup iterations for the chain.
#' @param num_thin Thinning interval for the chain.
#' @param num_save The number of samples to collect; in total, num_burn + num_save * num_thin iterations are run
#' @param num_print Interval for how often to print the chain's progress
#' @param update_Sigma_mu If true, Sigma_mu is updated in the Markov chain
#' @param update_s If true, s is updated using the Dirichlet prior.
#' @param update_alpha If true, alpha is updated using a scaled beta prime prior
#'
#' @return Returns a list containing the function arguments
#' @export
Opts <- function(num_burn = 5000, num_thin = 1, num_save = 5000,
                 num_print = 100, update_Sigma_mu = FALSE,
                 update_s = FALSE, update_alpha = FALSE) {


  return(list(num_burn = num_burn, num_thin = num_thin, num_save = num_save,
              num_print = num_print, update_Sigma_mu = update_Sigma_mu,
              update_Sigma = FALSE, update_s = update_s,
              update_alpha = update_alpha))

}
