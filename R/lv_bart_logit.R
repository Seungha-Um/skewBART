lv_bart_logit <- function(Y, X, cluster_idx, hypers, opts, sigma_mh = 1,
                          Y_test = NULL, X_test = NULL, idx_test = NULL) {

  N <- length(Y)
  N_clust <- length(unique(cluster_idx))
  stopifnot(max(unique(cluster_idx)) == N_clust & min(unique(cluster_idx)) == 1)
  num_burn <- opts$num_burn
  num_thin <- opts$num_thin
  num_save <- opts$num_save

  ## Impute Z
  a_vec <- ifelse(Y == 0, -Inf, 0)
  b_vec <- ifelse(Y == 0, 0, Inf)
  Z <- rtruncnorm(length(Y), a = a_vec, b = b_vec)

  ## Make the forest
  sbart_forest <- MakeForest(hypers, opts)

  ## Make the mcmc state
  state <- list(Y = Y, X = X, Z = Z, idx = cluster_idx, mu_hat = rep(0, N),
                b = runif(N_clust))

  ## Burn in
  for(i in 1:num_burn) {
    state <- iterate_gibbs(state, sbart_forest, sigma_mh)
    if(i %% 100 == 0) cat(paste0("\rFinish warmup ", i, "\t\t\t"))
  }

  ## Make stuff for saving
  b_mat <- matrix(NA, nrow = num_save, ncol = N_clust)
  mu_hat <- matrix(NA, nrow = num_save, ncol = N)
  heldout_loglik <- 0
  if(!is.null(Y_test)) {
    heldout_loglik <- matrix(NA, nrow = num_save, ncol = length(Y_test))
  }


  ## Run Save iterations
  for(i in 1:num_save) {
    for(j in 1:num_thin) {
      state <- iterate_gibbs(state, sbart_forest, sigma_mh)
    }

    b_mat[i,] <- state$b
    mu_hat[i,] <- state$mu_hat
    if(i %% 100 == 0) cat(paste0("\rFinish save ", i, "\t\t\t"))

    ## Heldout log likelihood
    if(!is.null(Y_test)) {
      num_test_clust <- max(idx_test)
      b_test <- runif(num_test_clust)
      X_b_test <- cbind(X_test, b_test[idx_test])
      mu_test <- sbart_forest$predict(X_b_test)
      prob_test <- pnorm(mu_test)
      heldout_loglik[i,] <- dbinom(x = Y_test, size = 1, prob = prob_test, log = TRUE)
    }

  }

  return(list(b = b_mat, mu_hat = mu_hat, state = state,
              forest = sbart_forest, heldout_loglik = heldout_loglik))

}

iterate_gibbs <- function(state, sbart_forest, sigma_mh = 1) {
  X_b          <- cbind(state$X, state$b[state$idx])
  state$mu_hat <- sbart_forest$do_gibbs(X_b, state$Z, X_b, 1)
  params       <- sbart_forest$get_params()
  b_prop       <- propose_b(state$b, sigma_mh)
  X_b_prop     <- cbind(state$X, b_prop[state$idx])
  mu_prop      <- sbart_forest$predict(X_b_prop)
  b_new        <- update_b(Z = state$Z, b_new = b_prop, mu_new = mu_prop,
                           b_old = state$b, mu_old = state$mu_hat,
                           cluster_idx = state$idx, sigma = params$sigma)
  state$b      <- b_new
  X_b          <- cbind(state$X, state$b[state$idx])
  state$mu_hat <- sbart_forest$predict(X_b)
  state$Z      <- update_z(mu = state$mu_hat, y = state$Y)
  return(state)
}

propose_b <- function(b, sigma) {
  pnorm(qnorm(b) + sigma * rnorm(length(b)))
}
