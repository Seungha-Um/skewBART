update_sigma <- function(Y, Sigma, hypers)
{

  N <- nrow(Y)

  ## Extract current parameters
  sigma_1_old <- sqrt(Sigma[1,1])
  sigma_2_old <- sqrt(Sigma[2,2] - Sigma[2,1]^2 / Sigma[1,1])
  phi_old     <- Sigma[1,2] / Sigma[1,1]


  ## Update sigma_1
  a_old        <- 1 / sigma_1_old^2
  a_prop       <- rgamma(1, shape = N/2 + 1, rate = sum(Y[,1]^2) / 2)
  sigma_1_prop <- 1 / sqrt(a_prop)

  numerator <- dcauchy(sigma_1_prop, 0, hypers$sigma_1_hat) * a_prop^(-3/2)
  denom     <- dcauchy(sigma_1_old, 0, hypers$sigma_1_hat) * a_old^(-3/2)
  sigma_1   <- ifelse(runif(1) < numerator/denom, sigma_1_prop, sigma_1_old)

  ## Update sigma_2
  delta        <- Y[,2] - phi_old * Y[,1]
  b_old        <- 1 / sigma_2_old^2
  b_prop       <- rgamma(1, shape = N/2 + 1, rate = sum(delta^2)/2)
  sigma_2_prop <- 1 / sqrt(b_prop)

  num_2 <- dcauchy(sigma_2_prop, 0, hypers$sigma_2_hat) * b_prop^(-3/2)
  den_2 <- dcauchy(sigma_2_old, 0, hypers$sigma_2_hat) * b_old^(-3/2)
  sigma_2 <- ifelse(runif(1) < num_2 / den_2, sigma_2_prop, sigma_2_old)

  ## Update phi
  prec <- sum(Y[,1]^2)
  mu   <- sum(Y[,1] * Y[,2]) / prec
  phi_prop <- rnorm(1, mu, sigma_2 / sqrt(prec))

  num_phi <- (phi_prop^2 + hypers$r_hat^2)^(-3/2)
  den_phi <- (phi_old^2 + hypers$r_hat^2)^(-3/2)
  phi     <- ifelse(runif(1) < num_phi / den_phi, phi_prop, phi_old)

  ## Update Sigma
  Sigma_out <- c(sigma_1^2, phi * sigma_1^2, phi * sigma_1^2,
                 phi^2 * sigma_1^2 + sigma_2^2)
  Sigma_out <- matrix(Sigma_out, nrow = 2)

  return(Sigma_out)

}
