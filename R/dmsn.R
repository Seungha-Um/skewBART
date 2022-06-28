dmsn <- function(y, mu, Sigma, lambda, give_log = FALSE) {
  stopifnot(nrow(Sigma) == 2)
  stopifnot(ncol(Sigma) == 2)

  Lambda <- diag(lambda)
  Omega <- Sigma + Lambda %*% t(Lambda)
  Delta <- diag(2) + t(Lambda) %*% solve(Sigma) %*% Lambda
  Delta_inv <- solve(Delta)

  r <- y - mu
  upper <- t(t(Lambda) %*% solve(Omega) %*% t(r))

  ff <- function(i) {
    return(log(mvtnorm::pmvnorm(upper = upper[i,], sigma = Delta_inv)))
  }
  pmvs <- sapply(1:nrow(r), ff)
  out <- log(4) + mvtnorm::dmvnorm(r, sigma = Omega, log = TRUE) + pmvs
  if(!give_log) {
    out <- exp(out)
  }

  return(out)
}
