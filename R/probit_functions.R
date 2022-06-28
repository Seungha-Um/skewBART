update_z <- function(mu, y) {
  a_vec <- ifelse(y == 0, -Inf, 0)
  b_vec <- ifelse(y == 0, 0, Inf)
  return(rtruncnorm(length(y), a = a_vec, b = b_vec, mean = mu, sd = 1))
}
