#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
arma::vec update_b(const arma::vec& Z, const arma::vec& b_new, 
                   const arma::vec& mu_new, const arma::vec& b_old, 
                   const arma::vec& mu_old, const arma::uvec& cluster_idx, 
             double sigma) {
  
  int N_cluster = b_new.size();
  int N_total = Z.size();
  vec b = zeros<vec>(N_cluster);
  vec loglik_before = zeros<vec>(N_cluster);
  vec loglik_after = zeros<vec>(N_cluster);
  
  for(int i = 0; i < N_cluster; i++) {
    double z_old = R::qnorm(b_old(i), 0.0, 1.0, 1, 0);
    double z_new = R::qnorm(b_new(i), 0.0, 1.0, 1, 0);
    loglik_before(i) = R::dnorm4(z_old, 0.0, 1.0, 1);
    loglik_after(i) = R::dnorm4(z_new, 0.0, 1.0, 1);
  }
  
  for(int i = 0; i < N_total; i++) {
    int idx = cluster_idx(i) - 1;
    loglik_before(idx) = 
      loglik_before(idx) + R::dnorm4(Z(i), mu_old(i), sigma, 1); 
    loglik_after(idx) = 
      loglik_after(idx) + R::dnorm4(Z(i), mu_new(i), sigma, 1);
  }
  
  vec log_accept = loglik_after - loglik_before;
  for(int i = 0; i < N_cluster; i++) {
    b(i) = std::log(R::unif_rand()) < log_accept(i) ? b_new(i) : b_old(i);
  }
  
  return b;
  
}