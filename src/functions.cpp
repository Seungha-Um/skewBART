#include "functions.h"

using namespace arma;
using namespace Rcpp;

int sample_class(const arma::vec& probs) {
  double U = R::unif_rand();
  double foo = 0.0;
  int K = probs.size();

  // Sample
  for(int k = 0; k < K; k++) {
    foo += probs(k);
    if(U < foo) {
      return(k);
    }
  }
  return K - 1;
}


int sample_class(int n) {
  double U = R::unif_rand();
  double p = 1.0 / ((double)n);
  double foo = 0.0;

  for(int k = 0; k < n; k++) {
    foo += p;
    if(U < foo) {
      return k;
    }
  }
  return n - 1;
}

int sample_class_row(const arma::sp_mat& probs, int row) {

  double U = R::unif_rand();
  double foo = 0.0;

  sp_mat::const_row_iterator it = probs.begin_row(row);
  sp_mat::const_row_iterator it_end = probs.end_row(row);

  for(; it != it_end; ++it) {
    foo += (*it);
    if(U < foo) {
      return it.col();
    }
  }
  return it.col();
}

double logit(double x) {
  return log(x) - log(1.0-x);
}

double expit(double x) {
  return 1.0 / (1.0 + exp(-x));
}

double activation(double x, double c, double tau) {
  return 1.0 - expit((x - c) / tau);
}

// [[Rcpp::export]]
arma::vec rmvnorm(const arma::vec& mean, const arma::mat& Precision) {
  arma::vec z = arma::zeros<arma::vec>(mean.size());
  for(int i = 0; i < mean.size(); i++) {
    z(i) = norm_rand();
  }
  arma::mat Sigma = inv_sympd(Precision);
  arma::mat L = chol(Sigma, "lower");
  arma::vec h = mean + L * z;
  return h;
}


// [[Rcpp::export]]
double rlgam(double shape) {
  if(shape >= 0.1) return log(Rf_rgamma(shape, 1.0));

  double a = shape;
  double L = 1.0/a- 1.0;
  double w = exp(-1.0) * a / (1.0 - a);
  double ww = 1.0 / (1.0 + w);
  double z = 0.0;
  do {
    double U = unif_rand();
    if(U <= ww) {
      z = -log(U / ww);
    }
    else {
      z = log(unif_rand()) / L;
    }
    double eta = z >= 0 ? -z : log(w)  + log(L) + L * z;
    double h = -z - exp(-z / a);
    if(h - eta > log(unif_rand())) break;
  } while(true);

  // Rcout << "Sample: " << -z/a << "\n";

  return -z/a;
}

double log_sum_exp(const arma::vec& x) {
  double M = x.max();
  return M + log(sum(exp(x - M)));
}

double cauchy_jacobian(double tau, double sigma_hat) {
  double sigma = pow(tau, -0.5);
  int give_log = 1;

  double out = Rf_dcauchy(sigma, 0.0, sigma_hat, give_log);
  out = out - M_LN2 - 3.0 / 2.0 * log(tau);

  return out;

}

double logpdf_beta(double x, double a, double b) {
  return (a-1.0) * log(x) + (b-1.0) * log(1 - x) - Rf_lbeta(a,b);
}

double update_sigma_halfcauchy(const arma::vec& r, double sigma_hat, double sigma_old,
                    double temperature) {

  double SSE = dot(r,r) * temperature;
  double n = r.size() * temperature;

  double shape = 0.5 * n + 1.0;
  double scale = 2.0 / SSE;
  double sigma_prop = pow(Rf_rgamma(shape, scale), -0.5);

  double tau_prop = pow(sigma_prop, -2.0);
  double tau_old = pow(sigma_old, -2.0);

  double loglik_rat = cauchy_jacobian(tau_prop, sigma_hat) -
    cauchy_jacobian(tau_old, sigma_hat);

  return log(unif_rand()) < loglik_rat ? sigma_prop : sigma_old;

}

bool do_mh(double loglik_new, double loglik_old,
           double new_to_old, double old_to_new) {

  double cutoff = loglik_new + new_to_old - loglik_old - old_to_new;

  return log(unif_rand()) < cutoff ? true : false;

}

