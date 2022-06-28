#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <RcppArmadillo.h>

int sample_class(const arma::vec& probs);
int sample_class(int n);
int sample_class_row(const arma::sp_mat& probs, int row);
double logit(double x);
double expit(double x);
double activation(double x, double c, double tau);
arma::vec rmvnorm(const arma::vec& mean, const arma::mat& Precision);
double rlgam(double shape);
double log_sum_exp(const arma::vec& x);
double cauchy_jacobian(double tau, double sigma_hat);
double logpdf_beta(double x, double a, double b);

// Samplers for common priors
double update_sigma_halfcauchy(const arma::vec& r, double sigma_hat, double sigma_old,
                    double temperature);
bool do_mh(double loglik_new, double loglik_old,
           double new_to_old, double old_to_new);


#endif
