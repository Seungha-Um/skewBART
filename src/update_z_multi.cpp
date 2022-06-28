#include "functions.h"

using namespace arma;
using namespace Rcpp;


// [[Rcpp::export]]
List update_z_multi_2(const arma::mat& Y,
                      const arma::mat& mu,
                      const arma::mat& Sigma,
                      const arma::mat& Z_0)
{

  mat delta = Y - mu;

  // Update lambda
  mat A                = inv(Sigma);
  vec mu_lambda        = zeros<vec>(2);
  mat Sigma_inv_lambda = zeros<mat>(2,2);
  mat Z_tmp            = zeros<mat>(2,2);

  for(int i = 0; i < delta.n_rows; i++) {
    Z_tmp(0,0)         = Z_0(i,0); Z_tmp(1,1) = Z_0(i,1);
    Sigma_inv_lambda  += Z_tmp * A * Z_tmp;
    mu_lambda         += Z_tmp * A * trans(delta.row(i));
  }
  mu_lambda  = inv(Sigma_inv_lambda) * mu_lambda;
  vec lambda = rmvnorm(mu_lambda, Sigma_inv_lambda);

  // Update Z
  mat Lambda  = zeros<mat>(2,2); Lambda(0,0) = lambda(0); Lambda(1,1) = lambda(1);
  mat prec_Z  = Lambda * A * Lambda + eye<mat>(2,2);
  mat Sigma_Z = inv(prec_Z);
  // mat mean_Z  = zeros<mat>(delta.n_rows, 2);
  mat mean_Z = delta * A * Lambda * Sigma_Z;

  List out;
  out["lambda"]  = lambda;
  out["mean_Z"]  = mean_Z;
  out["Sigma_Z"] = Sigma_Z;

  return out;

}



