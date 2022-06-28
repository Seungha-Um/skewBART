#ifndef FOREST_H
#define FOREST_H

#include <RcppArmadillo.h>
#include "functions.h"

class Hypers;
class Forest;
struct Opts;
class MNode;
typedef std::vector<MNode*> MNodevec;


class Hypers {

  public:

  double alpha;
  double beta;
  double gamma;
  arma::mat Sigma;
  arma::mat Sigma_hat;
  arma::mat Sigma_mu;
  arma::mat Sigma_mu_hat;
  arma::mat A;
  arma::mat B;

  double temperature;
  int num_tree;
  arma::vec s;
  arma::vec logs;
  arma::sp_umat group;
  arma::uvec group_size;
  arma::sp_mat p_group_var;

  // Updates
  // void UpdateSigma(const arma::vec& residuals);
  // void UpdateSigmaMu(const arma::vec& means);
  /* void UpdateAlpha(); */

  // Construct/Destructure
  Hypers(Rcpp::List hypers);

  // Utilities
  int SampleVar() const;

};


class SuffStats {
 public:

  arma::vec sum_Y;
  arma::mat sum_YYt;
  double n;
  int J;

 SuffStats(int JJ) : J(JJ) {
    n = 0.0;
    sum_Y = arma::zeros<arma::vec>(J);
    sum_YYt = arma::zeros<arma::mat>(J);
  }

};

class MNode {

  public:

  bool is_leaf;
  bool is_root;
  bool is_left;
  MNode* left;
  MNode* right;
  MNode* parent;

  // Branch params
  int depth;
  int var;
  double val;
  arma::sp_vec lowers;
  arma::sp_vec uppers;
  arma::sp_uvec parent_split_indicators;

  // Sufficient Statistics
  SuffStats* ss;
  void ResetSuffStat() {
    ss->n = 0.0;
    ss->sum_Y = arma::zeros<arma::vec>(mu.size());
    ss->sum_YYt = arma::zeros<arma::mat>(mu.size(), mu.size());
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }


  // Leaf parameters
  arma::vec mu;

  // Constructor / Destructor
  MNode(const Hypers& hypers);
  MNode(MNode* parent, const Hypers& hypers, bool is_left);
  ~MNode();

  // Updates and such
  void BirthLeaves(const Hypers& hypers);
  void GenBelow(const Hypers& hypers);
  void get_limits_below();
  void DeleteLeaves();
  void UpdateMu(const arma::mat& Y, const arma::mat& X, const Hypers& hypers);

};


struct Opts {

  int num_burn;
  int num_thin;
  int num_save;
  int num_print;

  bool update_Sigma_mu;
  bool update_Sigma;
  bool update_s;
  bool update_alpha;

  Opts(Rcpp::List opts);

};

// Functions for computing with the forest
void GetSuffStats(MNode* n, const arma::mat& y,
                  const arma::mat& X, const Hypers& hypers);
void GetSuffStats(MNode* n, const arma::vec& y, const arma::mat& yyt,
                    const arma::vec& x, const Hypers& hypers);

double LogLT(MNode* n, const arma::mat& Y,
             const arma::mat& X, const Hypers& hypers);


// Split probability, used in growing nodes
double SplitProb(MNode* node, const Hypers& hypers);

// Other node-specific functions
void leaves(MNode* x, MNodevec& leafs);
std::vector<MNode*> leaves(MNode* x);



class Forest {

 private:

  MNodevec trees;
  Hypers hypers;
  Opts opts;

 public:

  Forest(Rcpp::List hypers, Rcpp::List opts);
  ~Forest();
  void IterateGibbs(arma::mat& Y_hat, const arma::mat& X, const arma::mat& Y);
  arma::cube do_gibbs(const arma::mat& X,
                     const arma::mat& Y,
                     const arma::mat& X_test,
                     int num_iter);

  // Getters and Setters, from R
  void set_s(const arma::vec& s);
  void set_sigma(const arma::mat& Sigma);
  arma::vec get_s();
  Rcpp::List get_params();

  // Prediction interfact
  arma::mat predict_mat(const arma::mat& X_test);

  // Var counts
  arma::uvec get_counts();

};



std::vector<MNode*> not_grand_branches(MNode* tree);
void not_grand_branches(std::vector<MNode*>& ngb, MNode* node);
void branches(MNode* n, MNodevec& branch_vec);
std::vector<MNode*> branches(MNode* root);
MNode* rand(MNodevec& ngb);


// Getting the counts
arma::uvec get_var_counts(std::vector<MNode*>& forest, const Hypers& hypers);
void get_var_counts(arma::uvec& counts, MNode* node, const Hypers& hypers);


// Functions for collecting quantities from the forest
/* arma::mat get_means(Nodevec& forest); */
/* void get_means(Node* node, std::vector<double>& means); */


// Predictions
arma::mat predict(const MNodevec& forest, const arma::mat& X,
                  const Hypers& hypers);
arma::mat predict(MNode* n, const arma::mat& X, const Hypers& hypers);
arma::vec predict(MNode* n, const arma::vec& x, const Hypers& hypers);



std::vector<MNode*> not_grand_branches(MNode* tree);
void not_grand_branches(std::vector<MNode*>& ngb, MNode* node);
void branches(MNode* n, MNodevec& branch_vec);
std::vector<MNode*> branches(MNode* root);
MNode* rand(MNodevec& ngb);


// Getting the counts
arma::uvec get_var_counts(std::vector<MNode*>& forest, const Hypers& hypers);
void get_var_counts(arma::uvec& counts, MNode* node, const Hypers& hypers);


// Functions for collecting quantities from the forest
/* arma::mat get_means(Nodevec& forest); */
/* void get_means(Node* node, std::vector<double>& means); */


// Predictions
arma::mat predict(const MNodevec& forest, const arma::mat& X,
                  const Hypers& hypers);
arma::mat predict(MNode* n, const arma::mat& X, const Hypers& hypers);
arma::vec predict(MNode* n, const arma::vec& x, const Hypers& hypers);



// Various tree probabilities
double growth_prior(int leaf_depth, const Hypers& hypers);



// Sampling and selecting nodes
MNode* birth_node(MNode* tree, double* leaf_node_probability);
double probability_node_birth(MNode* tree);
MNode* death_node(MNode* tree, double* p_not_grand);


// MCMC on the trees
void birth_death(MNode* tree, const arma::mat& X, const arma::mat& Y,
                 const Hypers& hypers);
void node_birth(MNode* tree, const arma::mat& X, const arma::mat& Y,
                const Hypers& hypers);
void node_death(MNode* tree, const arma::mat& X, const arma::mat& Y,
                const Hypers& hypers);
MNode* draw_prior(MNode* tree, const arma::mat& X, const arma::mat& Y, const Hypers& hypers);


// Functions for the perturb algorithm
double calc_cutpoint_likelihood(MNode* node);
std::vector<double> get_perturb_limits(MNode* branch);
void perturb_decision_rule(MNode* tree,
                           const arma::mat& X,
                           const arma::mat& Y,
                           const Hypers& hypers);

// The Bayesian backfitting algorithm
void TreeBackfit(MNodevec& forest, arma::mat& Y_hat,
                 const Hypers& hypers, const arma::mat& X, const arma::mat& Y,
                 const Opts& opts);




#endif
