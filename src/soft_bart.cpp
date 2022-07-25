#include "soft_bart.h"

using namespace Rcpp;
using namespace arma;

void UHypers::UpdateSigma(const arma::vec& residuals) {
  sigma = update_sigma_halfcauchy(residuals, sigma_hat, sigma, temperature);
}

void UHypers::UpdateSigmaMu(const arma::vec& means) {
  sigma_mu = update_sigma_halfcauchy(means, sigma_mu_hat, sigma_mu, temperature);
}

UForest::UForest(Rcpp::List hypers, Rcpp::List opts) : hypers(hypers), opts(opts) {
  trees.resize(this->hypers.num_tree);
  for(int i = 0; i < this->hypers.num_tree; i++) {
    trees[i] = new Node(this->hypers);
  }
}

UForest::~UForest() {
  for(int i = 0; i < trees.size(); i++) delete trees[i];
}

UOpts::UOpts(Rcpp::List opts) {
  num_burn        = opts["num_burn"];
  num_thin        = opts["num_thin"];
  num_save        = opts["num_save"];
  num_print       = opts["num_print"];
  update_sigma_mu = opts["update_sigma_mu"];
  update_sigma    = opts["update_sigma"];
  update_s        = opts["update_s"];
  update_alpha    = opts["update_alpha"];
  update_tau      = opts["update_tau"];
}

UHypers::UHypers(Rcpp::List hypers)
{
  alpha        = hypers["alpha"];
  beta         = hypers["beta"];
  gamma        = hypers["gamma"];
  sigma_hat    = hypers["sigma_hat"];
  sigma        = sigma_hat;
  sigma_mu_hat = hypers["sigma_mu_hat"];
  sigma_mu     = sigma_mu_hat;
  shape        = hypers["shape"];
  tau_rate     = hypers["tau_rate"];
  temperature  = hypers["temperature"];
  num_tree    = hypers["num_tree"];

  group = as<arma::uvec>(hypers["group"]);
  num_groups = group.max() + 1;

  // Initialize s
  s = 1.0 / num_groups * ones<vec>(num_groups);
  logs = log(s);

  // Initialize group indicators
  group_to_vars.resize(s.size());
  for(int i = 0; i < s.size(); i++) {
    group_to_vars[i].resize(0);
  }
  int P = group.size();
  for(int p = 0; p < P; p++) {
    int idx = group(p);
    group_to_vars[idx].push_back(p);
  }

}

Node::Node(const UHypers& hypers) {

  is_leaf = true;
  is_root = true;
  is_left = true;
  left = this;
  right = this;
  parent = this;

  depth = 0;
  var = 0;
  val = 0.0;
  lowers = zeros<sp_mat>(hypers.group.size(), 1);
  uppers = zeros<sp_mat>(hypers.group.size(), 1);
  parent_split_indicators = zeros<sp_umat>(hypers.group.size(), 1);
  tau = R::rgamma(1.0, 1.0/hypers.tau_rate);
  mu = 0.0;
  current_weight = 1.0;

}

Node::Node(Node* parent, const UHypers& hypers, bool is_left) {
  parent->is_leaf = false;

  is_leaf = true;
  is_root = false;
  left = this;
  right = this;
  this->parent = parent;
  this->is_left = is_left;

  depth = parent->depth + 1;
  var = 0;
  val = 0.0;
  lowers = parent->lowers;
  uppers = parent->uppers;
  parent_split_indicators = parent->parent_split_indicators;
  current_weight = 0.0;
  mu = 0.0;
  tau = parent->tau;

  // Update the bounds
  int pvar = parent->var;
  if(is_left) {
    uppers(pvar)= parent->val;
  }
  else {
    lowers(pvar) = parent->val;
  }


}

Node::~Node() {
  if(!is_leaf) {
    delete left;
    delete right;
  }
}

int UHypers::SampleVar() const {

  int group_idx = sample_class(s);
  int var_idx = sample_class(group_to_vars[group_idx].size());

  return group_to_vars[group_idx][var_idx];
}

void Node::BirthLeaves(const UHypers& hypers) {
  if(is_leaf) {
    // Rcout << "Get Vars";
    var = hypers.SampleVar();
    // Rcout << "OK Vars";
    if(parent_split_indicators(var) == 0) {
      parent_split_indicators(var) = 1;
      uppers(var) = 1.0;
      lowers(var) = 0.0;
    }
    // Rcout << "Sampling val";
    val = (uppers(var) - lowers(var)) * unif_rand() + lowers(var);
    // Rcout << "Make leftright";
    left = new Node(this, hypers, true);
    right = new Node(this, hypers, false);
  }
}

void Node::GenBelow(const UHypers& hypers) {
  double grow_prob = SplitProb(this, hypers);
  double u = unif_rand();
  if(u < grow_prob) {
    // Rcout << "BL";
    BirthLeaves(hypers);
    // Rcout << "Grow left";
    left->GenBelow(hypers);
    right->GenBelow(hypers);
  }
}

void Node::GetW(const arma::mat& X, int i) {

  if(!is_leaf) {

    double weight = activation(X(i,var), val, tau);
    left->current_weight = weight * current_weight;
    right->current_weight = (1 - weight) * current_weight;

    left->GetW(X,i);
    right->GetW(X,i);

  }
}

void Node::DeleteLeaves() {
  delete left;
  delete right;
  left = this;
  right = this;
  is_leaf = true;
  if(is_root || parent->parent_split_indicators(var) == 0) {
    parent_split_indicators(var) = 0;
    uppers(var) = 0.0;
    lowers(var) = 0.0;
  }
  var = 0;
  val = 0.0;
}

void Node::UpdateMu(const arma::vec& Y, const arma::mat& X, const UHypers& hypers) {

  std::vector<Node*> leafs = leaves(this);
  int num_leaves = leafs.size();

  // Get mean and covariance
  vec mu_hat = zeros<vec>(num_leaves);
  mat Omega_inv = zeros<mat>(num_leaves, num_leaves);
  GetSuffStats(this, Y, X, hypers, mu_hat, Omega_inv);

  vec mu_samp = rmvnorm(mu_hat, Omega_inv);
  for(int i = 0; i < num_leaves; i++) {
    leafs[i]->mu = mu_samp(i);
  }
}

void Node::UpdateTau(const arma::vec& Y,
                     const arma::mat& X,
                     const UHypers& hypers) {

  double tau_old = tau;
  double tau_new = tau_proposal(tau);

  double loglik_new = loglik_tau(tau_new, X, Y, hypers) + logprior_tau(tau_new, hypers.tau_rate);
  double loglik_old = loglik_tau(tau_old, X, Y, hypers) + logprior_tau(tau_old, hypers.tau_rate);
  double new_to_old = log_tau_trans(tau_old);
  double old_to_new = log_tau_trans(tau_new);

  bool accept_mh = do_mh(loglik_new, loglik_old, new_to_old, old_to_new);

  if(accept_mh) {
    SetTau(tau_new);
  }
  else {
    SetTau(tau_old);
  }

}

// Local tau stuff
void Node::SetTau(double tau_new) {
  tau = tau_new;
  if(!is_leaf) {
    left->SetTau(tau_new);
    right->SetTau(tau_new);
  }
}

double Node::loglik_tau(double tau_new, const arma::mat& X,
                        const arma::vec& Y, const UHypers& hypers) {

  double tau_old = tau;
  SetTau(tau_new);
  double out = LogLT(this, Y, X, hypers);
  SetTau(tau_old);
  return out;

}


double SplitProb(Node* node, const UHypers& hypers) {
  return hypers.gamma * pow(1.0 + node->depth, -hypers.beta);
}

double growth_prior(int leaf_depth, const UHypers& hypers) {
  return hypers.gamma * pow(1.0 + leaf_depth, -hypers.beta);
}


void leaves(Node* x, std::vector<Node*>& leafs) {
  if(x->is_leaf) {
    leafs.push_back(x);
  }
  else {
    leaves(x->left, leafs);
    leaves(x->right, leafs);
  }
}

std::vector<Node*> leaves(Node* x) {
  std::vector<Node*> leafs(0);
  leaves(x, leafs);
  return leafs;
}

std::vector<Node*> not_grand_branches(Node* tree) {
  std::vector<Node*> ngb(0);
  not_grand_branches(ngb, tree);
  return ngb;
}

void not_grand_branches(std::vector<Node*>& ngb, Node* node) {
  if(!node->is_leaf) {
    bool left_is_leaf = node->left->is_leaf;
    bool right_is_leaf = node->right->is_leaf;
    if(left_is_leaf && right_is_leaf) {
      ngb.push_back(node);
    }
    else {
      not_grand_branches(ngb, node->left);
      not_grand_branches(ngb, node->right);
    }
  }
}
void branches(Node* n, Nodevec& branch_vec) {
  if(!(n->is_leaf)) {
    branch_vec.push_back(n);
    branches(n->left, branch_vec);
    branches(n->right, branch_vec);
  }
}

std::vector<Node*> branches(Node* root) {
  std::vector<Node*> branch_vec;
  branch_vec.resize(0);
  branches(root, branch_vec);
  return branch_vec;
}

Node* rand(std::vector<Node*> ngb) {

  int N = ngb.size();
  arma::vec p = ones<vec>(N) / ((double)(N));
  int i = sample_class(p);
  return ngb[i];
}

Node* birth_node(Node* tree, double* leaf_node_probability) {
  std::vector<Node*> leafs = leaves(tree);
  Node* leaf = rand(leafs);
  *leaf_node_probability = 1.0 / ((double)leafs.size());

  return leaf;
}

double probability_node_birth(Node* tree) {
  return tree->is_leaf ? 1.0 : 0.5;
}

Node* death_node(Node* tree, double* p_not_grand) {
  std::vector<Node*> ngb = not_grand_branches(tree);
  Node* branch = rand(ngb);
  *p_not_grand = 1.0 / ((double)ngb.size());

  return branch;
}

arma::vec get_means(std::vector<Node*>& forest) {
  std::vector<double> means(0);
  int num_tree = forest.size();
  for(int t = 0; t < num_tree; t++) {
    get_means(forest[t], means);
  }

  // Convert std::vector to armadillo vector, deep copy
  vec out(&(means[0]), means.size());
  return out;
}

void get_means(Node* node, std::vector<double>& means) {

  if(node->is_leaf) {
    means.push_back(node->mu);
  }
  else {
    get_means(node->left, means);
    get_means(node->right, means);
  }
}

arma::vec predict(const std::vector<Node*>& forest,
                  const arma::mat& X,
                  const UHypers& hypers) {

  vec out = zeros<vec>(X.n_rows);
  int num_tree = forest.size();

  for(int t = 0; t < num_tree; t++) {
    out = out + predict(forest[t], X, hypers);
  }

  return out;
}

arma::vec predict(Node* n, const arma::mat& X, const UHypers& hypers) {

  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();
  int N = X.n_rows;
  vec out = zeros<vec>(N);

  for(int i = 0; i < N; i++) {
    n->GetW(X,i);
    for(int j = 0; j < num_leaves; j++) {
      out(i) = out(i) + leafs[j]->current_weight * leafs[j]->mu;
    }
  }

  return out;

}

arma::uvec get_var_counts(std::vector<Node*>& forest, const UHypers& hypers) {
  arma::uvec counts = zeros<uvec>(hypers.s.size());
  int num_tree = forest.size();
  for(int t = 0; t < num_tree; t++) {
    get_var_counts(counts, forest[t], hypers);
  }
  return counts;
}

void get_var_counts(arma::uvec& counts, Node* node, const UHypers& hypers) {
  if(!node->is_leaf) {
    int group_idx = hypers.group(node->var);
    counts(group_idx) = counts(group_idx) + 1;
    get_var_counts(counts, node->left, hypers);
    get_var_counts(counts, node->right, hypers);
  }
}

/*Computes the sufficient statistics Omega_inv and mu_hat described in the
  paper; mu_hat is the posterior mean of the leaf nodes, Omega_inv is that
  posterior covariance*/
void GetSuffStats(Node* n, const arma::vec& y,
                  const arma::mat& X, const UHypers& hypers,
                  arma::vec& mu_hat_out, arma::mat& Omega_inv_out) {


  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();
  vec w_i = zeros<vec>(num_leaves);
  vec mu_hat = zeros<vec>(num_leaves);
  mat Lambda = zeros<mat>(num_leaves, num_leaves);

  for(int i = 0; i < X.n_rows; i++) {
    n->GetW(X, i);
    for(int j = 0; j < num_leaves; j++) {
      w_i(j) = leafs[j]->current_weight;
    }
    mu_hat = mu_hat + y(i) * w_i;
    Lambda = Lambda + w_i * trans(w_i);
  }

  Lambda = Lambda / pow(hypers.sigma, 2) * hypers.temperature;
  mu_hat = mu_hat / pow(hypers.sigma, 2) * hypers.temperature;
  Omega_inv_out = Lambda + eye(num_leaves, num_leaves) / pow(hypers.sigma_mu, 2);
  mu_hat_out = solve(Omega_inv_out, mu_hat);

}

double LogLT(Node* n, const arma::vec& Y,
             const arma::mat& X, const UHypers& hypers) {

  // Rcout << "Leaves ";
  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();

  // Get sufficient statistics
  arma::vec mu_hat = zeros<vec>(num_leaves);
  arma::mat Omega_inv = zeros<mat>(num_leaves, num_leaves);
  GetSuffStats(n, Y, X, hypers, mu_hat, Omega_inv);

  int N = Y.size();

  // Rcout << "Compute ";
  double out = -0.5 * N * log(M_2_PI * pow(hypers.sigma,2)) * hypers.temperature;
  out -= 0.5 * num_leaves * log(M_2_PI * pow(hypers.sigma_mu,2));
  double val, sign;
  log_det(val, sign, Omega_inv / M_2_PI);
  out -= 0.5 * val;
  out -= 0.5 * dot(Y, Y) / pow(hypers.sigma, 2) * hypers.temperature;
  out += 0.5 * dot(mu_hat, Omega_inv * mu_hat);

  // Rcout << "Done";
  return out;

}

void birth_death(Node* tree, const arma::mat& X, const arma::vec& Y,
                 const UHypers& hypers) {


  double p_birth = probability_node_birth(tree);

  if(unif_rand() < p_birth) {
    node_birth(tree, X, Y, hypers);
  }
  else {
    node_death(tree, X, Y, hypers);
  }
}

void node_birth(Node* tree, const arma::mat& X, const arma::vec& Y,
                const UHypers& hypers) {

  // Rcout << "Sample leaf";
  double leaf_probability = 0.0;
  Node* leaf = birth_node(tree, &leaf_probability);

  // Rcout << "Compute prior";
  int leaf_depth = leaf->depth;
  double leaf_prior = growth_prior(leaf_depth, hypers);

  // Get likelihood of current state
  // Rcout << "Current likelihood";
  double ll_before = LogLT(tree, Y, X, hypers);
  ll_before += log(1.0 - leaf_prior);

  // Get transition probability
  // Rcout << "Transistion";
  double p_forward = log(probability_node_birth(tree) * leaf_probability);

  // Birth new leaves
  // Rcout << "Birth";
  leaf->BirthLeaves(hypers);

  // Get likelihood after
  // Rcout << "New Likelihood";
  double ll_after = LogLT(tree, Y, X, hypers);
  ll_after += log(leaf_prior) +
    log(1.0 - growth_prior(leaf_depth + 1, hypers)) +
    log(1.0 - growth_prior(leaf_depth + 1, hypers));

  // Get Probability of reverse transition
  // Rcout << "Reverse";
  std::vector<Node*> ngb = not_grand_branches(tree);
  double p_not_grand = 1.0 / ((double)(ngb.size()));
  double p_backward = log((1.0 - probability_node_birth(tree)) * p_not_grand);

  // Do MH
  double log_trans_prob = ll_after + p_backward - ll_before - p_forward;
  if(log(unif_rand()) > log_trans_prob) {
    leaf->DeleteLeaves();
  }
  else {
    // Rcout << "Accept!";
  }
}

void node_death(Node* tree, const arma::mat& X, const arma::vec& Y,
                const UHypers& hypers) {

  // Select branch to kill Children
  double p_not_grand = 0.0;
  Node* branch = death_node(tree, &p_not_grand);

  // Compute before likelihood
  int leaf_depth = branch->left->depth;
  double leaf_prob = growth_prior(leaf_depth - 1, hypers);
  double left_prior = growth_prior(leaf_depth, hypers);
  double right_prior = growth_prior(leaf_depth, hypers);
  double ll_before = LogLT(tree, Y, X, hypers) +
    log(1.0 - left_prior) + log(1.0 - right_prior) + log(leaf_prob);

  // Compute forward transition prob
  double p_forward = log(p_not_grand * (1.0 - probability_node_birth(tree)));

  // Save old leafs, do not delete (they are dangling, need to be handled by the end)
  Node* left = branch->left;
  Node* right = branch->right;
  branch->left = branch;
  branch->right = branch;
  branch->is_leaf = true;

  // Compute likelihood after
  double ll_after = LogLT(tree, Y, X, hypers) + log(1.0 - leaf_prob);

  // Compute backwards transition
  std::vector<Node*> leafs = leaves(tree);
  double p_backwards = log(1.0 / ((double)(leafs.size())) * probability_node_birth(tree));

  // Do MH and fix dangles
  branch->left = left;
  branch->right = right;
  double log_trans_prob = ll_after + p_backwards - ll_before - p_forward;
  if(log(unif_rand()) > log_trans_prob) {
    branch->is_leaf = false;
  }
  else {
    branch->DeleteLeaves();
  }
}

Node* draw_prior(Node* tree, const arma::mat& X, const arma::vec& Y, const UHypers& hypers) {

  // Compute loglik before
  Node* tree_0 = tree;
  // Rcout << "A";
  double loglik_before = LogLT(tree_0, Y, X, hypers);

  // Make new tree and compute loglik after
  Node* tree_1 = new Node(hypers);
  // Rcout << "C";
  tree_1->GenBelow(hypers);
  // Rcout << "B";
  double loglik_after = LogLT(tree_1, Y, X, hypers);
  // Rcout << "Calc loglik" << "\n";

  // Do MH
  if(log(unif_rand()) < loglik_after - loglik_before) {
    delete tree_0;
    tree = tree_1;
  }
  else {
    delete tree_1;
  }
  return tree;
}

double calc_cutpoint_likelihood(Node* node) {
  if(node->is_leaf) return 1;

  double out = 1.0 / (node->uppers(node->var) - node->lowers(node->var));
  out = out * calc_cutpoint_likelihood(node->left);
  out = out * calc_cutpoint_likelihood(node->right);

  return out;
}

/*
  get_perturb_limits:

  Input: a branch node pointer
  Output: a 2-d vector out, such that (out[0], out[1]) represents the interval in
          which branch->val can vary without contradicting any of the branch
          nodes further down the tree. Note: THIS DOES NOT MODIFY THE TREE,
          IT ONLY COMPUTES THE LIMITS

  Algorithm: First, manually traverse backwards up the tree, checking to see
             if we encounter any nodes splitting on the same var. If we do, we
             update min and max. Next, we collect the branches below the
             current node. If any left ancestor splits on the current node,
             then we modify the min; otherwise, we modify the max.

*/

std::vector<double> get_perturb_limits(Node* branch) {
  double min = 0.0;
  double max = 1.0;

  Node* n = branch;
  while(!(n->is_root)) {
    if(n->is_left) {
      n = n->parent;
      if(n->var == branch->var) {
        if(n->val > min) {
          min = n->val;
        }
      }
    }
    else {
      n = n->parent;
      if(n->var == branch->var) {
        if(n->val < max) {
          max = n->val;
        }
      }
    }
  }
  std::vector<Node*> left_branches = branches(n->left);
  std::vector<Node*> right_branches = branches(n->right);
  for(int i = 0; i < left_branches.size(); i++) {
    if(left_branches[i]->var == branch->var) {
      if(left_branches[i]->val > min)
        min = left_branches[i]->val;
    }
  }
  for(int i = 0; i < right_branches.size(); i++) {
    if(right_branches[i]->var == branch->var) {
      if(right_branches[i]->val < max) {
        max = right_branches[i]->val;
      }
    }
  }

  std::vector<double> out; out.push_back(min); out.push_back(max);
  return out;
}

void Node::get_limits_below() {

  if(is_root) {
    lowers = zeros<sp_mat>(lowers.size(), 1);
    uppers = zeros<sp_mat>(uppers.size(), 1);
    parent_split_indicators = zeros<sp_umat>(lowers.size(), 1);
  }
  lowers = parent->lowers;
  uppers = parent->uppers;
  parent_split_indicators = parent->parent_split_indicators;
  if(!is_root) {
    if(is_left) {
      uppers(parent->var) = parent->val;
    }
    else {
      lowers(parent->var) = parent->val;
    }
  }
  if(!is_leaf) {
    if(parent_split_indicators(var) == 0) {
      parent_split_indicators(var) = 1;
      uppers(var) = 1.0;
      lowers(var) = 0.0;
    }
    left->get_limits_below();
    right->get_limits_below();
  }
}

void perturb_decision_rule(Node* tree,
                           const arma::mat& X,
                           const arma::vec& Y,
                           const UHypers& hypers) {

  // Randomly choose a branch; if no branches, we automatically reject
  std::vector<Node*> bbranches = branches(tree);
  if(bbranches.size() == 0)
    return;

  // Select the branch
  Node* branch = rand(bbranches);

  // Calculuate tree likelihood before proposal
  double ll_before = LogLT(tree, Y, X, hypers);

  // Calculate product of all 1/(B - A) here
  double cutpoint_likelihood = calc_cutpoint_likelihood(tree);

  // Calculate backward transition density
  std::vector<double> lims = get_perturb_limits(branch);
  double backward_trans = 1.0/(lims[1] - lims[0]);

  // save old split
  int old_feature = branch->var;
  double old_value = branch->val;

  // Modify the branch
  branch->var = hypers.SampleVar();
  lims = get_perturb_limits(branch);
  branch->val = lims[0] + (lims[1] - lims[0]) * unif_rand();
  tree->get_limits_below();

  // Calculate likelihood after proposal
  double ll_after = LogLT(tree, Y, X, hypers);

  // Calculate product of all 1/(B-A)
  double cutpoint_likelihood_after = calc_cutpoint_likelihood(tree);

  // Calculate forward transition density
  double forward_trans = 1.0/(lims[1] - lims[0]);

  // Do MH
  double log_trans_prob =
    ll_after + log(cutpoint_likelihood_after) + log(backward_trans)
    - ll_before - log(cutpoint_likelihood) - log(forward_trans);

  if(log(unif_rand()) > log_trans_prob) {
    branch->var = old_feature;
    branch->val = old_value;
    tree->get_limits_below();
  }
}

double logprior_tau(double tau, double tau_rate) {
  int DO_LOG = 1;
  return Rf_dexp(tau, 1.0 / tau_rate, DO_LOG);
}

double tau_proposal(double tau) {
  double U = 2.0 * unif_rand() - 1;
  return pow(5.0, U) * tau;
  // double w = 0.2 * unif_rand() - 0.1;
  // return tau + w;
}

double log_tau_trans(double tau_new) {
  return -log(tau_new);
  // return 0.0;
}

arma::vec get_tau_vec(const std::vector<Node*>& forest) {
  int t = forest.size();
  vec out = zeros<vec>(t);
  for(int i = 0; i < t; i++) {
    out(i) = forest[i]->tau;
  }
  return out;
}

void TreeBackfit(std::vector<Node*>& forest, arma::vec& Y_hat,
                 const UHypers& hypers, const arma::mat& X, const arma::vec& Y,
                 const UOpts& opts) {

  double MH_BD = 0.7;
  double MH_PRIOR = 0.4;

  int num_tree = hypers.num_tree;
  for(int t = 0; t < num_tree; t++) {
    // Rcout << "Getting backfit quantities";
    arma::vec Y_star = Y_hat - predict(forest[t], X, hypers);
    arma::vec res = Y - Y_star;

    if(unif_rand() < MH_PRIOR) {
      // Rcout << "Draw Prior";
      forest[t] = draw_prior(forest[t], X, res, hypers);
      // Rcout << "Done";
    }
    if(forest[t]->is_leaf || unif_rand() < MH_BD) {
      // Rcout << "BD step";
      birth_death(forest[t], X, res, hypers);
      // Rcout << "Done";
    }
    else {
      // Rcout << "Change step";
      perturb_decision_rule(forest[t], X, res, hypers);
      // Rcout << "Done";
    }
    if(opts.update_tau) forest[t]->UpdateTau(res, X, hypers);
    forest[t]->UpdateMu(res, X, hypers);
    Y_hat = Y_star + predict(forest[t], X, hypers);
  }
}

void UForest::IterateGibbs(arma::vec& Y_hat,
                          const arma::mat& X,
                          const arma::vec& Y) {


  // Rcout << "Backfitting";
  TreeBackfit(trees, Y_hat, hypers, X, Y, opts);
  arma::vec res = Y - Y_hat;
  arma::vec means = get_means(trees);
  if(opts.update_sigma) hypers.UpdateSigma(res);
  if(opts.update_sigma_mu) hypers.UpdateSigmaMu(means);
  // if(opts.update_s) UpdateS(trees, hypers);
  // if(opts.update_alpha) hypers.UpdateAlpha();
  // if(opts.update_num_tree) update_num_tree(forest, hypers, opts, Y, Y - Y_hat, X);

  Rcpp::checkUserInterrupt();

}

// Rcpp::List do_soft_bart(const arma::mat& X,
//                         const arma::vec& Y,
//                         const arma::mat& X_test,
//                         Hypers& hypers,
//                         const Opts& opts) {


//   std::vector<Node*> forest = init_forest(X, Y, hypers);

//   vec Y_hat = zeros<vec>(X.n_rows);

//   // Do burn_in

//   for(int i = 0; i < opts.num_burn; i++) {

//     // Don't update s for half of the burn-in
//     if(i < opts.num_burn / 2) {
//       // Rcout << "Iterating Gibbs\n";
//       IterateGibbsNoS(forest, Y_hat, hypers, X, Y, opts);
//     }
//     else {
//       IterateGibbsWithS(forest, Y_hat, hypers, X, Y, opts);
//     }

//     if((i+1) % opts.num_print == 0) {
//       // Rcout << "Finishing warmup " << i + 1 << ": tau = " << hypers.width << "\n";
//       Rcout << "Finishing warmup " << i + 1
//             // << " tau_rate = " << hypers.tau_rate
//             << " Number of trees = " << hypers.num_tree
//             << "\n"
//         ;
//     }

//   }

//   // Make arguments to return
//   mat Y_hat_train = zeros<mat>(opts.num_save, X.n_rows);
//   mat Y_hat_test = zeros<mat>(opts.num_save, X_test.n_rows);
//   vec sigma = zeros<vec>(opts.num_save);
//   vec sigma_mu = zeros<vec>(opts.num_save);
//   vec alpha = zeros<vec>(opts.num_save);
//   vec beta = zeros<vec>(opts.num_save);
//   vec gamma = zeros<vec>(opts.num_save);
//   mat s = zeros<mat>(opts.num_save, hypers.s.size());
//   // mat logZ = zeros<mat>(opts.num_save, hypers.s.size());
//   vec a_hat = zeros<vec>(opts.num_save);
//   vec b_hat = zeros<vec>(opts.num_save);
//   vec mean_log_Z = zeros<vec>(opts.num_save);
//   umat var_counts = zeros<umat>(opts.num_save, hypers.s.size());
//   vec tau_rate = zeros<vec>(opts.num_save);
//   uvec num_tree = zeros<uvec>(opts.num_save);
//   vec loglik = zeros<vec>(opts.num_save);
//   mat loglik_train = zeros<mat>(opts.num_save, Y_hat.size());

//   // Do save iterations
//   for(int i = 0; i < opts.num_save; i++) {
//     for(int b = 0; b < opts.num_thin; b++) {
//       IterateGibbsWithS(forest, Y_hat, hypers, X, Y, opts);
//     }

//     // Save stuff
//     Y_hat_train.row(i) = Y_hat.t();
//     Y_hat_test.row(i) = trans(predict(forest, X_test, hypers));
//     sigma(i) = hypers.sigma;
//     sigma_mu(i) = hypers.sigma_mu;
//     s.row(i) = trans(hypers.s);
//     // logZ.row(i) = trans(hypers.logZ);
//     a_hat(i) = hypers.a_hat;
//     b_hat(i) = hypers.b_hat;
//     mean_log_Z(i) = hypers.mean_log_Z;
//     var_counts.row(i) = trans(get_var_counts(forest, hypers));
//     alpha(i) = hypers.alpha;
//     beta(i) = hypers.beta;
//     gamma(i) = hypers.gamma;
//     tau_rate(i) = hypers.tau_rate;
//     loglik_train.row(i) = trans(loglik_data(Y,Y_hat,hypers));
//     loglik(i) = sum(loglik_train.row(i));
//     num_tree(i) = hypers.num_tree;

//     if((i + 1) % opts.num_print == 0) {
//       // Rcout << "Finishing save " << i + 1 << ": tau = " << hypers.width << "\n";
//       Rcout << "Finishing save " << i + 1 << "\n";
//     }

//   }

//   Rcout << "Number of leaves at final iterations:\n";
//   for(int t = 0; t < hypers.num_tree; t++) {
//     Rcout << leaves(forest[t]).size() << " ";
//     if((t + 1) % 10 == 0) Rcout << "\n";
//   }

//   List out;
//   out["y_hat_train"] = Y_hat_train;
//   out["y_hat_test"] = Y_hat_test;
//   out["sigma"] = sigma;
//   out["sigma_mu"] = sigma_mu;
//   out["s"] = s;
//   // out["logZ"] = logZ;
//   out["a_hat"] = a_hat;
//   out["b_hat"] = b_hat;
//   out["mean_log_Z"] = mean_log_Z;
//   out["alpha"] = alpha;
//   out["beta"] = beta;
//   out["gamma"] = gamma;
//   out["var_counts"] = var_counts;
//   out["tau_rate"] = tau_rate;
//   out["num_tree"] = num_tree;
//   out["loglik"] = loglik;
//   out["loglik_train"] = loglik_train;


//   return out;

// }



// double alpha_to_rho(double alpha, double scale) {
//   return alpha / (alpha + scale);
// }

// double rho_to_alpha(double rho, double scale) {
//   return scale * rho / (1.0 - rho);
// }



/*Note: Because the shape of the Dirichlet will mostly be small, we sample from
  the Dirichlet distribution by sampling log-gamma random variables using the
  technique of Liu, Martin, and Syring (2017+) and normalizing using the
  log-sum-exp trick */
// void UpdateS(std::vector<Node*>& forest, Hypers& hypers) {

//   // Get shape vector
//   vec shape_up = hypers.alpha / ((double)hypers.s.size()) * ones<vec>(hypers.s.size());
//   shape_up = shape_up + get_var_counts(forest, hypers);

//   // Sample unnormalized s on the log scale
//   for(int i = 0; i < shape_up.size(); i++) {
//     hypers.logZ(i) = rlgam(shape_up(i));
//   }
//   // Normalize s on the log scale, then exponentiate
//   hypers.logs = hypers.logZ - log_sum_exp(hypers.logZ);
//   hypers.s = exp(hypers.logs);
//   hypers.logZ = hypers.logs + rlgam(hypers.alpha);

// }

// // NOTE: the log-likelihood here is -n Gam(alpha/n) + alpha * mean_log_Z + (shape - 1) * log(alpha) - rate * alpha
// void Hypers::UpdateAlpha() {


//   // Get the Gamma approximation

//   double n = logZ.size();
//   double R = mean(logZ); mean_log_Z = R;
//   double alpha_hat = exp(log_sum_exp(logZ));
//   a_hat = alpha_shape_1 + alpha_hat * alpha_hat * Rf_trigamma(alpha_hat / n) / n;
//   b_hat = 1.0 / alpha_scale + (a_hat - alpha_shape_1) / alpha_hat +
//     Rf_digamma(alpha_hat / n) - R;
//   int M = 10;
//   for(int i = 0; i < M; i++) {
//     alpha_hat = a_hat / b_hat;
//     a_hat = alpha_shape_1 + alpha_hat * alpha_hat * Rf_trigamma(alpha_hat / n) / n;
//     b_hat = 1.0 / alpha_scale + (a_hat - alpha_shape_1) / alpha_hat +
//       Rf_digamma(alpha_hat / n) - R;
//   }
//   double A = a_hat * .75;
//   double B = b_hat * .75;

//   // double n = logZ.size();
//   // double R = sum(logZ);
//   // double alpha_hat = exp(log_sum_exp(logZ)) / n;
//   // a_hat = 1.0 + alpha_hat * alpha_hat * n * Rf_trigamma(alpha_hat);
//   // b_hat = (a_hat - 1.0) / alpha_hat + n * Rf_digamma(alpha_hat) - R;
//   // int M = 10;
//   // for(int i = 0; i < M; i++) {
//   //   alpha_hat = a_hat / b_hat;
//   //   a_hat = 1.0 + alpha_hat * alpha_hat * n * Rf_trigamma(alpha_hat);
//   //   b_hat = (a_hat - 1.0) / alpha_hat + n * Rf_digamma(alpha_hat) - R;
//   // }
//   // a_hat = a_hat / 1.3;
//   // b_hat = b_hat / 1.3;

//   // Sample from the gamma approximation
//   double alpha_prop = R::rgamma(A, 1.0 / B);


//   // Compute logliks
//   double loglik_new = - n * R::lgammafn(alpha_prop / n) + alpha_prop * R +
//     (alpha_shape_1 - 1.0) * log(alpha_prop) - alpha_prop / alpha_scale +
//     R::dgamma(alpha, A, 1.0 / B, 1);
//   double loglik_old = -n * R::lgammafn(alpha / n) + alpha * R +
//     (alpha_shape_1 - 1.0) * log(alpha) - alpha / alpha_scale +
//     R::dgamma(alpha_prop, A, 1.0 / B, 1);

//   // Accept or reject
//   if(log(unif_rand()) < loglik_new - loglik_old) {
//     alpha = alpha_prop;
//   }

//   // arma::vec logliks = zeros<vec>(rho_propose.size());
//   // rho_loglik loglik;
//   // loglik.mean_log_s = mean(logs);
//   // loglik.p = (double)s.size();
//   // loglik.alpha_scale = alpha_scale;
//   // loglik.alpha_shape_1 = alpha_shape_1;
//   // loglik.alpha_shape_2 = alpha_shape_2;

//   // for(int i = 0; i < rho_propose.size(); i++) {
//   //   logliks(i) = loglik(rho_propose(i));
//   // }

//   // logliks = exp(logliks - log_sum_exp(logliks));
//   // double rho_up = rho_propose(sample_class(logliks));
//   // alpha = rho_to_alpha(rho_up, alpha_scale);

// }

// // void Hypers::UpdateAlpha() {

// //   double rho = alpha_to_rho(alpha, alpha_scale);
// //   double psi = mean(log(s));
// //   double p = (double)s.size();

// //   double loglik = alpha * psi + Rf_lgammafn(alpha) - p * Rf_lgammafn(alpha / p) +
// //     logpdf_beta(rho, alpha_shape_1, alpha_shape_2);

// //   // 50 MH proposals
// //   for(int i = 0; i < 50; i++) {
// //     double rho_propose = Rf_rbeta(alpha_shape_1, alpha_shape_2);
// //     double alpha_propose = rho_to_alpha(rho_propose, alpha_scale);

// //     double loglik_propose = alpha_propose * psi + Rf_lgammafn(alpha_propose) -
// //       p * Rf_lgammafn(alpha_propose/p) +
// //       logpdf_beta(rho_propose, alpha_shape_1, alpha_shape_2);

// //     if(log(unif_rand()) < loglik_propose - loglik) {
// //       alpha = alpha_propose;
// //       rho = rho_propose;
// //       loglik = loglik_propose;
// //     }
// //   }
// // }

// // void Hypers::UpdateAlpha() {

// //   rho_loglik loglik;
// //   loglik.mean_log_s = mean(logs);
// //   loglik.p = (double)s.size();
// //   loglik.alpha_scale = alpha_scale;
// //   loglik.alpha_shape_1 = alpha_shape_1;
// //   loglik.alpha_shape_2 = alpha_shape_2;

// //   double rho = alpha_to_rho(alpha, alpha_scale);
// //   rho = slice_sampler(rho, loglik, 0.1, 0.0 + exp(-10.0), 1.0);
// //   alpha = rho_to_alpha(rho, alpha_scale);
// // }


// // double loglik_data(const arma::vec& Y, const arma::vec& Y_hat, const Hypers& hypers) {
// //   vec res = Y - Y_hat;
// //   double out = -0.5 * Y.size() * log(M_2_PI * pow(hypers.sigma,2.0)) -
// //     dot(res, res) * 0.5 / pow(hypers.sigma,2.0);
// //   return out;
// // }

// arma::vec loglik_data(const arma::vec& Y, const arma::vec& Y_hat, const Hypers& hypers) {
//   vec res = Y - Y_hat;
//   vec out = zeros<vec>(Y.size());
//   for(int i = 0; i < Y.size(); i++) {
//     out(i) = -0.5 * log(M_2_PI * pow(hypers.sigma,2)) - 0.5 * pow(res(i) / hypers.sigma, 2);
//   }
//   return out;
// }

arma::mat UForest::do_gibbs(const arma::mat& X,
                           const arma::vec& Y,
                           const arma::mat& X_test,
                           int num_iter) {

  vec Y_hat = predict(trees, X, hypers);
  mat Y_out = zeros<mat>(num_iter, X_test.n_rows);

  for(int i = 0; i < num_iter; i++) {
    IterateGibbs(Y_hat, X, Y);
    vec tmp = predict(trees, X_test, hypers);
    Y_out.row(i) = tmp.t();
    if((i+1) % opts.num_print == 0) {
      Rcout << "Finishing iteration " << i+1 << std::endl;
    }
  }

  return Y_out;

}

arma::vec UForest::predict_vec(const arma::mat& X_test) {
  return predict(trees, X_test, hypers);
}

void UForest::set_s(const arma::vec& s_) {
  hypers.s = s_;
  hypers.logs = log(s_);
}

arma::vec UForest::get_s() {
  return hypers.s;
}

Rcpp::List UForest::get_params() {

  List out;
  out["alpha"] = hypers.alpha;
  out["sigma"] = hypers.sigma;
  out["sigma_mu"] = hypers.sigma_mu;

  return out;
}

// arma::uvec Forest::get_counts() {
//   return get_var_counts(trees, hypers);
// }

// arma::umat Forest::get_tree_counts() {
//   for(int t = 0; t < hypers.num_tree; t++) {
//     std::vector<Node*> tree;
//     tree.resize(0);
//     tree.push_back(trees[t]);
//     tree_counts.col(t) = get_var_counts(tree, hypers);
//   }

//   return tree_counts;
// }

RCPP_MODULE(mod_forest) {

  class_<UForest>("UForest")

    .constructor<Rcpp::List, Rcpp::List>()
    .method("do_gibbs", &UForest::do_gibbs)
    .method("get_s", &UForest::get_s)
    .method("get_params", &UForest::get_params)
    .method("predict", &UForest::predict_vec)
    // .method("get_counts", &Forest::get_counts)
     .method("set_s", &UForest::set_s)
    // .method("get_tree_counts", &Forest::get_tree_counts)
//     .field("num_gibbs", &Forest::num_gibbs)
    ;

}
