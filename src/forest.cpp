#include "forest.h"

using namespace arma;
using namespace Rcpp;


Forest::Forest(Rcpp::List hypers, Rcpp::List opts) : hypers(hypers), opts(opts) {
  trees.resize(this->hypers.num_tree);
  for(int i = 0; i < this->hypers.num_tree; i++) {
    trees[i] = new MNode(this->hypers);
  }
}


Forest::~Forest() {
  for(int i = 0; i < trees.size(); i++) delete trees[i];
}



Opts::Opts(Rcpp::List opts) {
  num_burn        = opts["num_burn"];
  num_thin        = opts["num_thin"];
  num_save        = opts["num_save"];
  num_print       = opts["num_print"];
  update_Sigma_mu = opts["update_Sigma_mu"];
  update_Sigma    = opts["update_Sigma"];
  update_s        = opts["update_s"];
  update_alpha    = opts["update_alpha"];
}


Hypers::Hypers(Rcpp::List hypers)
{
  alpha        = hypers["alpha"];
  beta         = hypers["beta"];
  gamma        = hypers["gamma"];
  Sigma_hat    = as<mat>(hypers["Sigma_hat"]);
  Sigma        = Sigma_hat;
  Sigma_mu_hat = as<mat>(hypers["Sigma_mu_hat"]);
  Sigma_mu     = Sigma_mu_hat;
  A            = inv(Sigma);
  B            = inv(Sigma_mu);
  temperature  = hypers["temperature"];
  num_tree    = hypers["num_tree"];

  sp_mat tmp = as<sp_mat>(hypers["group"]);
  group = zeros<sp_umat>(tmp.n_rows, tmp.n_cols);
  for(sp_mat::iterator it = tmp.begin(); it != tmp.end(); ++it) {
    group(it.row(), it.col()) = std::round(*it);
  }
  group_size = sum(group, 1);
  p_group_var = tmp;
  for(sp_umat::iterator it = group.begin(); it != group.end(); ++it) {
    p_group_var(it.row(), it.col()) = 1.0 / group_size(it.row());
  }


  // Initialize s
  s = 1.0 / group.n_rows * ones<vec>(group.n_rows);
  logs = log(s);
}



MNode::MNode(const Hypers& hypers) {

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

  mu = zeros<vec>(hypers.Sigma.n_rows);
  ss = new SuffStats(hypers.Sigma.n_rows);
}

MNode::MNode(MNode* parent, const Hypers& hypers, bool is_left) {
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
  mu = zeros<vec>(hypers.Sigma.n_rows);
  ss = new SuffStats(hypers.Sigma.n_rows);

  // Update the bounds
  int pvar = parent->var;
  if(is_left) {
    uppers(pvar)= parent->val;
  }
  else {
    lowers(pvar) = parent->val;
  }


}

MNode::~MNode() {
  if(!is_leaf) {
    delete left;
    delete right;
  }
  delete ss;
}

int Hypers::SampleVar() const {
  uvec group_and_var = zeros<uvec>(2);
  group_and_var(0) = sample_class(s);
  group_and_var(1) = sample_class_row(p_group_var, group_and_var(0));
  return group_and_var(1);
}


void MNode::BirthLeaves(const Hypers& hypers) {
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
    left = new MNode(this, hypers, true);
    right = new MNode(this, hypers, false);
  }
}

void MNode::GenBelow(const Hypers& hypers) {
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

void MNode::get_limits_below() {

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


void MNode::DeleteLeaves() {
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


void MNode::UpdateMu(const arma::mat& Y, const arma::mat& X,
                     const Hypers& hypers) {

  GetSuffStats(this, Y, X, hypers);
  MNodevec leafs = leaves(this);
  int num_leaves = leafs.size();
  for(int l = 0; l < num_leaves; l++) {
    mat n_A_plus_B = leafs[l]->ss->n * hypers.A + hypers.B;
    vec Y_tilde = solve(n_A_plus_B, hypers.A * leafs[l]->ss->sum_Y);
    leafs[l]->mu = rmvnorm(Y_tilde, n_A_plus_B);
  }

}

void GetSuffStats(MNode* n, const arma::mat& Y,
                  const arma::mat& X,
                  const Hypers& hypers) {

  n->ResetSuffStat();
  int N = Y.n_rows;
  for(int i = 0; i < N; i++) {
    vec y = trans(Y.row(i));
    mat yyt = y * y.t();
    vec x = trans(X.row(i));
    GetSuffStats(n, y, yyt, x, hypers);
  }
}

void GetSuffStats(MNode* n, const arma::vec& y, const arma::mat& yyt,
                  const arma::vec& x, const Hypers& hypers) {

  n->ss->n += 1.0;
  n->ss->sum_Y = n->ss->sum_Y + y;
  n->ss->sum_YYt = n->ss->sum_YYt + yyt;
  if(!(n->is_leaf)) {
    if(x(n->var) < n->val) {
      GetSuffStats(n->left, y, yyt, x, hypers);
    }
    else {
      GetSuffStats(n->right, y, yyt, x, hypers);
    }
  }
}

// RECALL: A = inv(Sigma) and B = inv(Sigma_mu)
double LogLT(MNode* n, const arma::mat& Y,
             const arma::mat& X, const Hypers& hypers) {

  GetSuffStats(n, Y, X, hypers);

  // Rcout << "Leaves ";
  std::vector<MNode*> leafs = leaves(n);
  int num_leaves = leafs.size();

  double out = 0.0;

  double log_det_B, sign;
  log_det(log_det_B, sign, hypers.B);

  for(int l = 0; l < num_leaves; l++) {
    mat n_A_plus_B = leafs[l]->ss->n * hypers.A + hypers.B;
    vec Y_tilde = solve(n_A_plus_B, hypers.A * leafs[l]->ss->sum_Y);
    out -= 0.5 * trace(hypers.A * leafs[l]->ss->sum_YYt);
    out += 0.5 * as_scalar(Y_tilde.t() * n_A_plus_B * Y_tilde);
    out += 0.5 * log_det_B;

    double log_det_naplusb, sign2; log_det(log_det_naplusb, sign2, n_A_plus_B);
    out -= 0.5 * log_det_naplusb;
  }

  return out;
}

double SplitProb(MNode* node, const Hypers& hypers) {
  return hypers.gamma * pow(1.0 + node->depth, -hypers.beta);
}

void leaves(MNode* x, MNodevec& leafs) {
  if(x->is_leaf) {
    leafs.push_back(x);
  }
  else {
    leaves(x->left, leafs);
    leaves(x->right, leafs);
  }
}

std::vector<MNode*> leaves(MNode* x) {
  std::vector<MNode*> leafs(0);
  leaves(x, leafs);
  return leafs;
}



std::vector<MNode*> not_grand_branches(MNode* tree) {
  std::vector<MNode*> ngb(0);
  not_grand_branches(ngb, tree);
  return ngb;
}

void not_grand_branches(MNodevec& ngb, MNode* node) {
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
void branches(MNode* n, MNodevec& branch_vec) {
  if(!(n->is_leaf)) {
    branch_vec.push_back(n);
    branches(n->left, branch_vec);
    branches(n->right, branch_vec);
  }
}

std::vector<MNode*> branches(MNode* root) {
  std::vector<MNode*> branch_vec;
  branch_vec.resize(0);
  branches(root, branch_vec);
  return branch_vec;
}

MNode* rand(MNodevec& ngb) {

  int N = ngb.size();
  arma::vec p = ones<vec>(N) / ((double)(N));
  int i = sample_class(p);
  return ngb[i];
}


arma::uvec get_var_counts(MNodevec& forest, const Hypers& hypers) {
  arma::uvec counts = zeros<uvec>(hypers.group.n_cols);
  int num_tree = forest.size();
  for(int t = 0; t < num_tree; t++) {
    get_var_counts(counts, forest[t], hypers);
  }
  return counts;
}

void get_var_counts(arma::uvec& counts, MNode* node, const Hypers& hypers) {
  if(!node->is_leaf) {
    counts(node->var) = counts(node->var) + 1;
    get_var_counts(counts, node->left, hypers);
    get_var_counts(counts, node->right, hypers);
  }
}

// arma::uvec get_var_counts(MNodevec& forest, const Hypers& hypers) {
//   arma::uvec counts = zeros<uvec>(hypers.s.size());
//   int num_tree = forest.size();
//   for(int t = 0; t < num_tree; t++) {
//     get_var_counts(counts, forest[t], hypers);
//   }
//   return counts;
// }

// void get_var_counts(arma::uvec& counts, Node* node, const Hypers& hypers) {
//   if(!node->is_leaf) {
//     int group_idx = hypers.group(node->var);
//     counts(group_idx) = counts(group_idx) + 1;
//     get_var_counts(counts, node->left, hypers);
//     get_var_counts(counts, node->right, hypers);
//   }
// }

// arma::vec get_means(Nodevec& forest) {
//   std::vector<double> means(0);
//   int num_tree = forest.size();
//   for(int t = 0; t < num_tree; t++) {
//     get_means(forest[t], means);
//   }

//   // Convert std::vector to armadillo vector, deep copy
//   vec out(&(means[0]), means.size());
//   return out;
// }
//
// void get_means(Node* node, std::vector<double>& means) {
//
//   if(node->is_leaf) {
//     means.push_back(node->mu);
//   }
//   else {
//     get_means(node->left, means);
//     get_means(node->right, means);
//   }
// }


arma::mat predict(const MNodevec& forest, const arma::mat& X,
                  const Hypers& hypers) {

  mat out = zeros<mat>(X.n_rows, hypers.Sigma.n_rows);
  int num_tree = forest.size();

  for(int t = 0; t < num_tree; t++) {
    out = out + predict(forest[t], X, hypers);
  }
  return out;
}

// LEAVE OFF HERE
arma::mat predict(MNode* n, const arma::mat& X, const Hypers& hypers) {
  mat out = zeros<mat>(X.n_rows, hypers.Sigma.n_rows);
  int N = X.n_rows;
  for(int i = 0; i < N; i++) {
    vec x = trans(X.row(i));
    out.row(i) = trans(predict(n, x, hypers));
  }
  return out;
}

arma::vec predict(MNode* n, const arma::vec& x, const Hypers& hypers) {
  if(n->is_leaf) return n->mu;

  bool go_left = x(n->var) < n->val;
  if(go_left) {
    return predict(n->left, x, hypers);
  }
  else {
    return predict(n->right, x, hypers);
  }
}


double growth_prior(int leaf_depth, const Hypers& hypers) {
  return hypers.gamma * pow(1.0 + leaf_depth, -hypers.beta);
}


MNode* birth_node(MNode* tree, double* leaf_node_probability) {
  std::vector<MNode*> leafs = leaves(tree);
  MNode* leaf = rand(leafs);
  *leaf_node_probability = 1.0 / ((double)leafs.size());

  return leaf;
}

double probability_node_birth(MNode* tree) {
  return tree->is_leaf ? 1.0 : 0.5;
}


MNode* death_node(MNode* tree, double* p_not_grand) {
  std::vector<MNode*> ngb = not_grand_branches(tree);
  MNode* branch = rand(ngb);
  *p_not_grand = 1.0 / ((double)ngb.size());

  return branch;
}

void birth_death(MNode* tree, const arma::mat& X, const arma::mat& Y,
                 const Hypers& hypers) {


  double p_birth = probability_node_birth(tree);

  if(unif_rand() < p_birth) {
    node_birth(tree, X, Y, hypers);
  }
  else {
    node_death(tree, X, Y, hypers);
 }
}


void node_birth(MNode* tree, const arma::mat& X, const arma::mat& Y,
                const Hypers& hypers) {

  // Rcout << "Sample leaf";
  double leaf_probability = 0.0;
  MNode* leaf = birth_node(tree, &leaf_probability);

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
  std::vector<MNode*> ngb = not_grand_branches(tree);
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

void node_death(MNode* tree, const arma::mat& X, const arma::mat& Y,
                const Hypers& hypers) {

  // Select branch to kill Children
  double p_not_grand = 0.0;
  MNode* branch = death_node(tree, &p_not_grand);

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
  MNode* left = branch->left;
  MNode* right = branch->right;
  branch->left = branch;
  branch->right = branch;
  branch->is_leaf = true;

  // Compute likelihood after
  double ll_after = LogLT(tree, Y, X, hypers) + log(1.0 - leaf_prob);

  // Compute backwards transition
  std::vector<MNode*> leafs = leaves(tree);
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

MNode* draw_prior(MNode* tree, const arma::mat& X, const arma::mat& Y, const Hypers& hypers) {

  // Compute loglik before
  MNode* tree_0 = tree;
  double loglik_before = LogLT(tree_0, Y, X, hypers);

  // Make new tree and compute loglik after
  MNode* tree_1 = new MNode(hypers);
  tree_1->GenBelow(hypers);
  double loglik_after = LogLT(tree_1, Y, X, hypers);

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

double calc_cutpoint_likelihood(MNode* node) {
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

std::vector<double> get_perturb_limits(MNode* branch) {
  double min = 0.0;
  double max = 1.0;

  MNode* n = branch;
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
  std::vector<MNode*> left_branches = branches(n->left);
  std::vector<MNode*> right_branches = branches(n->right);
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


void perturb_decision_rule(MNode* tree,
                           const arma::mat& X,
                           const arma::mat& Y,
                           const Hypers& hypers) {

  // Randomly choose a branch; if no branches, we automatically reject
  std::vector<MNode*> bbranches = branches(tree);
  if(bbranches.size() == 0)
    return;

  // Select the branch
  MNode* branch = rand(bbranches);

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


void TreeBackfit(MNodevec& forest, arma::mat & Y_hat,
                 const Hypers& hypers, const arma::mat& X, const arma::mat& Y,
                 const Opts& opts) {

  double MH_BD = 0.7;
  double MH_PRIOR = 0.4;

  int num_tree = hypers.num_tree;
  for(int t = 0; t < num_tree; t++) {
    // Rcout << "Getting backfit quantities";
    arma::mat Y_star = Y_hat - predict(forest[t], X, hypers);
    arma::mat res = Y - Y_star;

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
    // Rcout << "Update mu";
    forest[t]->UpdateMu(res, X, hypers);
    // Rcout << "Refit";
    Y_hat = Y_star + predict(forest[t], X, hypers);
    // Rcout << "Done";
  }
}










void Forest::IterateGibbs(arma::mat& Y_hat,
                          const arma::mat& X,
                          const arma::mat& Y) {


  TreeBackfit(trees, Y_hat, hypers, X, Y, opts);
  arma::mat res = Y - Y_hat;
  // arma::mat means = get_means(trees);
  // if(opts.update_sigma) hypers.UpdateSigma(res);
  // if(opts.update_sigma_mu) hypers.UpdateSigmaMu(means);
  // if(opts.update_s) UpdateS(trees, hypers);
  // if(opts.update_alpha) hypers.UpdateAlpha();
  // if(opts.update_num_tree) update_num_tree(forest, hypers, opts, Y, Y - Y_hat, X);

  Rcpp::checkUserInterrupt();

}

arma::cube Forest::do_gibbs(const arma::mat& X,
                            const arma::mat& Y,
                            const arma::mat& X_test,
                            int num_iter) {

  mat Y_hat = predict(trees, X, hypers);
  cube Y_out = zeros<cube>(X_test.n_rows, Y.n_cols, num_iter);

  for(int i = 0; i < num_iter; i++) {
    // Rcout << "Iterate";
    IterateGibbs(Y_hat, X, Y);
    // Rcout << "Predict";
    Y_out.slice(i) = predict(trees, X_test, hypers);
    if((i+1) % opts.num_print == 0) {
      Rcout << "\rFinishing iteration " << i+1 << "\t\t\t";
    }
  }

  return Y_out;

}

void Forest::set_s(const arma::vec& s_) {
  hypers.s = s_;
  hypers.logs = log(s_);
}

arma::vec Forest::get_s() {
  return hypers.s;
}

void Forest::set_sigma(const arma::mat& Sigma_) {
  hypers.Sigma = Sigma_;
  hypers.A = inv(Sigma_);
}

Rcpp::List Forest::get_params() {

  List out;
  out["alpha"] = hypers.alpha;
  out["sigma"] = hypers.Sigma;
  out["sigma_mu"] = hypers.Sigma_mu;

  return out;
}

arma::mat Forest::predict_mat(const arma::mat& X_test) {
  return predict(trees, X_test, hypers);
}

arma::uvec Forest::get_counts() {
  return get_var_counts(trees, hypers);
}

RCPP_MODULE(multi_mod_forest) {

  class_<Forest>("Forest")

    .constructor<Rcpp::List, Rcpp::List>()
    .method("do_gibbs", &Forest::do_gibbs)
    .method("get_s", &Forest::get_s)
    .method("set_sigma", &Forest::set_sigma)
    // .method("get_params", &Forest::get_params)
    .method("predict", &Forest::predict_mat)
    .method("get_counts", &Forest::get_counts)
    .method("set_s", &Forest::set_s)
    ;

}
