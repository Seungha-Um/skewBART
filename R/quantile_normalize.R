trank <- function(x) {
  x_unique <- unique(x)
  x_ranks <- rank(x_unique, ties.method = "max")
  tx <- x_ranks[match(x,x_unique)] - 1
  
  tx <- tx / length(unique(tx))
  tx <- tx / max(tx)
  
  return(tx)
}

quantile_normalize_bart <- function(X) {
  apply(X = X, MARGIN = 2, trank)
}

preprocess_df <- function(X) {
  stopifnot(is.data.frame(X))
  
  X <- model.matrix(~.-1, data = X)
  group <- attr(X, "assign") - 1
  
  return(list(X = X, group = group))
  
}

quantile_normalize <- function(X ,test_X){
  X_trans <- quantile_normalize_bart(rbind(X, test_X))
  idx_train <- 1:nrow(X)
  X <- X_trans[idx_train,,drop=FALSE]
  test_X <- X_trans[-idx_train,,drop=FALSE]
  return(list(X=X, test_X=test_X))
}
