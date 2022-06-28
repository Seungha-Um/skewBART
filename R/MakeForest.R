multi_MakeForest <- function(hypers, opts) {
  mf <- Module(module = "multi_mod_forest", PACKAGE = "skewBART")
  return(new(mf$Forest, hypers, opts))
}
