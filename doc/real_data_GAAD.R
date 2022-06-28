## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ---- eval = TRUE, include=TRUE-----------------------------------------------
## Load library
library(tidyverse) # Load the tidyverse, mainly for ggplot
library(kableExtra) # For fancy tables
library(skewBART) # Our package
library(zeallot) # For the %<-% operator, used when generating data
options(knitr.table.format = 'markdown')

## -----------------------------------------------------------------------------
data(GAAD)
Y <- GAAD$subCAL
X <- GAAD %>% dplyr::select(Age, Female, Bmi, Smoker, Hba1cfull)

## -----------------------------------------------------------------------------
hypers <- UHypers(X, Y, num_tree = 20)
opts <- UOpts(num_burn = 250, num_save = 250)

## -----------------------------------------------------------------------------
x_grid <- expand.grid(Hba1cfull = seq(5, 16.4, length = 20), 
                      Smoker = c(0,1), Female = c(0,1))

## ---- warning=FALSE-----------------------------------------------------------
set.seed(77777)
fitted_skewbart <- skewBART_pd(X, Y, vars = c("Hba1cfull", "Smoker", "Female"), 
                               x_grid = x_grid,
                               hypers = hypers, opts = opts)

## -----------------------------------------------------------------------------
set.seed(77777)
PD <- GAAD$subPD
hypers <- UHypers(X, PD, num_tree = 20)
fitted_skewbart_pd <- skewBART_pd(X, PD, vars = colnames(x_grid), 
                                  x_grid = x_grid, 
                                  hypers = hypers, opts = opts)

## ---- warning = FALSE---------------------------------------------------------
set.seed(77777)
hypers <- UHypers(X, log(Y), num_tree = 20)
fitted_skewbart_log <- skewBART(X, log(Y), X, hypers, opts)

## -----------------------------------------------------------------------------
fitted_skewbart$loo$estimates["elpd_loo",1]

## -----------------------------------------------------------------------------
fitted_skewbart_log$loo$estimates["elpd_loo",1] - sum(log(Y))

## -----------------------------------------------------------------------------
hist(fitted_skewbart$alpha)
hist(fitted_skewbart_log$alpha)

## -----------------------------------------------------------------------------
head(fitted_skewbart$partial_dependence_samples)
head(fitted_skewbart$partial_dependence_summary)

## -----------------------------------------------------------------------------
ggplot(fitted_skewbart$partial_dependence_summary, 
      aes(x = Hba1cfull, y = y_hat_mean, color = interaction(Female, Smoker))) +
  geom_point() + geom_line() + theme_bw() + ylab("Predicted CAL") + 
  theme(legend.position = "bottom")

## -----------------------------------------------------------------------------
qplot(fitted_skewbart_log$y_hat_train_mean, log(Y)) +
  geom_abline(slope = 1, intercept = 0) + theme_bw() +
  xlab("Predicted log-CAL") + ylab("Observed log-CAL")

## -----------------------------------------------------------------------------
X <- GAAD %>% select("Age", "Female", "Bmi", "Smoker", "Hba1cfull") %>% as.matrix()
Y <- GAAD %>% select(subCAL, subPD) %>% as.matrix()

## -----------------------------------------------------------------------------
hypers <- Hypers(X = X, Y = Y, num_tree = 20)
opts <- Opts(num_burn = 50, num_save = 50, num_print = 10)

## ---- warning = FALSE---------------------------------------------------------
set.seed(77777)
fitted_Multiskewbart <- MultiskewBART_pd(
  X = X, Y = Y, vars = colnames(x_grid), x_grid = x_grid, hypers = hypers, 
  opts = opts
)

## -----------------------------------------------------------------------------
fitted_Multiskewbart$loo

## -----------------------------------------------------------------------------
head(fitted_Multiskewbart$partial_dependence_summary)
fitted_Multiskewbart$partial_dependence_summary %>%
  ggplot(aes(x = Hba1cfull, y = y_hat_mean, 
             color = interaction(Smoker, Female))) + 
  geom_line() + geom_point() + facet_wrap(.~outcome, scales = "free_y") + 
  theme_bw() +theme(legend.position = "bottom") + ylab("Prediction") +
  xlab("Hba1c")

## -----------------------------------------------------------------------------
skew_pd_cal <- fitted_skewbart$partial_dependence_summary %>%
  mutate(outcome = "subCAL", method = "skewBART")
skew_pd_pd <- fitted_skewbart_pd$partial_dependence_summary %>%
  mutate(outcome = "subPD", method = "skewBART")
mskew_pd <- fitted_Multiskewbart$partial_dependence_summary %>% 
  mutate(method = "MultiskewBART")
skew_pd <- rbind(skew_pd_cal, skew_pd_pd, mskew_pd)

skew_pd %>% 
  ggplot(aes(x = Hba1cfull, y = y_hat_mean, 
             color = interaction(Smoker, Female))) + 
  geom_line() + geom_point() + facet_grid(outcome~method, scales = "free_y") + 
  theme_bw() +theme(legend.position = "bottom") + ylab("Prediction") +
  xlab("Hba1c")

