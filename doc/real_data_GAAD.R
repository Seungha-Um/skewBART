## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ---- eval = TRUE, include=FALSE----------------------------------------------
## Load library
library(kableExtra)
options(knitr.table.format = 'markdown')
library(skewBART)
library(zeallot)
library(ggplot2)
library(glmnet)
library(methods)
library(truncnorm)
library(Matrix)
library(TruncatedNormal)
library(mvtnorm)
library(dplyr)

## ---- eval = FALSE------------------------------------------------------------
#  library(kableExtra)
#  options(knitr.table.format = 'markdown')
#  library(skewBART)
#  library(zeallot)
#  library(ggplot2)
#  library(glmnet)
#  library(methods)
#  library(truncnorm)
#  library(Matrix)
#  library(TruncatedNormal)
#  library(mvtnorm)
#  library(dplyr)

## -----------------------------------------------------------------------------
data(GAAD)
Y <- GAAD$subCAL
X <- GAAD %>% select("Age", "Female", "Bmi", "Smoker", "Hba1cfull")

## -----------------------------------------------------------------------------
hypers <- UHypers(X, Y, num_tree = 20)
opts <- UOpts(num_burn = 10, num_save = 10)

## -----------------------------------------------------------------------------
## Fit the model
fitted_skewbart <- skewBART(X, Y, X, hypers, opts)

## -----------------------------------------------------------------------------
mean(fitted_skewbart$alpha)

## -----------------------------------------------------------------------------
mean(fitted_skewbart$lambda)

## -----------------------------------------------------------------------------
uni_samples <- sapply(1:opts$num_save, function(i) fitted_skewbart$y_hat_test[i,] 
                  + fitted_skewbart$lambda[i] * sqrt(2/pi))
# The estimated CAL for the first 20 subjects
rowMeans(uni_samples)[1:20]

## -----------------------------------------------------------------------------
library(dplyr)
data(GAAD)
X <- GAAD %>% select("Age", "Female", "Bmi", "Smoker", "Hba1cfull")
Y <- GAAD %>% select(subCAL, subPD)

## -----------------------------------------------------------------------------
hypers <- Hypers(X = as.matrix(X), Y = Y, num_tree = 20)
opts <- Opts(num_burn = 10, num_save = 10)

## -----------------------------------------------------------------------------
fitted_Multiskewbart <- MultiskewBART(X = X, Y = Y, test_X = X, hypers=hypers, opts=opts)

## -----------------------------------------------------------------------------
colMeans(fitted_Multiskewbart$lambda)

## -----------------------------------------------------------------------------
# the first response
ind <- 1 # response index
samples_CAL <- sapply(1:opts$num_save, function(i) fitted_Multiskewbart$y_hat_test[,ind,i] + 
                        fitted_Multiskewbart$lambda[i,ind] * sqrt(2/pi))

# second response
ind <- 2 # response index
samples_PPD <- sapply(1:opts$num_save, function(i) fitted_Multiskewbart$y_hat_test[,ind,i] + 
                        fitted_Multiskewbart$lambda[i,ind] * sqrt(2/pi))

## -----------------------------------------------------------------------------
rowMeans(samples_CAL)[1:20]

## -----------------------------------------------------------------------------
rowMeans(samples_PPD)[1:20]

## -----------------------------------------------------------------------------
apply(fitted_Multiskewbart$Sigma, c(1,2), mean)

## -----------------------------------------------------------------------------
est_cov <- apply(fitted_Multiskewbart$Sigma, c(1,2), mean)
est_cov[1,2]/sqrt(est_cov[1,1])/sqrt(est_cov[2,2])

## -----------------------------------------------------------------------------
# MSE from univariate skewBART
mean((Y[,1] - rowMeans(uni_samples))^2) 

# MSE from multivariate skewBART
mean((Y[,1] - rowMeans(samples_CAL))^2)

