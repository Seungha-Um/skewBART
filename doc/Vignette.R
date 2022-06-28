## ----setup, include=FALSE-----------------------------------------------------
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

## ---- eval = TRUE, include=TRUE-----------------------------------------------
## Create a function for Friedman’s example
sim_fried <- function(N, P, alpha, sigma) {
  lambda <- alpha * sigma/sqrt(1+alpha^2)
  tau <- sigma/sqrt(1+alpha^2)
  X <- matrix(runif(N * P), nrow = N)
  mu <- 10 * sin(pi * X[,1] * X[,2]) + 20 * (X[,3] - 0.5)^2 + 10 * X[,4] + 5 * X[,5]
  Z <- abs(rnorm(N, mean=0, sd=1) )
  Y <- mu + lambda * Z + rnorm(N, mean=0, sd=sqrt(tau))
  EY <- mu + lambda * sqrt(2/pi)
  return(list(X = X, Y = Y, EY = EY, mu = mu, Z=Z, tau = tau, lambda = lambda))
}

## ---- eval = TRUE, include=TRUE-----------------------------------------------
## Traning dataset : n = 100 observations, P = 5 covariates, sigma = 1, alpha = 3 
set.seed(12345)
c(X,Y,EY,mu,Z,tau,lambda) %<-% sim_fried(100, 5, 1, 3)
c(test_X,test_Y,test_EY,test_mu,test_Z,test_tau,test_lambda)  %<-% sim_fried(50, 5, 1 ,3)

## ---- eval = TRUE, include=TRUE-----------------------------------------------
hypers <- UHypers(X, Y, num_tree = 20)
opts <- UOpts(num_burn = 250, num_save = 250)

## ---- eval = TRUE, include=TRUE-----------------------------------------------
## Fit the model
fitted_skewbart <- skewBART(X = X, Y = Y, test_X = test_X, 
                            hypers = hypers, opts = opts)

## ---- eval = TRUE, include=TRUE-----------------------------------------------
mean(fitted_skewbart$alpha)

## ---- eval = TRUE, include=TRUE-----------------------------------------------
fitted_skewbart$y_hat_test_mean

## ---- eval = TRUE, include=TRUE, fig.width = 7--------------------------------
# create a dataframe for the estimated alpha
df <- data.frame(iteration = 1:length(fitted_skewbart$alpha), alpha=fitted_skewbart$alpha)
p1 <- ggplot(df) + geom_line(aes(iteration, alpha), color = "darkgreen", alpha=0.7) + theme_bw() 

# create a dataframe for the fitted values
df_fitted <- data.frame(fitted = fitted_skewbart$y_hat_test_mean, observed = test_mu)
p2 <- ggplot(df_fitted) + 
  geom_point(aes(fitted, observed), alpha=0.8, shape=2, color = "darkgreen") +
  geom_abline(intercept = 0, slope = 1, alpha=0.8) +
  theme_bw() 
library(ggpubr)
ggarrange(p1, p2)

## ---- eval = TRUE-------------------------------------------------------------
## Create a function for Friedman example with multivariate framework
sim_data_multi <- function(N, P, lambda, tau, rho) {
  X <- matrix(runif(N * P), nrow = N)
  mu <- 10 * sin(pi * X[,1] * X[,2]) + 20 * (X[,3] - 0.5)^2 + 10 * X[,4] + 5 * X[,5] 
  Z <- cbind(lambda[1] * abs(rnorm(N)), lambda[2] * abs(rnorm(N)))
  Sigma <- matrix(c(tau[1], sqrt(tau[1]*tau[2])*rho, sqrt(tau[1]*tau[2])*rho, tau[2]), 2, 2)
  Err <- MASS::mvrnorm(n=N, mu=c(0,0), Sigma = Sigma)
  Y <- cbind(mu, mu) + Z + Err
  EY <- rbind(mu, mu) + lambda * sqrt(2/pi)
  return( list(X = X, Y = Y, EY=EY, mu = mu, lambda = lambda, tau=tau, Z= Z, Sigma = Sigma) )
}

## ---- eval = TRUE-------------------------------------------------------------
## Simulate dataset
## Traning dataset : n = 100 observations, P = 5 covariates, 
## lambda = (2,3), tau = c(1,1), rho = 0.5.
set.seed(12345)
c(X,Y,EY,mu,lambda,tau,Z,Sigma) %<-% sim_data_multi(100, 5, c(2,3), c(1,1), 0.5)
c(test_X,test_Y,test_EY,test_mu,test_lambda,test_tau,test_Z,test_Sigma) %<-%
  sim_data_multi(50, 5, c(2,3), c(1,1), 0.5)

## ---- eval = TRUE-------------------------------------------------------------
## Create a list of the hyperparameters of the model. 
hypers <- Hypers(X = X, Y = Y, num_tree = 20)
opts <- Opts(num_burn = 50, num_save = 50)

## ---- eval = TRUE, include=TRUE-----------------------------------------------
fitted_Multiskewbart <- MultiskewBART(X = X, Y = Y, test_X = test_X, hypers=hypers, opts=opts) 

## ---- eval = TRUE, include=TRUE-----------------------------------------------
mean(fitted_Multiskewbart$lambda[,1])
mean(fitted_Multiskewbart$lambda[,2])

## ---- eval = TRUE, include=TRUE-----------------------------------------------
head(fitted_Multiskewbart$y_hat_test_mean)

## ---- eval = TRUE, fig.width=7------------------------------------------------
# create a dataframe for estimated skewness levels (lambda)
df <- data.frame(iteration = 1:length(fitted_Multiskewbart$lambda[,1]),
                 lambda = c(fitted_Multiskewbart$lambda[,1], fitted_Multiskewbart$lambda[,2]),
                 grp = rep(c("lam1","lam2"), each=length(fitted_Multiskewbart$lambda[,1])))

ggplot(df) + geom_line(aes(iteration, lambda, colour=grp), alpha=0.7) +
  theme_bw() + facet_wrap(~grp) +
  scale_colour_manual(values=c(lam1 = "darkgreen", lam2= "brown")) +
  theme(legend.title=element_blank())

## ---- eval = TRUE, fig.width=7------------------------------------------------
df_fitted <- data.frame(fitted = c(fitted_Multiskewbart$y_hat_test_mean),
                        observed = rep(test_mu, 2), 
                        grp = rep(c("response_1", "response_2"),each = length(test_mu)))

ggplot(df_fitted) + geom_point(aes(fitted, observed, colour=grp), alpha=0.8, shape=2) +
  geom_abline(intercept = 0, slope = 1, alpha=0.8) +
  scale_colour_manual(values=c(response_1 = "darkgreen", response_2= "brown")) +
  theme_bw() + facet_wrap(~grp) + theme(legend.title=element_blank())

