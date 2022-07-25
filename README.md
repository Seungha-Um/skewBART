# Bayesian Additive Regression Trees for Multivariate skewed responses

This repository contains code to implement the methodology described in
the paper “Bayesian Additive Regression Trees for Multivariate Skewed
Responses”, by Um, Linero, Sinha and Bandyopadhyay (2022+, Statistics in
Medicine)

This package uses the primary functions from
[`SoftBART`](https://github.com/theodds/SoftBART) to incorporate the
SoftBART model as a component.

## Installation

The packages can be installed with the `devtools` package:

    library(devtools) 
    devtools::install_github(repo='Seungha-Um/skewBART') 

The package with the vignette’s included can be installed with

    devtools::install_github(repo='Seungha-Um/skewBART',build_vignettes = TRUE) 

and then accessed by running `browseVignettes("skewBART")` (to reproduce
our results, one will need to increase the number of MCMC samples).
Alternatively, vignettes are available at
[Simulation](https://rpubs.com/sheom0808/926961) and [GAAD
dataset](https://rpubs.com/sheom0808/926959).

One of the vignette’s replicates our analysis of the GAAD dataset, a
subset of which has been included, while the other illustrates our
methods on simulated data.
