# Bayesian Additive Regression Trees for Multivariate skewed responses

This repository contains code to implement the methodology described in
the paper “Bayesian Additive Regression Trees for Multivariate Skewed
Responses”, by Um, Linero, Sinha, and Bandyopadhyay (2022+, Statistics
in Medicine)

This package uses the primary functions from
[`SoftBART`](https://github.com/theodds/SoftBART) to incorporate the
SoftBART model as a component.

## Installation

The packages can be installed with the `devtools` package:

    library(devtools) 
    devtools::install_github(repo='Seungha-Um/skewBART') 
    # or 
    devtools::install_github(repo='Seungha-Um/skewBART',build_vignettes = TRUE) 
