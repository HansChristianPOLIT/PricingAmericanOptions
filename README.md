# Term Paper Repository - Monte Carlo Methods in Econometrics and Finance, Fall 2023

## Overview
This repository contains research on the pricing of American options, with a focus on overcoming the computational challenges associated with their early exercise feature. The research employs the regression-based Least-Squares Monte Carlo (LSM) method and the interpolation-based Dynamic Chebyshev method. Our study involves conducting numerical experiments based on the Black-Scholes model and the Merton Jump-Diffusion model.

## Key Findings
- **Convergence in Pricing American Options**: Both the LSM and Dynamic Chebyshev methods show convergence when used for pricing American options.
- **Performance Analysis**:
  - The Dynamic Chebyshev method, while accurate in pricing options, shows a slower performance compared to LSM.
  - For the LSM method, the incorporation of variance reduction techniques, especially the use of control variates sampled at exercise time, shows significant advantages over traditional sampling at expiry.
- **Model-Specific Considerations**:
  - Within the realm of the Merton Jump-Diffusion model, it is necessary to use an adjusted truncation domain when employing the Dynamic Chebyshev method.

## Contributors
- Hans Christian Jul Lehmann
- Nicholas Stampe Meier

## Dependencies
To run the notebooks in this repository, the following packages are required:
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `os` (for storing plots) 
- `time`