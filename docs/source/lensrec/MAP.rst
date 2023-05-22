==================================
iterative lensing reconstruction
==================================

This section describes the idea behind iterative lensing reconstruction and its most important concept.


likelihood
---------------

We start with the following likelihood `lh` for which we search the maximum-a-posterioi (MAP) point,

-2 ln(p(X|a)) = X cov^(-1) X + ln(det(cov))

where `p` is the data model, `X` is the data, `a` is the deflection field, and `cov` is the data-covariance.
We find the MAP point by following the gradient of the likelihood, which is done iteratively.

We choose a starting point for `a`, calculate the gradient of the likelihood and update `a` with it,
and repeat this until `a` has converged.

This gradient splits naturally into three pieces.
A term quadratic in the data `gqd`, a so called mean-field, and a prior term. 



Wiener-filter
---------------

Calculating `gqd` includes a Wiener-filtering of the data,

Ewf = (s/(s+N)) E,

where `s` is the expected signal, and `N` is the noise.
For iterative lensing reconstruction, `s` is given by the unlensed CMB E field `Eunl`, times the lensing operation `Lambda`,

s = Lambda Eunl