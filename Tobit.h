//
// Created by wangmk on 21/09/23.
//
#include <iostream>
#include <cmath>
#include <nlopt.h>
#include <armadillo>

using namespace std;
using namespace arma;

#ifndef BARTLETT_LRT_TOBIT_H
#define BARTLETT_LRT_TOBIT_H

//TODO: constructor
struct tobitinput{
    vec Y; // log count ratio
    vec Delta; // censorship
    mat X; // covariate matrix
    size_t n_sample;
    double stepsize; // gradient ascent step size
    tobitinput() = default;
    tobitinput(vec &dependent_vec, vec &censor_vec, mat &covar_mat):
        Y(dependent_vec), Delta(censor_vec), X(covar_mat), n_sample(covar_mat.n_rows), stepsize(0.1/covar_mat.n_rows){}
};


//TODO: constructor
struct tobitoutput{
    vec params; // estimated parameter
    double llk; // log likelihood
    int nevals; // optimization status
    tobitoutput(vec &estimate, double maxllk, int nevals):
        params(estimate), llk(maxllk), nevals(nevals){}
};

// tobit loglikelihood function
double tobitllk_vanilla(unsigned ndim, const double* params, double* grad, void* input);
double tobitllk_firth(unsigned ndim, const double* params, double* grad, void* input);
// tobit estimation
tobitoutput estimation(void *input, bool null=false);

#endif //BARTLETT_LRT_TOBIT_H
