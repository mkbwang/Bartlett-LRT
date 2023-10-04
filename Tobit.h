//
// Created by wangmk on 21/09/23.
//
#include <iostream>
#include <cmath>
//#include <nlopt.h>
#include <armadillo>

using namespace std;
using namespace arma;

#ifndef BARTLETT_LRT_TOBIT_H
#define BARTLETT_LRT_TOBIT_H

#define SUCCESS 0;
#define FAIL 1;
#define STOPEARLY 2;


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


struct tobit_vanilla{

    vec Y; // response variable, dimension N
    vec Delta; // indicator of nonzero counts, dimension N
    mat X; // covariate matrix (including intercept), dimension N*P
    double tolerance; // relative tolerance of llk to stop the NR iterations
    size_t maxiter; // max iterations
    bool isnull; // if fitting a null model, the second parameter will never be updated and always fixed to zero
    size_t N; // number of individuals N
    size_t P; // number of effect sizes to estimate P


    vec params; // parameters to estimate, including rho(effect sizes) and phi(-log scale), dimension P+1
    uvec fixed; // indicator of whether any parameter is fixed
    vec Z; // exp(phi)*Y - X^t rho, dimension N
    vec cumnorm_z; // cumulative distribution of standard normal for each individual z, dimension N
    vec exp_2z2; // exp(-0.5*z^2), dimension N
    double llk; // log likelihood, scalar
    vec deriv_z; // first derivative of llk over each individual z, dimension N
    vec deriv_2z; // second derivative of llk over each individual z, dimension N
    vec score; // derivative of llk over all the parameters parameter (score equation), dimension P+1
    //vec step_params; // store the most recent step size in the iterations for the parameters, dimension P+!
    //vec step_score; // store the most recent step size in the iterations for the score equations, dimension P+1
    mat hessian; // second derivative (hessian) of llk over each parameter, dimension (P+1)*(P+1)
    size_t iter_counter;
    int convergence_code; // an integer representing the convergence codes

    //constructor
    tobit_vanilla(const vec& Y_input, const vec& delta_input, const mat&X_input,
                  double tolerance = 1e-4, size_t maxiter=50);
    void reset(bool null=false); // reset the parameters
    // update the vectors related to z
    void update_utils();
    //calculate log likelihood
    double calc_llk();
    //update derivatives
    void update_deriv();
    //update hessian
    int update_hessian(); // check if negative hessian (information) is positive definite
    //update parameter
    void update_param();
    //master function
    int fit();
    //return estimated values
    vec return_param();
    double return_llk();

};


// tobit loglikelihood function
//double tobitllk_vanilla(unsigned ndim, const double* params, double* grad, void* input);
//double tobitllk_firth(unsigned ndim, const double* params, double* grad, void* input);
// tobit estimation
//tobitoutput estimation(void *input, bool null=false);

#endif //BARTLETT_LRT_TOBIT_H
