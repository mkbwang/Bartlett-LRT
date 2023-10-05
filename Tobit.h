//
// Created by wangmk on 21/09/23.
//
#include <iostream>
#include <cmath>
//#include <nlopt.h>
#include <armadillo>

using namespace std;
using namespace arma;

#ifndef TOBIT_H
#define TOBIT_H

#define SUCCESS 0;
#define FAIL 1;
#define STOPEARLY 2;


class model{
protected:
    vec Y; // response variable, dimension N
    vec Delta; // indicator of nonzero counts, dimension N
    mat X; // covariate matrix (including intercept), dimension N*P
    double tolerance; // relative tolerance of llk to stop the NR/BFGS iterations
    size_t maxiter; // max iterations
    bool isnull; // if fitting a null model, the second parameter will never be updated and always fixed to zero
    size_t N; // number of individuals N
    size_t P; // number of effect sizes to estimate P
    vec params; // parameters to estimate, including rho(effect sizes) and phi(-log scale), dimension P+1
    double llk; // log likelihood, scalar

public:
    model(const vec& Y_input, const vec& delta_input, const mat&X_input,
          double tolerance = 1e-4, size_t maxiter=50):
            Y(Y_input), Delta(delta_input), X(X_input), N(X_input.n_rows), P(X_input.n_cols),
            tolerance(tolerance), maxiter(maxiter), isnull(false), params(vec(X_input.n_cols+1, fill::zeros)),
            llk(0){};
    virtual void reset(bool null=false) = 0; // reset the parameters
    // update some utility variables for calculation of likelihood and gradients
    virtual void update_utils() = 0;
    //calculate log likelihood
    virtual double calc_llk() = 0;
    //update derivatives
    virtual void update_deriv() = 0;
    //update hessian
    virtual int update_hessian() = 0; // check if negative hessian (information) is positive definite
    //update parameter
    virtual void update_param() = 0;
    //master function for fitting the models
    virtual int fit() = 0;
    //return estimated values
    vec return_param(){
        return params;
    };
    double return_llk(){
        return llk;
    };
    virtual ~model() = default; //destructor

};

class tobit_vanilla: public model{
protected:
    uvec fixed; // indicator of whether any parameter is fixed
    vec Z; // exp(phi)*Y - X^t rho, dimension N
    vec cumnorm_z; // cumulative distribution of standard normal for each individual z, dimension N
    vec exp_2z2; // exp(-0.5*z^2), dimension N
    vec deriv_z; // first derivative of llk over each individual z, dimension N
    vec deriv_2z; // second derivative of llk over each individual z, dimension N
    vec score; // derivative of llk over all the parameters parameter (score equation), dimension P+1
    mat hessian; // second derivative (hessian) of llk over each parameter, dimension (P+1)*(P+1)
    size_t iter_counter;
    int convergence_code; // an integer representing the convergence codes
public:
    tobit_vanilla(const vec& Y_input, const vec& delta_input, const mat&X_input,
                  double tolerance = 1e-4, size_t maxiter=50);
    void reset(bool null=false);
    void update_utils();
    double calc_llk();
    void update_deriv();
    int update_hessian();
    void update_param();
    int fit();
    int return_iterations();
};

class tobit_firth;




// tobit loglikelihood function
//double tobitllk_vanilla(unsigned ndim, const double* params, double* grad, void* input);
//double tobitllk_firth(unsigned ndim, const double* params, double* grad, void* input);
// tobit estimation
//tobitoutput estimation(void *input, bool null=false);

#endif //BARTLETT_LRT_TOBIT_H
