#include <iostream>
#include <cmath>
#include <armadillo>
#include "Tobit.h"
#include <chrono>


using namespace std;
using namespace arma;


int countrows(string filename){
    int nline = 0;
    string line;
    ifstream newfile(filename);
    while (getline(newfile, line)){
        ++nline;
    }
    newfile.close();
    return nline;
}

int countcols(string filename){
    int ncol = 0;
    string line;
    ifstream newfile(filename);
    getline(newfile, line);
    stringstream s;
    s << line;
    double value;
    while(s >> value) ncol++;
    newfile.close();
    return ncol;
}

tobit_firth readdata(unsigned int nrows, unsigned int ncols, string inputfile){

    vec Y(nrows);
    vec Delta(nrows);
    mat X(nrows, ncols-1, fill::ones);

    unsigned int rowid = 0;
    string line;
    ifstream newfile(inputfile);
    while(getline(newfile, line)){
        stringstream s;
        s << line;
        s >> Y(rowid) >> Delta(rowid);
        for (int j=1; j<ncols-1; j++){
            s >> X(rowid, j);
        }
        s.str(string());
        rowid++;
    }
    tobit_firth tbmodel_obj{Y, Delta, X, 1e-5, 50};
    return tbmodel_obj;

}


int main()
{

    string inputfile="/home/wangmk/UM/Research/MDAWG/Bartlett-LRT/example_data.txt";
//    string outputfile="/home/wangmk/UM/Research/MDAWG/Bartlett-LRT/teststats.txt";
    int nrows=countrows(inputfile);
    int ncols=countcols(inputfile);

    tobit_firth tbmodel = readdata(nrows, ncols, inputfile);
//    vec teststat(501, fill::zeros);
//    Col<int> iterations(501, fill::zeros);
//    vec prevalences(501, fill::zeros);
//    Col<int> fail(501, fill::zeros);

    // first fit the true data
    cout << "Full Model: " << endl;
    int convergence = tbmodel.fit();
    cout << "Convergence code: " << convergence << endl;
    cout << "Number of iterations: " << tbmodel.return_iterations() << endl;
    vec estimates = tbmodel.return_param();
    size_t length=estimates.n_elem;
    vec beta = estimates.head(length-1) / estimates(length-1);
    cout << "Estimated Effect Sizes: " << endl;
    beta.as_row().print();
    double sigma = 1 / estimates(length-1);
    cout << "Estimated scale: " << sigma << endl;
    cout << "Log Likelihood: " << tbmodel.return_llk() << endl;

    tbmodel.reset(true);
    cout << "Null Model: " << endl;
    convergence = tbmodel.fit();
    cout << "Convergence code: " << convergence << endl;
    cout << "Number of iterations: " << tbmodel.return_iterations() << endl;
    estimates = tbmodel.return_param();
    length = estimates.n_elem;
    beta = estimates.head(length-1) / estimates(length-1);
    cout << "Estimated Effect Sizes: " << endl;
    beta.as_row().print();
    sigma = 1 / estimates(length-1);
    cout << "Estimated scale: " << sigma << endl;
    cout << "Log Likelihood: " << tbmodel.return_llk() << endl;

//    tobitoutput full_estimates = estimation(&inputdata, false);
//    cout << "Full Model Estimates: " << endl;
//    full_estimates.params.as_row().print();
//    cout << "Full loglik: " << full_estimates.llk << endl;
//    cout << "Number of evaluations: " << full_estimates.nevals << endl;
//    tobitoutput null_estimates = estimation(&inputdata, true);
//    cout << "Null Model Estimates: " << endl;
//    null_estimates.params.as_row().print();
//    cout << "Reduced loglik: " <<  null_estimates.llk << endl;

//    teststat(0) = 2*(full_estimates.llk - null_estimates.llk);
//    iterations(0) = full_estimates.nevals;
//    prevalences(0) = accu(inputdata.Delta)/nrows;
//    fail(0) = 0;
//
//    tobitinput boot_input = inputdata;
//    //bootstrap
//    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//
//    for (unsigned int j=1; j<501; j++){
//        try{
//            arma_rng::set_seed(j);
//            // Col<uword> selected_indices = randperm(nrows, nrows);
//            Col<uword> selected_indices = randi<uvec>(nrows, distr_param(0, nrows-1));
//            boot_input.Y = inputdata.Y.elem(selected_indices);
//            boot_input.Delta = inputdata.Delta.elem(selected_indices);
//            prevalences(j) = accu(boot_input.Delta)/nrows;
//            tobitoutput boot_full_estimates = estimation(&boot_input, false);
//            tobitoutput boot_null_estimates = estimation(&boot_input, true);
//            teststat(j) = 2*(boot_full_estimates.llk - boot_null_estimates.llk);
//            iterations(j) = boot_full_estimates.nevals;
//        } catch(std::overflow_error& err){
//            fail(j)=1;
//        }
//
//    }
//
//    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//    std::cout << "Bootstrap Time (sec) = " <<  \
//        (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  << \
//        std::endl;
//
//    ofstream output(outputfile);
//    output << "Fail" << " " << "Prevalence" << " " << "Iterations" << " " << "Teststat" << endl;
//    for(int j = 0; j<501; j ++){
//        output << fail(j) << " " << prevalences(j) << " " << iterations(j) << " " << teststat(j) << endl ;
//    }
//    output.close();

    return 0;

}
