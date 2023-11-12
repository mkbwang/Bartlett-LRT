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
    vec Nmt(nrows); // numerator
    vec Denom(nrows); // denominator
    mat X(nrows, ncols-3, fill::ones);

    unsigned int rowid = 0;
    string line;
    ifstream newfile(inputfile);
    while(getline(newfile, line)){
        stringstream s;
        s << line;
        s >> Y(rowid) >> Delta(rowid) >> Nmt(rowid) >> Denom(rowid);
        for (int j=1; j<ncols-3; j++){
            s >> X(rowid, j);
        }
        s.str(string());
        rowid++;
    }
    tobit_firth tbmodel_obj{Nmt, Denom, Delta, X, 1e-5, 50};
    return tbmodel_obj;

}


int main()
{

    string inputfile="/home/wangmk/UM/Research/MDAWG/Bartlett-LRT/example_data.txt";
    // string outputfile="/home/wangmk/UM/Research/MDAWG/Bartlett-LRT/teststats_old.txt";
    string outputfile="/home/wangmk/UM/Research/MDAWG/Bartlett-LRT/teststats_new.txt";
    int nrows=countrows(inputfile);
    int ncols=countcols(inputfile);

    tobit_firth tbmodel = readdata(nrows, ncols, inputfile);
    vec teststat(2001, fill::zeros);
    Col<int> iterations(2001, fill::zeros);
    vec prevalences(2001, fill::zeros);
    Col<int> fail(2001, fill::zeros);

    // first fit the true data
    cout << "Full Model: " << endl;
    int convergence = tbmodel.fit();
    cout << "Convergence code: " << convergence << endl;
    int n_iter_full = tbmodel.return_iterations();
    cout << "Number of iterations: " << n_iter_full << endl;
    vec estimates = tbmodel.return_param();
    size_t length=estimates.n_elem;
    vec beta = estimates.head(length-1) / estimates(length-1);
    cout << "Estimated Effect Sizes: " << endl;
    beta.as_row().print();
    double sigma = 1 / estimates(length-1);
    cout << "Estimated scale: " << sigma << endl;
    double full_llk = tbmodel.return_llk();
    cout << "Log Likelihood: " << full_llk << endl;

    tbmodel.reset(true);
    cout << "Null Model: " << endl;
    convergence = tbmodel.fit();
    cout << "Convergence code: " << convergence << endl;
    int n_iter_null = tbmodel.return_iterations();
    cout << "Number of iterations: " << n_iter_null << endl;
    estimates = tbmodel.return_param();
    length = estimates.n_elem;
    beta = estimates.head(length-1) / estimates(length-1);
    cout << "Estimated Effect Sizes: " << endl;
    beta.as_row().print();
    sigma = 1 / estimates(length-1);
    cout << "Estimated scale: " << sigma << endl;
    double reduced_llk = tbmodel.return_llk();
    cout << "Log Likelihood: " << reduced_llk << endl;


    teststat(0) = 2*(full_llk - reduced_llk);
    iterations(0) = n_iter_full;
    prevalences(0) = tbmodel.return_prevalences();
    fail(0) = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (unsigned int j=1; j<2001; j++){
        try{
            tbmodel.reorder(true); // bootstrap the responses
            tbmodel.reset(false, {1});
            prevalences(j) = tbmodel.return_prevalences();
            convergence = tbmodel.fit();
            if (convergence != 0){
                throw std::runtime_error("Full model didn't converge");
            }
            n_iter_full = tbmodel.return_iterations();
            full_llk = tbmodel.return_llk();
            iterations(j) = n_iter_full;

            tbmodel.reset(true, {1});
            convergence = tbmodel.fit();
            if (convergence != 0){
                throw std::runtime_error("Reduced model didn't converge");
            }
            reduced_llk = tbmodel.return_llk();
            teststat(j) = 2*(full_llk - reduced_llk);

        } catch(std::runtime_error& err){
            fail(j)=1;
        }

    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Bootstrap Time (sec) = " <<  \
        (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  << \
        std::endl;

    ofstream output(outputfile);
    output << "Fail" << " " << "Prevalence" << " " << "Iterations" << " " << "Teststat" << endl;
    for(int j = 0; j<2001; j ++){
        output << fail(j) << " " << prevalences(j) << " " << iterations(j) << " " << teststat(j) << endl ;
    }
    output.close();

    return 0;

}
