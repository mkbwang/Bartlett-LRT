#include <iostream>
#include <cmath>
#include <nlopt.h>
#include <armadillo>
#include "Tobit.h"

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



int main()
{
    string inputfile="/home/wangmk/UM/Research/MDAWG/Bartlett-LRT/example_data.txt";
    int nrows=countrows(inputfile);
    int ncols=countcols(inputfile);
    cout << "Number of rows: " << nrows << endl;
    cout << "Number of cols: " << ncols << endl;
}
