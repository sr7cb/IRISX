#include <iris/iris.hpp>
#include <iris/iris_openmp.h>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#include <include/interface.hpp>
#include <include/protoeulerlib.hpp>
#pragma GCC diagnostic pop
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <typeinfo>
#include <string>
#include <cxxabi.h>

void checkDataHolders(double * spiral, double * proto, int dim_r, int dim_c) {
  int * a = new int[4];
  bool correct = true;
  double maxdelta = 0.0;
  for(int i = 0; i < 4; i++) {
      a[i] = 0;
      for(int j = 0; j < dim_r; j++) {
          for (int k = 0; k < dim_c; k++) {
              double s = spiral[i*(dim_r*dim_c) + ((j*dim_c+k))];
              double p = proto[i*(dim_r*dim_c) + ((j*dim_c+k))];
              bool elem_correct = ( (fabs(s - p) < 1e-5));
              maxdelta = maxdelta < (double)(fabs(s -p)) ? (double)(fabs(s -p)) : maxdelta;
              correct &= elem_correct;
              if(elem_correct == false) {
                  std::cout << i*(dim_r*dim_c) + ((j*dim_c+k)) << " ";
                  std::cout << s << " " << p << std::endl;
                  // exit(0);
                  a[i]++;
              }
          }
      }
  }
  printf ( "Correct: %s\tMax delta = %E\n", (correct ? "True" : "False"), maxdelta );
  if(!correct) {
    for(int i = 0; i < 4; i++) {
        std::cout << "Patch " << i << "had this many errors " << a[i] << std::endl;
    }
  }
}

int main(int argc, char** argv) {
  // getIRISARCH();
  // std::cout << getIRISARCH() << std::endl;
  iris::Platform platform;
  platform.init(&argc, &argv, true);
  // auto start = std::chrono::high_resolution_clock::now();
  std::string mystring;
  int n,m,k;
  n = 136;
  m = 136;
  k = 4;

  std::vector<int> sizes{n*m*k, n*m*k, 1, 1, 1, n, m};
  double *Y, *X;
  X = new double[n*m*k];
  Y = new double[n*m*k];
  std::ifstream myinputfile;
  std::string inputfilename = "include/input_" + std::to_string(n) + "_" + std::to_string(m) + ".txt"; 
  std::cout<< inputfilename;
  myinputfile.open(inputfilename);
  int x = 0;
  if(myinputfile.is_open()) {
    std::cout << "file: " << inputfilename << " opened for input read" << std::endl;
      while(myinputfile.good() && x != n*m*k) {
          myinputfile >> mystring;
          X[x] = (std::stod(mystring));
          x++;
      }
  }

  // for(int i = 0; i < n*m*k*2; i++) {
  //   X[i] = 1.0;
  // }

  double gamma, a_scale, dx;
  gamma = 1.4;
  a_scale = 0.00390625;
  dx = 0.015625;
  // buildInputBuffer(X, sizes);
    std::vector<void*> args{Y,X, &gamma, &a_scale, &dx};
    ProtoProblem pp(args,sizes,"level_euler");
  for(int i = 0; i < 1; i++) {
  std::cout << pp.gpuTime << std::endl;
  pp.transform();
  std::cout << pp.gpuTime << std::endl;
  std::cout << "kernel execution time is " << pp.getTime() << std::endl;
  }
  double * correct_ans = new double[(n-8)*(m-8)*k];
  std::ifstream mycorrectoutputfile; 
  std::string outputfilename = "include/correctresult_" + std::to_string(n) + "_" + std::to_string(m) + ".txt"; 
  mycorrectoutputfile.open(outputfilename);
    x = 0;
  if(mycorrectoutputfile.is_open()) {
      std::cout << "file: " << outputfilename << " opened for output check" << std::endl;
      while(mycorrectoutputfile.good() && x != (n-8)*(m-8)*k) {
        mycorrectoutputfile >> mystring;
        correct_ans[x] = (std::stod(mystring));
        x++;
      }
  }
  for(int i = 0; i < 10; i++)
    std::cout << Y[i] << std::endl;
  checkDataHolders(Y, correct_ans, n-8, m-8);

  platform.finalize();
  return 0;
}