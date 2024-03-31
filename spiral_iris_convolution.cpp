#include <iris/iris.hpp>
#include <iris/iris_openmp.h>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#include <include/interface.hpp>
#include <include/convlib.hpp>
#pragma GCC diagnostic pop
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <typeinfo>
#include <string>
#include <cxxabi.h>
#include <complex>

static void buildInputBuffer ( double *host_X, std::vector<int> sizes )
{
    for ( int imm = 0; imm < sizes.at(0); imm++ ) {
        for ( int inn = 0; inn < sizes.at(1); inn++ ) {
            for ( int ikk = 0; ikk < sizes.at(2); ikk++ ) {
                int offset = (ikk + inn * sizes.at(2) + imm * sizes.at(1) * sizes.at(2)) * 2;
                host_X[offset + 0] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
                host_X[offset + 1] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
            }
        }
    }
    return;
}

int main(int argc, char** argv) {

  iris::Platform platform;
  platform.init(&argc, &argv, true);
  auto start = std::chrono::high_resolution_clock::now();
  int n,m,k;
  n = 24;
  m = 32;
  k = 40;

  std::vector<int> sizes{n*m*k, n*m*k, n*m*k*2, n, m, k};
  double *Y, *X, *sym;
  X = new double[n*m*k];
  Y = new double[n*m*k];
  sym = new double[n*m*k*2];
  for(int i = 0; i < n*m*k; i++) {
    X[i] = 1.0;
  }
    
  std::vector<void*> args{Y,X,sym};

  ConvProblem cp(args,sizes,"conv");
  
  cp.readKernels(); 
  cp.createGraph();
  cp.transform();

//   std::cout << "kernel execution time is " << cp.getTime() << std::endl;

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << duration.count() << std::endl;

  for(int i = 0; i < 10; i++) {
    std::cout << Y[i] << std::endl;
  }

//   printf("%s\n", b);

  platform.finalize();

  return 0;
}