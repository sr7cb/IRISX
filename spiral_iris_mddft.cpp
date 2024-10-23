#include <iris/iris.hpp>
#include <iris/iris_openmp.h>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#include <include/interface.hpp>
#include <include/mddftlib.hpp>
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
  // getIRISARCH();
  // std::cout << getIRISARCH() << std::endl;
  iris::Platform platform;
  platform.init(&argc, &argv, true);
  auto start = std::chrono::high_resolution_clock::now();
  int n,m,k;
  n = 64;
  m = 64;
  k = 64;

  int iter = 5;
  std::vector<int> sizes{n*m*k*2, n*m*k*2, n*m*k*2, n, m, k};

  // buildInputBuffer(X, sizes);
  std::vector<void*> args;
MDDFTProblem mdp("mddft");
for(int i = 0; i < iter; i++){
  double *Y, *X, *sym;
  X = new double[n*m*k*2];
  Y = new double[n*m*k*2];
  sym = new double[n*m*k*2];
  for(int ii = 0; ii < n*m*k*2; ii++) {
    X[ii] = 1.0;
  }
  args.push_back(Y);
  args.push_back(X);
  args.push_back(sym);
  mdp.setArgs(args);
  mdp.setSizes(sizes);
  mdp.readKernels();
  mdp.createGraph();
}
  mdp.transform();
    for(int j = 0; j < iter; j++) {
    for(int i = 0; i < 10; i++) {
      std::cout << ((double*)args.at(j*3))[i] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "checking first round clearing args" << std::endl;
  for(int i = 0; i < args.size(); i++) {
    std::cout << (double*)(args.at(i)) << std::endl;
    delete [](double*)(args.at(i));
  }

  mdp.resetInput();
  args.clear();
  std::cout << args.size() << std::endl;
for(int i = 0; i < iter; i++){
  double *Y, *X, *sym;
  X = new double[n*m*k*2];
  Y = new double[n*m*k*2];
  sym = new double[n*m*k*2];
  for(int ii = 0; ii < n*m*k*2; ii++) {
    X[ii] = 1.0;
  }
  args.push_back(Y);
  args.push_back(X);
  args.push_back(sym);
  mdp.setArgs(args);
  mdp.setSizes(sizes);
  // mdp.readKernels();
  // mdp.createGraph();
}
  std::cout << args.size() << std::endl;
  mdp.transform();
  std::cout << "fixed graph new inputs" << std::endl;
  for(int j = 0; j < iter; j++) {
    for(int i = 0; i < 10; i++) {
      std::cout << ((double*)args.at(j*3))[i] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "kernel execution time is " << mdp.getTime() << std::endl;

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << duration.count() << std::endl;
  for(int j = 0; j < iter; j++) {
    for(int i = 0; i < 10; i++) {
      std::cout << ((double*)args.at(j*3))[i] << " ";
    }
    std::cout << std::endl;
  }

//   printf("%s\n", b);

  platform.finalize();

  return 0;
}