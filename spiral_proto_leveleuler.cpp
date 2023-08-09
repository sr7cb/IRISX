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

int main(int argc, char** argv) {
  // getIRISARCH();
  // std::cout << getIRISARCH() << std::endl;
  // iris::Platform platform;
  // platform.init(&argc, &argv, true);
  // auto start = std::chrono::high_resolution_clock::now();
  int n,m,k;
  n = 40;
  m = 40;
  k = 4;

  std::vector<int> sizes{n,m,k};
  double *Y, *X;
  X = new double[n*m*k*2];
  Y = new double[n*m*k*2];
  for(int i = 0; i < n*m*k*2; i++) {
    X[i] = 1.0;
  }

  double gamma, a_scale, dx;
  gamma = 1;
  a_scale = 1;
  dx = 1;
  // buildInputBuffer(X, sizes);
  std::vector<void*> args{Y,X, &gamma, &a_scale, &dx};

  ProtoProblem pp(args,sizes,"level_euler");
  
  pp.transform();

  std::cout << "kernel execution time is " << pp.getTime() << std::endl;

  // auto stop = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  // std::cout << duration.count() << std::endl;
  // for(int i = 0; i < 10; i++) {
  //   std::cout << Y[i] << std::endl;
  // }

//   printf("%s\n", b);

  // platform.finalize();

  return 0;
}