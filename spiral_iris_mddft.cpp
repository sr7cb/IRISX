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
  // iris::Platform platform;
  // platform.init(&argc, &argv, true);
  int n,m,k;
  n = 8;
  m = 8;
  k = 8;

  std::vector<int> sizes{n,m,k};
  double *Y, *X, *sym;
  X = new double[n*m*k*2];
  Y = new double[n*m*k*2];
  sym = new double[n*m*k*2];
  for(int i = 0; i < n*m*k*2; i++) {
    X[i] = 1.0;
  }
  // buildInputBuffer(X, sizes);
  std::vector<void*> args{Y,X,sym};
  MDDFTProblem mdp(args,sizes,"mddft");

  mdp.transform();

  for(int i = 0; i < 10; i++) {
    std::cout << Y[i] << std::endl;
  }

//   printf("%s\n", b);

  // platform.finalize();

  return 0;
}