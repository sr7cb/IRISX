#include <iris/iris.hpp>
#include <iris/iris_openmp.h>
#include <vector>
#include <include/interface.hpp>
#include <include/nttlib.hpp>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <typeinfo>
#include <string>
#include <cxxabi.h>
#include <complex>
#include <fstream>


static constexpr auto test = R"(extern "C" __global__ void ker_mddft_spiral0(double *X, double *P1, double *D3){
  if(blockIdx.x*blockDim.x + threadIdx.x == 0)
    printf("Hello from ker_mddft_spiral0\n");
}

extern "C" __global__ void ker_mddft_spiral1(double *P2, double *P1, double *D3){
  if(blockIdx.x*blockDim.x + threadIdx.x == 1)
    printf("Hello from ker_mddft_spiral0\n");
}

extern "C" __global__ void ker_mddft_spiral2(double *Y, double *P2, double *D3){
  if(blockIdx.x*blockDim.x + threadIdx.x == 2)
    printf("Hello from ker_mddft_spiral0\n");
}
)";

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
  int n,m;
  n = 2;
  m = 64;

  int iter = 2;
  std::vector<int> sizes{m*n, m*n, 0};

  // buildInputBuffer(X, sizes);
  std::vector<void*> args;
  NTTProblem ntt("ntt");
  for(int i = 0; i < iter; i++){
    uint32_t *Y, *X, *sym;
    X = new uint32_t[n*m];
    Y = new uint32_t[n*m];
    sym = new uint32_t[n*m];
    for(int ii = 0; ii < n*m; ii++) {
      X[ii] = 1.0;
    }
    args.push_back(Y);
    args.push_back(X);
    args.push_back(sym);
    ntt.setArgs(args);
    ntt.setSizes(sizes);
    ntt.readKernels();
    ntt.createGraph();
    ntt.transform();

    if(std::stoi(argv[1]) == 1){
      std::string filePath =  getIRISX() + "/kernel.cu";

      std::ofstream outfile(filePath, std::ios::out | std::ios::trunc);

      if (outfile.is_open()) {
          outfile << test;
          outfile.close();
          std::cout << "Raw string written to file successfully." << std::endl;
      } else {
          std::cerr << "Error opening file." << std::endl;
      }
      std::string command;
      command.append("nvcc -Xcudafe --diag_suppress=declared_but_not_referenced -ptx " + getIRISX() + "/kernel.cu");
      system(command.c_str());
      std::cout << "rewrite kernel.ptx" << std::endl;
    } else if(std::stoi(argv[1]) == 2) {
      system("rm kernel.ptx kernel.cu");
      std::cout << "removed kernels" << std::endl;
    }
  }

    for(int j = 0; j < iter; j++) {
    for(int i = 0; i < 10; i++) {
      std::cout << ((double*)args.at(j*3))[i] << " ";
    }
    std::cout << std::endl;
  }

  platform.finalize();

  return 0;
}