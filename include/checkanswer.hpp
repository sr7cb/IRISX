#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <limits>
#include <stdio.h>
#include <typeinfo>
#include <cxxabi.h>

void checkDataHolders(double *irisx_out, int boxSize, int level) {
  std::string file_name;
  if(boxSize == 32){
    file_name = "32x3D/32boxanswer"+std::to_string(level)+".txt";
  } else if(boxSize == 64) {
    file_name = "64x3D/64boxanswer"+std::to_string(level)+".txt";
  }
  std::ifstream myfile;
  myfile.open(file_name);
  double *X = new double[boxSize*boxSize*boxSize*5];
  int x = 0;
  std::string mystring;
  if(myfile.is_open()) {
    // std::cout << "file: " << file_name << " opened for input read" << std::endl;
      while(myfile.good() && x != boxSize*boxSize*boxSize*5) {
          myfile >> mystring;
          X[x] = (std::stod(mystring));
          x++;
      }
  }

  bool correct = true;
  double maxdelta = 0.0;
  for(int i = 0; i < boxSize*boxSize*boxSize*5; i++) {
    bool elem_correct = ( (fabs(X[i] - irisx_out[i]) < 1e-7));
    correct &= elem_correct;
    if(!correct) {
      std::cout << "answer not correct exiting" << std::endl;
      exit(-1);
    }
  }
  std::cout << "correct" << std::endl;
  delete []X;
}