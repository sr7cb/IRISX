#ifndef FFTX_MDDFT_CPUBACKEND_HEADER
#define FFTX_MDDFT_CPUBACKEND_HEADER

//  Copyright (c) 2018-2022, Carnegie Mellon University
//  See LICENSE for details

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <cstdio>
#include <string>
#include <cstdio>      // perror
#include <unistd.h>    // dup2
#include <sys/types.h> // rest for open/close
#include <sys/utsname.h> // check machine name
#include <sys/stat.h>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <array>
#include <dlfcn.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <cstring>
#include <chrono>
#include <regex>
#pragma once

#define DEBUGOUT 0

// std::string getIRISARCH() {
//     const char * tmp2 = std::getenv("IRIS_ARCHS");
//     std::string tmp(tmp2 ? tmp2 : "");
//     if (tmp.empty()) {
//         std::cout << "[ERROR] No such variable found, please set IRIS_ARCHS env variable" << std::endl;
//         exit(-1);
//     }
//     return tmp;
// }

std::string getSPIRALHOME() {
    const char * tmp2 = std::getenv("SPIRAL_HOME");//required >8.3.1
    std::string tmp(tmp2 ? tmp2 : "");
    if (tmp.empty()) {
        std::cout << "[ERROR] No such variable found, please download and set SPIRAL_HOME env variable" << std::endl;
        exit(-1);
    }    
    return tmp;
}

std::string getIRIS() {
    const char * tmp2 = std::getenv("IRIS");//required >8.3.1
    std::string tmp(tmp2 ? tmp2 : "");
    if (tmp.empty()) {
        std::cout << "[ERROR] No such variable found, please download and set SPIRAL_HOME env variable" << std::endl;
        exit(-1);
    }      
    return tmp;
}

std::string getIRISARCH();

int redirect_input(int);
void restore_input(int);

class Executor {
    private:
        int x;
        enum string_code {
            zero,
            one,
            two,
            constant,
            pointer_int,
            pointer_float,
            pointer_double,
            mone
        };
        std::vector<void*> kernelargs;
        std::vector<std::tuple<std::string, int, std::string>> device_names;
        // std::string kernel_name;
        std::vector<std::string> kernel_names;
        std::vector<int> kernel_params;
        std::string kernel_preamble;
        std::string kernels;
        std::vector<std::vector<int>> values;
        //std::vector<std::string> kernels;
        std::vector<std::tuple<std::string, int, std::string>> in_params;
        std::vector<void*> params; 
        std::vector<void *> data;
        std::vector<int> data_lengths;
        void * shared_lib;
        float CPUTime;
    public:
        string_code hashit(std::string const& inString);
        float initAndLaunch(std::vector<void*>& args, std::vector<int> sizes, std::string name);
        void execute(std::string file_name, std::string arch);
        float getKernelTime();
        void parseDataStructure(std::string input);
        //void returnData(std::vector<fftx::array_t<3,std::complex<double>>> &out1);
};

Executor::string_code Executor::hashit(std::string const& inString) {
    if(inString == "int") return zero;
    if(inString == "float") return one;
    if(inString == "double") return two;
    if(inString == "constant") return constant;
    if(inString == "pointer_int") return pointer_int;
    if(inString == "pointer_float") return pointer_float;
    if(inString == "pointer_double") return pointer_double;
    return mone;
}

void Executor::parseDataStructure(std::string input) {
    std::istringstream stream(input);
    char delim = ' ';
    std::string line;
    std::string b = "------------------";
    while(std::getline(stream, line)){
        if(line.find("JIT BEGIN") != std::string::npos)
            break;
    }
    while(std::getline(stream,line)) {
        if(line == b) {
            break;
        }
        std::istringstream ss(line);
        std::string s;
        //int counter = 0;
        std::vector<std::string> words;
        while(std::getline(ss,s,delim)) {
            words.push_back(s);
        }
        int test = atoi(words.at(0).c_str());
        switch(test) {
            case 0:
                device_names.push_back(std::make_tuple(words.at(1), atoi(words.at(2).c_str()), words.at(3)));
                break;
            case 1:
                in_params.push_back(std::make_tuple(words.at(1), atoi(words.at(2).c_str()), words.at(3)));
                break;
            case 2:
                kernel_names.push_back(words.at(1));
                kernel_params.push_back(atoi(words.at(2).c_str()));
                kernel_params.push_back(atoi(words.at(3).c_str()));
                kernel_params.push_back(atoi(words.at(4).c_str()));
                kernel_params.push_back(atoi(words.at(5).c_str()));
                kernel_params.push_back(atoi(words.at(6).c_str()));
                kernel_params.push_back(atoi(words.at(7).c_str()));
                break;
            case 3:
                int loc = atoi(words.at(1).c_str());
                int size = atoi(words.at(2).c_str());
                int dt = atoi(words.at(3).c_str());
                std::cout << loc << ":" << dt << std::endl;
                //convert this to a string because spiral prints string type
                switch(dt) {
                    case 0: //int
                    {
                        // if(words.size() < 5) {
                        //     int * data1 = new int[size];
                        //     memset(data1, 0, size * sizeof(int));
                        //     data.push_back(data1);
                        // }
                        // else {
                        //     int * data1 = new int[size];
                        //     for(int i = 4; i < words.size(); i++) {
                        //         data1[i-4] = atoi(words.at(i).c_str());
                        //     }
                        //     data.push_back(data1);
                        // }
                        break;
                    }
                    case 1: //float
                    {
                        // if(words.size() < 5) {
                        //     float * data1 = new float[size];
                        //     memset(data1, 0, size * sizeof(float));
                        //     data.push_back(data1);
                        // }
                        // else {
                        //     float * data1 = new float[size];
                        //     for(int i = 4; i < words.size(); i++) {
                        //         data1[i-4] = std::stof(words.at(i));
                        //     }
                        //     data.push_back(data1);
                        // }
                        break;
                    }
                    case 2: //double
                    {
                        std::cout << "This is the words size double\n" << words.size() << std::endl;
                        if(words.size() < 5) {
                            double * data1 = new double[size];
                            // memset(data1, 0, size * sizeof(double));
                            data.push_back(data1);
                            data_lengths.push_back(size); 
                        }
                        else {
                            double * data1 = new double[size];
                            for(int i = 4; i < words.size(); i++) {
                                data1[i-4] = std::stod(words.at(i));
                            }
                            data.push_back(data1);
                            data_lengths.push_back(words.size()-4); 
                        }
                        break;
                    }
                    case 3: //constant
                    {
                        std::cout << "This is the words size constant\n" << words.size() << std::endl;
                        if(words.size() < 5) {
                            double * data1 = new double[size];
                            // memset(data1, 0, size * sizeof(double));
                            data.push_back(data1);
                            data_lengths.push_back(size); 
                        }
                        else {
                            double * data1 = new double[words.size()-4];
                            for(int i = 4; i < words.size(); i++) {
                                data1[i-4] = std::stod(words.at(i));
                            }
                            data.push_back(data1);
                            data_lengths.push_back(words.size()-4);  
                        }
                        break;
                    }
                }
                break;
        }
    }
    while(std::getline(stream, line)) {
        kernels += line;
        kernels += "\n";
    }
    if ( DEBUGOUT ) std::cout << "parsed input\n";

}


float Executor::initAndLaunch(std::vector<void*>& args, std::vector<int> sizes, std::string name) {
    iris::Platform platform;
    platform.init(NULL, NULL, true);
    size_t size = sizes.at(0) * sizes.at(1) * sizes.at(2);
    if ( DEBUGOUT) {
        std::cout << "In init and launch" << std::endl;
        std::cout << "size " << size << std::endl;
        std::cout << "start\n";
        std::cout << device_names.size() << std::endl;
        std::cout << kernel_names.size() << std::endl;
        std::cout << "end\n";
        for(int i = 0; i < device_names.size(); i++) {
                std::cout << std::get<0>(device_names[i]) << std::endl;
            }
        for(int i = 0; i < kernel_names.size(); i++) {
            std::cout << kernel_names[i] << std::endl;
        }
        std::cout << "data size " << data.size() << std::endl;
        std::cout << "data_lengths size " << data_lengths.size() << std::endl;
        // for(int i = 0; i < data.size(); i++) {
        //     std::cout << "size of list " << data_lengths.at(i) << std::endl;
        //     for(int j = 0; j < data_lengths.at(i); j++){
        //        std::cout << ((double*)data.at(i))[j] << std::endl;
        //     }
        // }
        // exit(0);
    }
    iris_mem mem_X;
    iris_mem mem_Y;
    iris_mem mem_sym;
#if 0
    iris_mem_create(size * sizeof(double) * 2, &mem_X);
    iris_mem_create(size * sizeof(double) * 2, &mem_Y);
    iris_mem_create(size * sizeof(double) * 2, &mem_sym);
#else
    iris_data_mem_create(&mem_Y, args.at(0), size * sizeof(double) * 2);
    iris_data_mem_create(&mem_X, args.at(1), size * sizeof(double) * 2);
    iris_data_mem_create(&mem_sym, args.at(2), size * sizeof(double) * 2);
#endif

    std::vector<void*> params = { &mem_Y, &mem_X, &mem_sym};
    std::vector<int> params_info = { iris_w, iris_r, iris_r};
    // std::vector<void*> temps;
    int pointers = 0;
    for(int i = 0; i < device_names.size(); i++) {
        std::string test = std::get<2>(device_names[i]);
        std::cout << std::get<0>(device_names[i]) << ":" << test << std::endl;
        switch(hashit(test)) {
            case constant:
            {
                // temps.push_back(data.at(i));
                iris_mem * mem_p = new iris_mem;

#if 0
                iris_mem_create(data_lengths.at(i) * sizeof(double), mem_p);
#else
    		iris_data_mem_create(mem_p, data.at(i), 
				std::get<1>(device_names.at(i)) * sizeof(double));
#endif


                params.push_back(mem_p);
                params_info.push_back(iris_rw);
                pointers++;
                break;
            }
            case pointer_int:
            {
                iris_mem * mem_p = new iris_mem;
                //iris_mem_create(std::get<1>(device_names.at(i)) * sizeof(int), mem_p);
                params.push_back(mem_p);
                params_info.push_back(iris_rw);
                pointers++;
                break;
            }
            case pointer_float:
            {
                iris_mem * mem_p = new iris_mem;
                //iris_mem_create(std::get<1>(device_names.at(i)) * sizeof(float), mem_p);
                params.push_back(mem_p);
                params_info.push_back(iris_rw);
                pointers++;
                break;
            }
            case pointer_double:
            {
                // double * temp = new double[std::get<1>(device_names.at(i))];
                // temps.push_back(temp);
                iris_mem * mem_p = new iris_mem;
#if 0
                iris_mem_create(std::get<1>(device_names.at(i)) * sizeof(double), mem_p);
#else
    		iris_data_mem_create(mem_p, data.at(i), 
				std::get<1>(device_names.at(i)) * sizeof(double));
#endif
                params.push_back(mem_p);
                params_info.push_back(iris_rw);
                pointers++;
                break;
            }
            case two:
            {
                // double * temp = new double[std::get<1>(device_names.at(i))];
                // temps.push_back(temp);
                iris_mem * mem_p = new iris_mem;
#if 0
                iris_mem_create(std::get<1>(device_names.at(i)) * sizeof(double), mem_p);
#else
    		iris_data_mem_create(mem_p, data.at(i), 
				std::get<1>(device_names.at(i)) * sizeof(double));
#endif
                params.push_back(mem_p);
                params_info.push_back(iris_rw);
                pointers++;
                break;
            }
            default:
                break;
        }
    }
 
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < kernel_names.size(); i++) {
        iris_task task;
        iris_task_create(&task);

#if 0
        iris_task_h2d_full(task, mem_Y, args.at(0));
        iris_task_h2d_full(task, mem_X, args.at(1));
        iris_task_h2d_full(task, mem_sym, args.at(2));
        for(int j = 0; j < data.size(); j++)
            iris_task_h2d_full(task, ((*(iris_mem*)params.at(j+3))), data.at(j));
#endif  
        if(getIRISARCH() == "cuda" || getIRISARCH() == "hip") {
            if(DEBUGOUT) {
                std::cout << "grid: " << kernel_params[i*6] <<  ", " << kernel_params[i*6+1] << ", " << kernel_params[i*6+2] << std::endl;
                std::cout << "block: " << kernel_params[i*6+3] <<  ", " << kernel_params[i*6+4] <<  ", " << kernel_params[i*6+5] << std::endl;
            }
            std::vector<size_t> grid{(size_t)kernel_params[i*6]*kernel_params[i*6+3], (size_t)kernel_params[i*6+1]*kernel_params[i*6+4], (size_t)kernel_params[i*6+2]*kernel_params[i*6+5]};
            std::vector<size_t> block{(size_t)kernel_params[i*6+3], (size_t)kernel_params[i*6+4], (size_t)kernel_params[i*6+5]};
            iris_task_kernel(task, kernel_names.at(i).c_str(), 3, NULL, grid.data(), block.data(), 3+pointers, params.data(), params_info.data());
        } else{
            iris_task_kernel(task, kernel_names.at(i).c_str(), 1, NULL, &size, NULL, 3+pointers, params.data(), params_info.data());
        }
#if 0
        iris_task_d2h_full(task, mem_Y, args.at(0));
#else
	if(i == kernel_names.size() -1)   iris_task_dmem_flush_out(task, mem_Y);
#endif

#if 0
        for(int j = 0; j < data.size(); j++)
            iris_task_d2h_full(task, ((*(iris_mem*)params.at(j+3))), data.at(j));
#endif
        iris_task_submit(task, iris_roundrobin, NULL, 1);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;
    CPUTime = duration.count();
    return getKernelTime();
    platform.finalize();
}


void Executor::execute(std::string input, std::string arch) {
    if(getIRISARCH() == "cuda" || getIRISARCH() == "hip") {
        parseDataStructure ( input );
    } else {
        kernel_names.push_back("iris_spiral_kernel");
    }
    if(arch == "cuda") {
        std::cout << "in cuda code portion\n";
        system("nvcc -ptx kernel.cu");
    }
    else if(arch == "hip") {
        std::cout << "in hip code portion\n";
        system("hipcc --genco -o kernel.hip kernel.hip.cpp");
    }
    else {
        std::string command("gcc");
        command.append(" -I" + getIRIS() + "/include/" + " -I" + getSPIRALHOME() + "/namespaces/");
        command.append(" -O3 -std=c99 -fopenmp -march=native -mavx2 -fPIC -shared -I. -o kernel.openmp.so kernel_openmp.c");
        std::cout << command << std::endl;
        system(command.c_str());
    }
}

float Executor::getKernelTime() {
    return CPUTime;
}

#endif            //  FFTX_MDDFT_HIPBACKEND_HEADER
