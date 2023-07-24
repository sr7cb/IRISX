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

#if defined ( PRINTDEBUG )
#define DEBUGOUT 1
#else
#define DEBUGOUT 0
#endif

static constexpr auto cmake_script{
R"(
cmake_minimum_required ( VERSION 3.14 )
set ( CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel" )
project ( tmplib LANGUAGES C CXX )

if ( DEFINED ENV{SPIRAL_HOME} )
    set ( SPIRAL_SOURCE_DIR $ENV{SPIRAL_HOME} )
else ()
    if ( "x${SPIRAL_HOME}" STREQUAL "x" )
        message ( FATAL_ERROR "SPIRAL_HOME environment variable undefined and not specified on command line" )
    endif ()
    set ( SPIRAL_SOURCE_DIR ${SPIRAL_HOME} )
endif ()

if ( APPLE )
    if ( ${CMAKE_OSX_ARCHITECTURES} MATCHES "arm64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64.*" )
	    set ( ADDL_COMPILE_FLAGS -arch arm64 )
    elseif ( ${CMAKE_OSX_ARCHITECTURES} MATCHES "x86_64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64.*")
	    set ( ADDL_COMPILE_FLAGS -arch x86_64 )
    endif ()
endif ()

add_library                ( tmp SHARED spiral_generated.c )
target_include_directories ( tmp PRIVATE ${SPIRAL_SOURCE_DIR}/namespaces )
target_compile_options     ( tmp PRIVATE -shared -fPIC ${_addl_options} )

if ( WIN32 )
    set_property    ( TARGET tmp PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON )
endif ()
)"};


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
        //std::vector<std::string> kernels;
        std::vector<std::tuple<std::string, int, std::string>> in_params;
        std::vector<void*> params; 
        std::vector<void *> data;
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
                //convert this to a string because spiral prints string type
                switch(dt) {
                    case 0: //int
                    {
                        if(words.size() < 5) {
                            int * data1 = new int[size];
                            memset(data1, 0, size * sizeof(int));
                            data.push_back(data1);
                        }
                        else {
                            int * data1 = new int[size];
                            for(int i = 4; i < words.size(); i++) {
                                data1[i-4] = atoi(words.at(i).c_str());
                            }
                            data.push_back(data1);
                        }
                        break;
                    }
                    case 1: //float
                    {
                        if(words.size() < 5) {
                            float * data1 = new float[size];
                            memset(data1, 0, size * sizeof(float));
                            data.push_back(data1);
                        }
                        else {
                            float * data1 = new float[size];
                            for(int i = 4; i < words.size(); i++) {
                                data1[i-4] = std::stof(words.at(i));
                            }
                            data.push_back(data1);
                        }
                        break;
                    }
                    case 2: //double
                    {
                        if(words.size() < 5) {
                            double * data1 = new double[size];
                            memset(data1, 0, size * sizeof(double));
                            data.push_back(data1);
                        }
                        else {
                            double * data1 = new double[size];
                            for(int i = 4; i < words.size(); i++) {
                                data1[i-4] = std::stod(words.at(i));
                            }
                            data.push_back(data1);
                            break;    
                        }
                    }
                    case 3: //constant
                    {
                        if(words.size() < 5) {
                            double * data1 = new double[size];
                            memset(data1, 0, size * sizeof(double));
                            data.push_back(data1);
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
    //if ( DEBUGOUT) {
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
    //}
    iris_mem mem_X;
    iris_mem mem_Y;
    iris_mem mem_sym;
#if 1 
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
    std::vector<void*> temps;
    int pointers = 0;
    for(int i = 0; i < device_names.size(); i++) {
        std::string test = std::get<2>(device_names[i]);
        switch(hashit(test)) {
            case constant:
            {
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
                double * temp = new double[std::get<1>(device_names.at(i))];
                temps.push_back(temp);
                iris_mem * mem_p = new iris_mem;
#if 1 
                iris_mem_create(std::get<1>(device_names.at(i)) * sizeof(double), mem_p);
#else
    		iris_data_mem_create(mem_p, temps.at(i), 
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

#if 1
        iris_task_h2d_full(task, mem_Y, args.at(0));
        iris_task_h2d_full(task, mem_X, args.at(1));
        iris_task_h2d_full(task, mem_sym, args.at(2));
        for(int j = 0; j < temps.size(); j++)
            iris_task_h2d_full(task, ((*(iris_mem*)params.at(j+3))), temps.at(j));
#endif    
        std::cout << "grid: " << kernel_params[i*6] <<  ", " << kernel_params[i*6+1] << ", " << kernel_params[i*6+2] << std::endl;
        std::cout << "block: " << kernel_params[i*6+3] <<  ", " << kernel_params[i*6+4] <<  ", " << kernel_params[i*6+5] << std::endl;
        std::vector<size_t> grid{(size_t)kernel_params[i*6], (size_t)kernel_params[i*6+1], (size_t)kernel_params[i*6+2]};
        std::vector<size_t> block{(size_t)kernel_params[i*6+3], (size_t)kernel_params[i*6+4], (size_t)kernel_params[i*6+5]};
        //size_t grid = kernel_params[i*6] * kernel_params[i*6+1] * kernel_params[i*6+2];
        //size_t block = kernel_params[i*6+3] * kernel_params[i*6+4] * kernel_params[i*6+5];
        //std::cout << "launching: " << grid << ", " << block << std::endl;
        iris_task_kernel(task, kernel_names.at(i).c_str(), 3, NULL, grid.data(), block.data(), 3+pointers, params.data(), params_info.data());

#if 1
        iris_task_d2h_full(task, mem_Y, args.at(0));
#else
	if(i == kernel_names.size() -1)   iris_task_dmem_flush_out(task, mem_Y);
#endif

#if 1
        for(int j = 0; j < temps.size(); j++)
            iris_task_d2h_full(task, ((*(iris_mem*)params.at(j+3))), temps.at(j));
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
    parseDataStructure ( input );
    if(arch == "cuda") {
        std::cout << "in cuda code portion\n";
        system("nvcc -ptx kernel.cu");
    }
    else if(arch == "hip") {
        std::cout << "in hip code portion\n";
        system("hipcc --genco -o kernel.hip kernel.hip.cpp");
    }
    else {
        system("gcc -I$(IRIS)/include/ -O3 -std=c99 -fopenmp -fPIC -shared -I. -o kernel.openmp.so kernel.openmp.c");
    }
}

float Executor::getKernelTime() {
    return CPUTime;
}

#endif            //  FFTX_MDDFT_HIPBACKEND_HEADER
