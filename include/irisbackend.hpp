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
    std::cout << "In init and launch" << std::endl;
    size_t size = sizes.at(0) * sizes.at(1) * sizes.at(2);
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
#if 1
    iris::Mem mem_X(size);
    iris::Mem mem_Y(size);
    iris::Mem mem_sym(size);
#else
    iris_mem mem_X;
    iris_mem mem_Y;
    iris_mem mem_sym;
    iris_mem_create(size * sizeof(double), &mem_X);
    iris_mem_create(size * sizeof(double), &mem_Y);
    iris_mem_create(size * sizeof(double), &mem_sym);
#endif
    auto start = std::chrono::high_resolution_clock::now();

#if 1
    for(int i = 0; i < kernel_names.size(); i++) {
        iris::Task task;
        task.h2d(&mem_X, 0, size, args.at(1));
        task.h2d(&mem_sym, 0, size, args.at(2));
        void* params[3] = { &mem_Y, &mem_X, &mem_sym };
        int params_info[3] = { iris_w, iris_r, iris_r };
        size_t grid = kernel_params[i*6]*kernel_params[i*6+1]*kernel_params[i*6+2];
        size_t block = kernel_params[i*6+3]*kernel_params[i*6+4]*kernel_params[i*6+5];
        task.kernel(kernel_names.at(i).c_str(), 1, NULL, &grid, &block, 3, params, params_info);
        task.d2h(&mem_Y, 0, size, args.at(0));
        task.submit(iris_roundrobin, NULL, true);
    }
#else
    for(int i = 0; i < kernel_names.size(); i++) {
        iris_task task;
        iris_task_create(&task);
	iris_task_h2d_full(task, mem_Y, args.at(0));
	iris_task_h2d_full(task, mem_X, args.at(1));
	iris_task_h2d_full(task, mem_sym, args.at(2));
        void* params[3] = { &mem_Y, &mem_X, &mem_sym };
        int params_info[3] = { iris_w, iris_r, iris_r };
        //size_t grid = kernel_params[i*6] * kernel_params[i*6+1] * kernel_params[i*6+2];
        //size_t block = kernel_params[i*6+3] * kernel_params[i*6+4] * kernel_params[i*6+5];
	iris_task_kernel(task, kernel_names.at(i).c_str(), 1, NULL, &grid, &block, 3, params, params_info);
	//iris_task_kernel(task, kernel_names.at(i).c_str(), 1, NULL, &grid, &block, 3, params, params_info);
	iris_task_d2h_full(task, mem_Y, args.at(0));
	iris_task_submit(task, iris_gpu, NULL, 1);
    }
#endif
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;
    CPUTime = duration.count();
    return getKernelTime();
}

// float Executor::initAndLaunch(std::vector<void*>& args, std::string name) {
//     if ( DEBUGOUT) std::cout << "Loading shared library\n";
//     #if defined(_WIN32) || defined (_WIN64)
//         shared_lib = dlopen("temp/libtmp.dll", RTLD_LAZY);
//     #elif defined(__APPLE__)
//         shared_lib = dlopen("temp/libtmp.dylib", RTLD_LAZY);
//     #else
//         shared_lib = dlopen("temp/libtmp.so", RTLD_LAZY); 
//     #endif
//     std::ostringstream oss;
//     std::ostringstream oss1;
//     std::ostringstream oss2;
//     oss << "init_" << name << "_spiral";
//     oss1 << name << "_spiral";
//     oss2 << "destroy_" << name << "_spiral";
//     std::string init = oss.str();
//     std::string transform = oss1.str();
//     std::string destroy = oss2.str();
//     if(!shared_lib) {
//         std::cout << "Cannot open library: " << dlerror() << '\n';
//         exit(0);
//     }
//     else if(shared_lib){
//         void (*fn1) ()= (void (*)())dlsym(shared_lib, init.c_str());
//         void (*fn2) (double *, double *, double *) = (void (*)(double *, double *, double *))dlsym(shared_lib, transform.c_str());
//         void (*fn3) ()= (void (*)())dlsym(shared_lib, destroy.c_str());
//         auto start = std::chrono::high_resolution_clock::now();
//         if(fn1) {
//             fn1();
//         }else {
//             std::cout << init << "function didnt run" << std::endl;
//         }
//         if(fn2) {
//             fn2((double*)args.at(0),(double*)args.at(1), (double*)args.at(2));
//         }else {
//             std::cout << transform << "function didnt run" << std::endl;
//         }
//         if(fn3){
//             fn3();
//         }else {
//             std::cout << destroy << "function didnt run" << std::endl;
//         }
//         auto stop = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<float, std::milli> duration = stop - start;
//         CPUTime = duration.count();
//         dlclose(shared_lib);
//     }
//     // system("rm -rf temp");
//     return getKernelTime();
// }


void Executor::execute(std::string input, std::string arch) {
    parseDataStructure ( input );
    for(int i = 0; i < device_names.size(); i++) {
        std::cout << std::get<0>(device_names[i]) << std::endl;
    }
    for(int i = 0; i < kernel_names.size(); i++) {
        std::cout << kernel_names[i] << std::endl;
    }
    if(arch == "cuda") {
        std::cout << "in cuda code portion\n";
        system("nvcc -ptx kernel.cu");
    }
    else if(arch == "hip") {
        system("hipcc --genco -o kernel.hip kenrnel.hip.cpp");
    }
    else {
        system("gcc -I$(IRIS)/include/ -O3 -std=c99 -fopenmp -fPIC -shared -I. -o kernel.openmp.so kernel.openmp.c");
    }
}

// void Executor::execute(std::string result) {
//     if ( DEBUGOUT) std::cout << "entered CPU backend execute\n";
//     std::string compile;
    
//     if ( DEBUGOUT) {
//         std::cout << "created compile\n";
//     }

//     std::string result2 = result.substr(result.find("*/")+3, result.length());
//     int check = mkdir("temp", 0777);
//     // if((check)) {
//     //     std::cout << "failed to create temp directory for runtime code\n";
//     //     exit(-1);
//     // }
//     std::ofstream out("temp/spiral_generated.c");
//     out << result2;
//     out.close();
//     std::ofstream cmakelists("temp/CMakeLists.txt");
//     if(DEBUGOUT)
//         cmakelists << "set ( _addl_options -Wall -Wextra )" << std::endl;

//     cmakelists << cmake_script;
//     cmakelists.close();
//     if ( DEBUGOUT )
//         std::cout << "compiling\n";

//     char buff[FILENAME_MAX]; //create string buffer to hold path
//     getcwd( buff, FILENAME_MAX );
//     std::string current_working_dir(buff);
    
//     check = chdir("temp");
//     // if(!(check)) {
//     //     std::cout << "failed to create temp directory for runtime code\n";
//     //     exit(-1);
//     // }
//     #if defined(_WIN32) || defined (_WIN64)
//         system("cmake . && make");
//     #elif defined(__APPLE__)
//         struct utsname unameData;
//         uname(&unameData);
//         std::string machine_name(unameData.machine);
//         if(machine_name == "arm64")
//             system("cmake -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 . && make");
//         else
//             system("cmake . && make");
//     #else
//         system("cmake . && make"); 
//     #endif
//     check = chdir(current_working_dir.c_str());
//     // if((check)) {
//     //     std::cout << "failed to create temp directory for runtime code\n";
//     //     exit(-1);
//     // }
//     // system("cd ..;");
//     if ( DEBUGOUT )
//         std::cout << "finished compiling\n";
// }

float Executor::getKernelTime() {
    return CPUTime;
}

#endif            //  FFTX_MDDFT_HIPBACKEND_HEADER
