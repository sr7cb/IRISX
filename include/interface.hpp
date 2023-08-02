#ifndef FFTX_MDDFT_INTERFACE_HEADER
#define FFTX_MDDFT_INTERFACE_HEADER

//  Copyright (c) 2018-2022, Carnegie Mellon University
//  See LICENSE for details

#include <cstdlib>
#include <vector>
#include <functional>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <map>
// #include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <sys/stat.h>
#include <fcntl.h>
#include <memory>
#include <unistd.h>    // dup2
#include <sys/types.h> // rest for open/close
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <array>
#include <chrono>
#include <bits/stdc++.h>
// #if defined FFTX_CUDA
// #include "cudabackend.hpp"

// #elif defined FFTX_HIP
// #include "hipbackend.hpp"
// #endif
#include "irisbackend.hpp"

// #if defined (FFTX_CUDA) || defined(FFTX_HIP)
// #include "fftx_mddft_gpu_public.h"
// #include "fftx_imddft_gpu_public.h"
// #include "fftx_mdprdft_gpu_public.h"
// #include "fftx_imdprdft_gpu_public.h"
// #else
// #include "fftx_mddft_cpu_public.h"
// #include "fftx_imddft_cpu_public.h"
// #include "fftx_mdprdft_cpu_public.h"
// #include "fftx_imdprdft_cpu_public.h"
// #endif
#pragma once

#if defined ( PRINTDEBUG )
#define DEBUGOUT 1
#else
#define DEBUGOUT 0
#endif

class Executor;
class FFTXProblem;

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        // std::cout << buffer.data() << std::endl;
        result += buffer.data();
    }
    return result;
}

int redirect_input(const char* fname)
{
    int save_stdin = dup(0);
    //std::cout << "in redirect input " << fname << std::endl;
    int input = open(fname, O_RDONLY);
    dup2(input, 0); 
    close(input);
    return save_stdin;
}

int redirect_input(int input)
{
    int save_stdin = dup(0);
    dup2(input, 0);
    close(input);
    return save_stdin;
}

void restore_input(int saved_fd)
{
    close(0);
    dup2(saved_fd, 0);
    close(saved_fd);
}

// transformTuple_t * getLibTransform(std::string name, std::vector<int> sizes) {
//     if(name == "mddft") {
//         return fftx_mddft_Tuple(fftx::point_t<3>({{sizes.at(0), sizes.at(1), sizes.at(2)}}));
//     }
//     else if(name == "imddft") {
//         return fftx_imddft_Tuple(fftx::point_t<3>({{sizes.at(0), sizes.at(1), sizes.at(2)}}));
//     }
//     else if(name == "mdprdft") {
//         return fftx_mdprdft_Tuple(fftx::point_t<3>({{sizes.at(0), sizes.at(1), sizes.at(2)}}));
//     }
//     else if(name == "imdprdft") {
//         return fftx_imdprdft_Tuple(fftx::point_t<3>({{sizes.at(0), sizes.at(1), sizes.at(2)}}));
//     }
//     else {
//         std::cout << "non-supported fixed library transform" << std::endl; 
//         return nullptr;
//     }
// }

std::string getIRISARCH() {
    const char * tmp2 = std::getenv("IRIS_ARCHS");
    std::string tmp(tmp2 ? tmp2 : "");
    if (tmp.empty()) {
        std::cout << "[ERROR] No such variable found, please set IRIS_ARCHS env variable" << std::endl;
        exit(-1);
    }
    return tmp;
}

std::string getFFTX() {
     const char * tmp2 = std::getenv("FFTX_HOME");
    std::string tmp(tmp2 ? tmp2 : "");
    if (tmp.empty()) {
        std::cout << "[ERROR] No such variable found, please download and set FFTX_HOME env variable" << std::endl;
        exit(-1);
    }
    tmp += "/cache_jit_files/"; 
    return tmp;
}

std::string getSPIRAL() {
    const char * tmp2 = std::getenv("SPIRAL_HOME");//required >8.3.1
    std::string tmp(tmp2 ? tmp2 : "");
    if (tmp.empty()) {
        std::cout << "[ERROR] No such variable found, please download and set SPIRAL_HOME env variable" << std::endl;
        exit(-1);
    }
    tmp += "/bin/spiral";         
    return tmp;
}

void getImportAndConfIRIS(std::string arch) {
    std::cout << "Load(fftx);\nImportAll(fftx);" << std::endl;
    std::cout << "ImportAll(simt);\nLoad(jit);\nImport(jit);"<< std::endl;
    if(arch == "cuda" || arch == "cudaopenmp")
        std::cout << "conf := LocalConfig.fftx.confGPU();" << std::endl;
    else if(arch == "hip" || arch == "hipopenmp")
        std::cout << "conf := FFTXGlobals.defaultHIPConf();" << std::endl;
    else if(arch == "openmp")
        std::cout << "conf := FFTXGlobals.defaultConf();" << std::endl;
    else
        std::cout << "conf := FFTXGlobals.defaultConf();" << std::endl;

}

void printIRISBackend(std::string name, std::vector<int> sizes, std::string arch) {
    std::cout << "if 1 = 1 then opts:=conf.getOpts(transform);\ntt:= opts.tagIt(transform);\nif(IsBound(fftx_includes)) then opts.includes:=fftx_includes;fi;\nc:=opts.fftxGen(tt);\nfi;" << std::endl;
    std::cout << "GASMAN(\"collect\");" << std::endl;
    if(arch == "cuda") {
        std::cout << "PrintIRISMETAJIT(c,opts);" << std::endl;
        // std::cout << "opts.prettyPrint(c);" << std::endl;
    } else if(arch == "hip") {
        std::cout << "PrintIRISMETAJIT(c,opts);\n" << std::endl;
    } else if(arch == "openmp") {
        std::cout << "opts.prettyPrint(c);" << std::endl;
    } else {
        std::cout << "opts.prettyPrint(c);" << std::endl;
    }
}

class FFTXProblem {
public:

    std::vector<void*> args;
    std::vector<int> sizes;
    std::string res;
    std::map<std::vector<int>, Executor> executors;
    std::string name;
    FFTXProblem(){
    }

    FFTXProblem(std::string name1) {
        name = name1;
    }

    FFTXProblem(const std::vector<void*>& args1) {
        args = args1;

    }
    FFTXProblem(const std::vector<int>& sizes1) {
       sizes = sizes1;

    }
    FFTXProblem(const std::vector<void*>& args1, const std::vector<int>& sizes1) {
        args = args1;   
        sizes = sizes1;
    }
    FFTXProblem(const std::vector<int>& sizes1, std::string name1) {  
        sizes = sizes1;
        name = name1;
    }
     FFTXProblem(const std::vector<void*>& args1, const std::vector<int>& sizes1, std::string name1) {
        args = args1;   
        sizes = sizes1;
        name = name1;
    }

    void setSizes(const std::vector<int>& sizes1);
    void setArgs(const std::vector<void*>& args1);
    void setName(std::string name);
    void transform();
    std::string semantics2(std::string arch);
    virtual void randomProblemInstance() = 0;
    virtual void semantics() = 0;
    float gpuTime;
    void run(Executor e);
    std::string returnJIT();
    float getTime();
    ~FFTXProblem(){}

};

void FFTXProblem::setArgs(const std::vector<void*>& args1) {
    args = args1;
}

void FFTXProblem::setSizes(const std::vector<int>& sizes1) {
    sizes = sizes1;
}

void FFTXProblem::setName(std::string name1) {
    name = name1;
}

std::string FFTXProblem::semantics2(std::string arch) {
    std::cout << "this is arch " << arch << std::endl;
    std::string tmp = getSPIRAL();
    int p[2];
    if(pipe(p) < 0)
        std::cout << "pipe failed\n";
    std::stringstream out; 
    std::streambuf *coutbuf = std::cout.rdbuf(out.rdbuf()); //save old buf
    getImportAndConfIRIS(arch);
    semantics();
    printIRISBackend(name, sizes, arch);
    std::cout.rdbuf(coutbuf);
    std::string script = out.str();
    int res = write(p[1], script.c_str(), script.size());
    close(p[1]);
    int save_stdin = redirect_input(p[0]);
    std::string result = exec(tmp.c_str());
    restore_input(save_stdin);
    close(p[0]);
    result.erase(result.size()-8);
    std::string f("------------------");
    if(arch == "cuda") {
        result = result.substr(result.find("spiral> JIT BEGIN"));
        std::ofstream kernel, metakernel;
        kernel.open("kernel.cu");
        kernel << result.substr(result.find(f)+18);
        kernel.close();
        metakernel.open("kerneljit.cu");
        metakernel << result;
        metakernel.close();
    } else if(arch == "hip") {
        result.erase(result.size()-8);
        result = result.substr(result.find("spiral> JIT BEGIN"));
        std::ofstream kernel, metakernel;
        kernel.open("kernel.hip.cpp");
        kernel << result.substr(result.find(f)+18);
        kernel.close();
        metakernel.open("kerneljit.hip.cpp");
        metakernel << result;
        metakernel.close();
    } else if(arch == "openmp") {
        result = result.substr(result.find("#include"));
        std::ofstream kernel, metakernel;
        kernel.open("kernel_openmp.c");
        kernel << "#include <stdio.h>\n";
        kernel << "#include \"include/kernel_openmp.h\"\n"; 
        kernel << result;
        kernel << "\n";
        kernel << "int iris_spiral_kernel(double *Y, double *X, double *sym, size_t _off, size_t _ndr) {\n";
        kernel << "init_" << name << "_spiral();\n" << name << "_spiral(Y,X,sym);\n" << "destroy_" << name << "_spiral();\nreturn IRIS_SUCCESS;\n}";
        kernel.close();
        // metakernel.open("kerneljit.hip.cpp");
        // metakernel << result;
        // metakernel.close();
    } else if(arch == "cudaopenmp") {
        result = result.substr(result.find("#include"));
        std::ofstream kernel, metakernel;
        kernel.open("kernel_host2cuda.c");
        kernel << "#include <stdio.h>\n";
        kernel << "#include \"include/kernel_host2cuda.h\"\n"; 
        kernel << result;
        kernel << "\n";
        kernel << "int iris_spiral_kernel(double *Y, double *X, double *sym, size_t _off, size_t _ndr) {\n";
        kernel << "init_" << name << "_spiral();\n" << name << "_spiral(Y,X,sym);\n" << "destroy_" << name << "_spiral();\nreturn IRIS_SUCCESS;\n}";
        kernel.close();
        // metakernel.open("kerneljit.hip.cpp");
        // metakernel << result;
        // metakernel.close();
    } else if(arch == "hipopenmp") {
        result = result.substr(result.find("#include"));
        std::ofstream kernel, metakernel;
        kernel.open("kernel_host2openmp.c");
        kernel << "#include <stdio.h>\n";
        kernel << "#include \"include/kernel_host2hip.h\"\n"; 
        kernel << result;
        kernel << "\n";
        kernel << "int iris_spiral_kernel(double *Y, double *X, double *sym, size_t _off, size_t _ndr) {\n";
        kernel << "init_" << name << "_spiral();\n" << name << "_spiral(Y,X,sym);\n" << "destroy_" << name << "_spiral();\nreturn IRIS_SUCCESS;\n}";
        kernel.close();
        // metakernel.open("kerneljit.hip.cpp");
        // metakernel << result;
        // metakernel.close();
    }
     else{
        std::cout << "not supported arch" << std::endl;
        exit(-1);
    }
    // std::cout << result << std::endl;
    return result;
    // exit(0);
    // return nullptr;
    // return "cuda";
}


void FFTXProblem::transform(){

    // transformTuple_t *tupl = getLibTransform(name, sizes);
    // if(tupl != nullptr) { //check if fixed library has transform
    //     if ( DEBUGOUT) std::cout << "found size in fixed library\n";
    //     ( * tupl->initfp )();
    //     #if defined (FFTX_CUDA) ||  (FFTX_HIP)
    //         DEVICE_EVENT_T custart, custop;
    //         DEVICE_EVENT_CREATE ( &custart );
    //         DEVICE_EVENT_CREATE ( &custop );
    //         DEVICE_EVENT_RECORD ( custart );
    //     #else
    //         auto start = std::chrono::high_resolution_clock::now();
    //     #endif
    //         ( * tupl->runfp ) ( (double*)args.at(0), (double*)args.at(1), (double*)args.at(2) );
    //     #if defined (FFTX_CUDA) ||  (FFTX_HIP)
    //         DEVICE_EVENT_RECORD ( custop );
    //         DEVICE_EVENT_SYNCHRONIZE ( custop );
    //         DEVICE_EVENT_ELAPSED_TIME ( &gpuTime, custart, custop );
    //     #else
    //         auto stop = std::chrono::high_resolution_clock::now();
    //         std::chrono::duration<float, std::milli> duration = stop - start;
    //         gpuTime = duration.count();
    //     #endif
    //     ( * tupl->destroyfp )();
    //     //end time
    // }
    // else { // use RTC
        if(executors.find(sizes) != executors.end()) { //check in memory cache
            if ( DEBUGOUT) std::cout << "cached size found, running cached instance\n";
            run(executors.at(sizes));
        }
        else { //check filesystem cache
            // std::string tmp = getFFTX();
            std::string flag = "";
            if(getIRISARCH().find("openmp") != std::string::npos) {
                flag = "openmp";
            }
            std::stringstream ss(getIRISARCH());
            std::string word;
            while (!ss.eof()) {
                std::ostringstream oss;
                std::getline(ss, word, ',');
                std::cout << "looking for arch " << word << std::endl;
                if(word == "cuda" && flag == "")
                    oss << "kerneljit.cu";
                else if(word == "cuda" && flag == "openmp") 
                    oss << "kernel_host2cuda.c";
                else if(word == "hip" && flag == "")
                    oss << "kerneljit.hip.cpp";
                else if(word == "hip" && flag == "openmp")
                    oss << "kernel_host2hip.c";
                else if(word == "openmp") 
                    oss << "kernel_openmp.c";
                else
                    oss << "borken";
                std::string file_name = oss.str();
                std::ifstream ifs ( file_name );
                if(!ifs) {
                    std::cout << "arch " << word << " not found" << std::endl;
                    if(word != "openmp")
                        res = semantics2(word+flag); 
                    else
                        res = semantics2(word);
                }
            }
            Executor e;
            e.execute();
            executors.insert(std::make_pair(sizes, e));
            run(e);
            // else { //generate code at runtime
            //     std::cout << "haven't seen size, generating\n";
            //     res = semantics2();
            //     Executor e;
            //     e.execute(res, getIRISARCH());
            //     executors.insert(std::make_pair(sizes, e));
            //     run(e);
            // }
        }
    // }
}


void FFTXProblem::run(Executor e) {
    e.initAndLaunch(args, sizes, name);
    // #if (defined FFTX_HIP || FFTX_CUDA)
    // gpuTime = e.initAndLaunch(args);
    // #else
    // gpuTime = e.initAndLaunch(args, name);
    // #endif
}

float FFTXProblem::getTime() {
   return gpuTime;
}

std::string FFTXProblem::returnJIT() {
    if(!res.empty()) {
        return res;
    }
    else{
        return nullptr;
    }
}

#endif            // FFTX_MDDFT_INTERFACE_HEADER
