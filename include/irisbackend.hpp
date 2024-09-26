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
#include <unordered_map>
#pragma once

#if defined ( PRINTDEBUG )
#define DEBUGOUT 1
#else
#define DEBUGOUT 0
#endif

// #define ENABLE_KERNEL_FUSION
#define ENABLE_TASK_FUSION
#define GRAPH_MULTIPLE_EXECUTION
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
std::string getIRISX();

bool findOpenMP() {
    if(getIRISARCH().find("openmp") != std::string::npos) {
        return true;
    }
    return false;
}
int redirect_input(int);
void restore_input(int);

class Executor {
    private:
        int parsed = 0;
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
        /*Metadata variables*/
        std::vector<std::tuple<std::string, int, std::string>> device_names;
        std::vector<std::string> kernel_names;
        std::vector<int> kernel_params;
        std::string kernels;
        std::vector<std::tuple<std::string, int, std::string>> in_params;

        /*IRISX specific variables*/
        std::vector<std::vector<std::string>> kernel_args;
        std::unordered_map<std::string, void*> arg2index;
        std::unordered_map<void*, void*> args2output;
        std::unordered_map<int, iris_mem*> index2mem;  
        std::vector<std::tuple<std::string, std::string>> sig_types;
        std::vector<void*> params;
        std::vector<int> params_info;
        std::vector<std::vector<std::tuple<std::string,int>>> new_params_info; 
        std::vector<void *> data;
        std::vector<int> data_lengths;
        int graph_created = 0;
        iris_graph graph;
        int number_params = 0;

        /*Exeuction time in ms*/
        float CPUTime;
    public:
        string_code hashit(std::string const& inString); /*get data type function*/
        float initAndLaunch(std::vector<void*>& args, std::vector<int> sizes, std::string name);/*Execute IRISX task graph*/
        void execute();/*IRIS_ARCHS parser for compilation*/
        void execute2(std::string file_name, std::string arch);/*Backend compiler invocation*/
        float getKernelTime();/*Return runtime*/
        int parseDataStructure(std::string input);/*Metadata parser function*/
        void multiDeviceScheduling();/*IRISX custom scheduling policy*/
        void createGraph(std::vector<void*>& args, std::vector<int> sizes, std::string name, bool iris_graph_created);/*IRISX task graph creation*/
        void retainGraph();/*IRISX graph rentention*/
        int getGraphCreated();/*graph creation verification function*/
        void resetNumberParams();/*backend host pointer reset*/
};

/*get data type function*/
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

/*Metadata parser function*/
int Executor::parseDataStructure(std::string input) {
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
            {   
                kernel_names.push_back(words.at(1));
                kernel_params.push_back(atoi(words.at(2).c_str()));
                kernel_params.push_back(atoi(words.at(3).c_str()));
                kernel_params.push_back(atoi(words.at(4).c_str()));
                kernel_params.push_back(atoi(words.at(5).c_str()));
                kernel_params.push_back(atoi(words.at(6).c_str()));
                kernel_params.push_back(atoi(words.at(7).c_str()));
                std::vector<std::string> localv;
                std::vector<std::tuple<std::string,int>> local_p;
                for(int i = 8; i < words.size(); i+=2) {
                    localv.push_back(words.at(i));
                    local_p.push_back(std::make_pair(words.at(i), strtol(words.at(i+1).c_str(), NULL, 10)));
                }
                kernel_args.push_back(localv);
                new_params_info.push_back(local_p);
                break;
            }
            case 3:
            {
                int loc = atoi(words.at(1).c_str());
                int size = atoi(words.at(2).c_str());
                int dt = atoi(words.at(3).c_str());
                if(DEBUGOUT)
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
                      if(DEBUGOUT)
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
                      if(DEBUGOUT)
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
            case 4:
            {
                sig_types.push_back(std::make_tuple(words.at(1), words.at(2)));
                break;
            }
            break;
        }
    }
    while(std::getline(stream, line)) {
        kernels += line;
        kernels += "\n";
    }
    if ( DEBUGOUT ) std::cout << "parsed input\n";
    return 1;
}

/*IRISX task graph creation*/
void Executor::createGraph(std::vector<void*>& args, std::vector<int> sizes, std::string name, bool iris_graph_created) {
    if ( DEBUGOUT) {
        std::cout << "In create and run graph" << std::endl;
        for(int i = 0; i < sizes.size(); i++) {
            std::cout << "size " << i << ": " << sizes.at(i) << " ";
        }
        std::cout << std::endl;
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
    }

    /*Check if host arguements match kernels*/
    int user_size = args.size() - number_params;
    if(user_size != sig_types.size() && !findOpenMP()) {
        std::cout << "this is the passed in sig size " <<  args.size() << " this is the size of kernel " << sig_types.size() << std::endl;
        std::cout << "Error signatures do not match need to pass more parameters from driver to createGraph" << std::endl;
        std::cout << "Verify that the generated kernel files are the problem you want to run with proper arguments" << std::endl;
        exit(-1);
    }

    /*CPU host pointer task creation DEPRECATED*/
    if((getIRISARCH().find("cuda") == std::string::npos && getIRISARCH().find("hip") == std::string::npos) && findOpenMP()) {
        iris_mem * mem_X = new iris_mem;
        iris_mem * mem_Y = new iris_mem;
        iris_mem * mem_sym = new iris_mem;
        iris_data_mem_create(mem_Y, args.at(0), sizes.at(0) * sizeof(double));
        iris_register_pin_memory(args.at(0), sizes.at(0) * sizeof(double));
        iris_data_mem_create(mem_X, args.at(1), sizes.at(1) * sizeof(double));
        iris_register_pin_memory(args.at(1), sizes.at(1) * sizeof(double));
        iris_data_mem_create(mem_sym, args.at(2), sizes.at(2) * sizeof(double));
        iris_register_pin_memory(args.at(2), sizes.at(2) * sizeof(double));
        params.push_back(mem_Y);
        params.push_back(mem_X);
        params.push_back(mem_sym);
        params_info.push_back(iris_w);
        params_info.push_back(iris_r);
        params_info.push_back(iris_r);

    } else { /*Accelerator host pointer creation*/
      for(int i = 0; i < sig_types.size(); i++) {
          if(DEBUGOUT)
            std::cout << std::get<0>(sig_types.at(i)) << ":" << std::get<1>(sig_types.at(i)) << std::endl;
            std::string type = std::get<1>(sig_types.at(i));
            switch(hashit(type)) {
                case constant:
                {
                    break;
                }
                case pointer_int:
                {
                    iris_mem * mem_p = new iris_mem;
                    iris_data_mem_create(mem_p, args.at(i), sizes.at(i) * sizeof(int));
                    iris_register_pin_memory(args.at(i), sizes.at(i) * sizeof(int));
                    arg2index.insert(std::make_pair(std::get<0>(sig_types.at(i)), mem_p));
                    break;
                }
                case pointer_float:
                {
                    iris_mem * mem_p = new iris_mem;
                    iris_data_mem_create(mem_p, args.at(i), sizes.at(i) * sizeof(float));
                    iris_register_pin_memory(args.at(i), sizes.at(i) * sizeof(float));
                    arg2index.insert(std::make_pair(std::get<0>(sig_types.at(i)), mem_p));
                    break;
                }
                case pointer_double:
                {
                  if(args2output.find(args.at(i+number_params)) == args2output.end()) {
                      iris_mem * mem_p = new iris_mem;
                      iris_data_mem_create(mem_p, args.at(i+number_params), sizes.at(i) * sizeof(double));
                      iris_register_pin_memory(args.at(i+number_params), sizes.at(i) * sizeof(double));
                      args2output.insert(std::make_pair(args.at(i+number_params), mem_p));
                      index2mem.insert(std::make_pair(i+number_params, mem_p));
                      arg2index.insert(std::make_pair(std::get<0>(sig_types.at(i)), mem_p));
                  } else {
                      arg2index.insert(std::make_pair(std::get<0>(sig_types.at(i)), args2output.at(args.at(i+number_params))));
                  }
                  break;
                }
                case two:
                {
                  for(int j = 0; j < new_params_info.size(); j++) {
                      for(int k = 0; k < new_params_info.at(j).size(); k++) {
                          if(std::get<0>(new_params_info.at(j).at(k)) == std::get<0>(sig_types.at(i))) {
                              std::get<1>(new_params_info.at(j).at(k)) = sizeof(double);
                          }
                      }
                  }
                  arg2index.insert(std::make_pair(std::get<0>(sig_types.at(i)), args.at(i+number_params)));
                  break;
                } 
            }
        }
    }

    /*Intermediate memory creation*/
    int pointers = 0;
    for(int i = 0; i < device_names.size(); i++) {
        std::string test = std::get<2>(device_names[i]);
        if(DEBUGOUT)
        std::cout << std::get<0>(device_names[i]) << ":" << test << std::endl;
        switch(hashit(test)) {
            case constant:
            {
                iris_mem * mem_p = new iris_mem;
    		        iris_data_mem_create(mem_p, data.at(i), data_lengths.at(i) * sizeof(double));
                iris_register_pin_memory(data.at(i), data_lengths.at(i) * sizeof(double));
                pointers++;
                arg2index.insert(std::make_pair(std::get<0>(device_names[i]), mem_p));
                break;
            }
            case pointer_int:
            {
                iris_mem * mem_p = new iris_mem;
                iris_data_mem_create(mem_p, data.at(i), std::get<1>(device_names.at(i)) * sizeof(int));
                iris_register_pin_memory(data.at(i), std::get<1>(device_names.at(i)) * sizeof(int));
                pointers++;
                arg2index.insert(std::make_pair(std::get<0>(device_names[i]), mem_p));
                break;
            }
            case pointer_float:
            {
                iris_mem * mem_p = new iris_mem;
                iris_data_mem_create(mem_p, data.at(i), std::get<1>(device_names.at(i)) * sizeof(float));
                iris_register_pin_memory(data.at(i), std::get<1>(device_names.at(i)) * sizeof(float));
                pointers++;
                arg2index.insert(std::make_pair(std::get<0>(device_names[i]), mem_p));
                break;
            }
            case pointer_double:
            {
                iris_mem * mem_p = new iris_mem;
    		        iris_data_mem_create(mem_p, data.at(i), std::get<1>(device_names.at(i)) * sizeof(double));
                iris_register_pin_memory(data.at(i), std::get<1>(device_names.at(i)) * sizeof(double));
                pointers++;
                arg2index.insert(std::make_pair(std::get<0>(device_names[i]), mem_p));
                break;
            }
            case two:
            {
                iris_mem * mem_p = new iris_mem;
    		        iris_data_mem_create(mem_p, data.at(i), std::get<1>(device_names.at(i)) * sizeof(double));
                pointers++;
                arg2index.insert(std::make_pair(std::get<0>(device_names[i]), mem_p));
                break;
            }
            default:
                break;
        }
    }

    if(DEBUGOUT) {
      for (const auto& pair : arg2index) {
          std::cout << pair.first << " ";
      }
      std::cout << std::endl;
    }

    if(!iris_graph_created)
      iris_graph_create(&graph);

#ifndef ENABLE_TASK_FUSION
    /*Task creation*/
    iris_task task[kernel_names.size()];
#else
    iris_task task;
#endif

    if(DEBUGOUT) {
      std::cout << "number of kernels is: " << kernel_names.size() << std::endl;
    }
    for(int i = 0; i < kernel_names.size(); i++) {
      if(DEBUGOUT) {
        std::cout << "kernel name: " << kernel_names.at(i) << std::endl;
      }

#ifndef ENABLE_TASK_FUSION
        iris_task_create(&task[i]);
#else
        if(i == 0) iris_task_create(&task);
#endif
        std::vector<void*> local_params;
        std::vector<int> local_params_info; 
        if((getIRISARCH().find("cuda") != std::string::npos || getIRISARCH().find("hip") != std::string::npos || getIRISARCH().find("opencl") != std::string::npos) && !findOpenMP()) {
            if(DEBUGOUT) {
                std::cout << "grid: " << kernel_params[i*6] <<  ", " << kernel_params[i*6+1] << ", " << kernel_params[i*6+2] << std::endl;
                std::cout << "block: " << kernel_params[i*6+3] <<  ", " << kernel_params[i*6+4] <<  ", " << kernel_params[i*6+5] << std::endl;
            }
            std::vector<size_t> grid{(size_t)kernel_params[i*6]*kernel_params[i*6+3], (size_t)kernel_params[i*6+1]*kernel_params[i*6+4], (size_t)kernel_params[i*6+2]*kernel_params[i*6+5]};
            std::vector<size_t> block{(size_t)kernel_params[i*6+3], (size_t)kernel_params[i*6+4], (size_t)kernel_params[i*6+5]};
            for(int j = 0; j < kernel_args.at(i).size(); j++) {
              if(DEBUGOUT) 
                std::cout << " the first kernel arg " << kernel_args.at(i).at(j) << std::endl;
                if(arg2index.find(kernel_args.at(i).at(j)) != arg2index.end()) {
                  if(DEBUGOUT) 
                    std::cout <<" the second kernel arg " << arg2index.at(kernel_args.at(i).at(j)) << std::endl;
                    local_params.push_back(arg2index.at(kernel_args.at(i).at(j)));
                    local_params_info.push_back(std::get<1>(new_params_info.at(i).at(j)));

                }
            }
#ifndef ENABLE_TASK_FUSION
            iris_task_kernel(task[i], kernel_names.at(i).c_str(), 3, NULL, grid.data(), block.data(), kernel_args.at(i).size(), local_params.data(), local_params_info.data());
#else
            iris_task_kernel(task, kernel_names.at(i).c_str(), 3, NULL, grid.data(), block.data(), kernel_args.at(i).size(), local_params.data(), local_params_info.data());
#endif
        } else{/*CPU task creation DEPRECATED*/
            size_t value = (size_t)sizes.at(0);
#ifndef ENABLE_TASK_FUSION
            iris_task_kernel(task[i], kernel_names.at(i).c_str(), 1, NULL, &value, NULL, 3+pointers, params.data(), params_info.data());
#else
            iris_task_kernel(task, kernel_names.at(i).c_str(), 1, NULL, &value, NULL, 3+pointers, params.data(), params_info.data());
#endif
        }
      /*output flush to host*/
	    if(i == kernel_names.size() -1)
#ifndef ENABLE_TASK_FUSION
          iris_task_d2h_full(task[i], *(iris_mem*)local_params.at(0), args.at(number_params));  
#else
          iris_task_d2h_full(task, *(iris_mem*)local_params.at(0), args.at(number_params));  
#endif
 
#ifndef ENABLE_TASK_FUSION
        iris_graph_task(graph, task[i], iris_gpu, NULL);
#else
	    if(i == 0)
            iris_graph_task(graph, task, iris_gpu, NULL);
#endif
 
    }
    if(DEBUGOUT) {
      std::cout << "number of tasks added " << kernel_names.size() << std::endl;
      std::cout << "total number of tasks in graph " << iris_graph_tasks_count(graph) << std::endl;
    }
    /*arguement reset for DAG fusion*/
    arg2index.clear();
    number_params += sig_types.size();
    multiDeviceScheduling();
}

void Executor::retainGraph(){
    iris_graph_retain(graph, true);
    graph_created = 1;
}

float Executor::initAndLaunch(std::vector<void*>& args, std::vector<int> sizes, std::string name) {
  /*if execute without task graph creation*/
  if(graph_created == 0) {
    createGraph(args, sizes, name, graph_created == 0 ? false : true);
    iris_graph_retain(graph, true);
    graph_created = 1;
    if(DEBUGOUT)
    std::cout << "has the graph been created " << graph_created << std::endl;
  }
  
  if(DEBUGOUT)
    std::cout << "the size of sig mem is" << index2mem.size() << " and " << args2output.size() << std::endl;
  /*IRISX host memory update on each graph submission*/
  for(int i = 0; i < args.size(); i++) {
    if(index2mem.find(i) != index2mem.end()) {
      //std::cout << "updated host pointer dmem object " << i << std::endl;
      iris_data_mem_update(*index2mem.at(i), args.at(i));
      for (auto it = args2output.begin(); it != args2output.end(); ) {
        if (it->second == index2mem.at(i)) {
            it = args2output.erase(it);
            break;
        } else {
            ++it; 
        }
      }
      args2output.insert(std::make_pair(args.at(i), index2mem.at(i)));
    }
  }

  if(DEBUGOUT)
    std::cout << "Executing graph" << std::endl;
#ifndef GRAPH_MULTIPLE_EXECUTION
  std::cout << "hello from graph multi execution" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  iris_graph_submit(graph, iris_default, 1);
  iris_graph_wait(graph);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Graph submission time: " << duration.count() << std::endl;
  CPUTime = duration.count();
  std::cout << "graph submission time: " << duration.count() << std::endl;
#else
  for(int m = 0; m < 10; m++){
    for(int i = 0; i < args.size(); i++) {
        if(index2mem.find(i) != index2mem.end()) {
            //std::cout << "updated host pointer dmem object " << i << std::endl;
            iris_data_mem_update(*index2mem.at(i), args.at(i));
            for (auto it = args2output.begin(); it != args2output.end(); ) {
                if (it->second == index2mem.at(i)) {
                    it = args2output.erase(it);
                break;
                } else {
                    ++it; 
                }
            }
            args2output.insert(std::make_pair(args.at(i), index2mem.at(i)));
        }
    } 
  
    auto start = std::chrono::high_resolution_clock::now();
    iris_graph_submit(graph, iris_default, 1);
    iris_graph_wait(graph);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    CPUTime = duration.count();
    std::cout << "graph submission time: " << duration.count() << std::endl;
  }
#endif

  return getKernelTime();
}

void Executor::execute() {
    if(DEBUGOUT)
        std::cout << "in execute" << std::endl;
    std::string flag = "";
    if(getIRISARCH().find("openmp") != std::string::npos) {
        flag = "openmp";
    }
    if(DEBUGOUT)
        std::cout << "determined if openmp is provided" << std::endl;
    std::stringstream ss(getIRISARCH());
    std::string word;
    while (!ss.eof()) {
        std::ostringstream oss;
        std::getline(ss, word, ':');
        std::cout << word << std::endl;
        if(word == "cuda" && flag == "")
            oss << "kerneljit.cu";
        else if(word == "cuda" && flag == "openmp") 
            oss << "kernel_host2cuda.cu";
        else if(word == "hip" && flag == "")
            oss << "kerneljit.hip.cpp";
        else if(word == "hip" && flag == "openmp")
            oss << "kernel_host2hip.c";
        else if(word == "opencl" && flag == "")
                oss << "kerneljit.cl";
        else if(word == "openmp") 
            oss << "kernel_openmp.c";
        else
            oss << "borken";
        std::string file_name = getIRISX().append("/" + oss.str());
        std::ifstream ifs ( file_name );
        if(ifs) {
            std::string fcontent ( ( std::istreambuf_iterator<char>(ifs) ),
                                    ( std::istreambuf_iterator<char>()    ) );
            if(word != "openmp")
                execute2(fcontent, word+flag); 
            else
                execute2(fcontent, word);
        }
    }
}

void Executor::execute2(std::string input, std::string arch) {
  if(DEBUGOUT)
    std::cout << "arch in execute2 " << arch << std::endl;
    if((arch == "cuda" || arch == "hip" || arch == "opencl") && parsed == 0 && !findOpenMP()) {
        parsed = parseDataStructure ( input );
        if(DEBUGOUT)
        std::cout << "parsed " << parsed << std::endl;
    } else {
      if(DEBUGOUT)
        std::cout << "got into else branch for 1 kernel\n";
        if(kernel_names.size() == 0)
            kernel_names.push_back("iris_spiral_kernel");
    }
    if(arch == "cuda") {
      if(DEBUGOUT)
        std::cout << "in cuda code portion\n";
        std::string command;
        command.append("nvcc -Xcudafe --diag_suppress=declared_but_not_referenced -ptx " + getIRISX() + "/kernel.cu");
        system(command.c_str());
    }
    else if(arch == "hip") {
      if(DEBUGOUT)
        std::cout << "in hip code portion\n";
        std::string command;
        command.append("hipcc --genco -o kernel.hip " + getIRISX() + "/kernel.hip.cpp");
        system(command.c_str());
    }
    else if(arch == "opencl") {
        if(DEBUGOUT)
          std::cout << "in opencl code portion\n";
          std::string command;
          command.append("clang -cc1 -finclude-default-header -triple spir " + getIRISX() + "kernel.cl -O0 -flto -emit-llvm-bc -o kernel.bc");
    }
    else {
        std::string command;
        if(arch == "cudaopenmp") {
            command.append("nvcc");
            command.append(" -I" + getIRIS() + "/include/" + " -I" + getSPIRALHOME() + "/namespaces/ -I.");
            command.append(" -O3");
            command.append(" --compiler-options \'-fPIC\' -shared -o kernel.host2cuda.so kernel_host2cuda.cu -lcuda -lcudart");
        }
        else if(arch == "hipopenmp") {
            command.append("hipcc");
            command.append(" -I" + getIRIS() + "/include/" + " -I" + getSPIRALHOME() + "/namespaces/");
            command.append(" -O3 -std=c99");
            command.append(" -fPIC -shared -I. -o kernel.host2hip.so kernel_host2hip.c");
        } else {
            command.append("gcc");
            command.append(" -I" + getIRIS() + "/include/" + " -I" + getSPIRALHOME() + "/namespaces/");
            command.append(" -O3 -std=c99");
            command.append(" -fopenmp -march=native -mavx2 -fPIC -shared -I. -o kernel.openmp.so kernel_openmp.c");
        }
        if(DEBUGOUT)
        std::cout << command << std::endl;
        system(command.c_str());
    }



}

int Executor::getGraphCreated() {
  std::cout << "Get graph created: " << graph_created << std::endl;
  return graph_created;
}

void Executor::resetNumberParams() {
  number_params = 0;
}

float Executor::getKernelTime() {
    return CPUTime;
}

void Executor::multiDeviceScheduling(){
    int ndevices = 0;
    iris_device_count(&ndevices);
    //int dev_map[16][16];
    int id = 0;  
	int ntasks = iris_graph_tasks_count(graph);
    //printf(" Number of devices: %d and Number of tasks %d \n", ndevices, ntasks);
    iris_task *tasks = NULL;
    if (ntasks > 0)
        tasks = (iris_task *)malloc(sizeof(iris_task)*ntasks);
    iris_graph_get_tasks(graph, tasks);
    for(int i=0; i<ntasks; i++) {
        iris_task task = tasks[i];
        //printf("task %s and task serial %d, total devices %d, device %d \n", iris_task_get_name(task), i, ndevices, id );
        //int id = dev_map[r%nrows][c%ncols];

        /*if(i >=8 && i <= 12)
          id = 1;
        // else if(i >= 13 && i <= 17)
        //   id = 2;
        else
          id = 0;
        */
        iris_task_set_policy(task, id);

#ifndef ENABLE_TASK_FUSION
        if (((i+1) % 19) == 0) {
#elif defined(ENABLE_KERNEL_FUSION) and not defined(ENABLE_TASK_FUSION)
        if (((i+1) % 5) == 0) {
#else
        if (((i+1) % 1) == 0) {
#endif
 
            id = id + 1;
            if (id == ndevices) id = 0;
        }
    }
}
#endif            //  FFTX_MDDFT_HIPBACKEND_HEADER
