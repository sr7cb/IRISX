#include <iostream>
#if defined PROTO_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#endif
#include "Proto.H"
#include <chrono>
#include "include/checkanswer.hpp"
#include "examples/_common/InputParser.H"

#if defined IRIS
#include <iris/iris.hpp>
#include <iris/iris_openmp.h>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#include <include/interface.hpp>
#include <include/protoeulerlib.hpp>
#pragma GCC diagnostic pop
ProtoProblem pp("leveleuler");
std::vector<void*> largs;
int n,m;
int k = 5;
#include "include/BoxOp_Euler_iris.hpp"
#else
#include "include/BoxOp_Euler.hpp"
#endif
#include "/ccs/home/sanilrao/IPDPS24/include/LevelRK4_DAG.H"
using namespace Proto;


template<typename T, unsigned int C, MemType MEM>
PROTO_KERNEL_START 
void f_initialize_(Point& a_pt, Var<T, C, MEM>& a_U, double a_dx, const double a_gamma)
{
    double x = a_pt[0]*a_dx + a_dx/2.0;
    double rho0 = a_gamma;
    double p0 = 1.0;
    double umag = 0.0;
    double rho = rho0;
    rho += .01*rho0*sin(2*2*M_PI*x);
    double p = p0*pow(rho/rho0,a_gamma);
    a_U(0) = rho;
    double c0 = sqrt(a_gamma*p0/rho0);
    double c = sqrt(a_gamma*p/rho);
    umag = 2*(c-c0)/(a_gamma-1.);
    a_U(1) = rho*umag;
    //NOTE: This assumes that C=DIM+2
    for(int dir=2; dir<=DIM; dir++)
        a_U(dir)=0.0;
    double ke = 0.5*umag*umag;
    a_U(C-1) = p/(a_gamma-1.0) + rho*ke;
}
PROTO_KERNEL_END(f_initialize_, f_initialize)


int main(int argc, char** argv)
{

#if defined IRIS
  iris_init(&argc, &argv, 1);
  // int n,m,k;
  //   n = 64+8; //40
  //   m = 64+8; //40
  //   k = 5;
  //   std::vector<int> sizes{(n-8)*(m-8)*(n-8)*k, n*m*n*k, 1, 1, 1, n, m};
  //   double * a = new double[n*m*k];
  //   double * b = new double[n*m*k];
  //   double c = 1.4;
  //   double d =  0.00390625;
  //   double e = 1.0 / ((n-8)*2);
  //   // double c = 1;
  //   // double d = 1;
  //   // double e = 1;
  //   std::vector<void*> largs{a, b, &c, &d, &e};
  //   pp.setArgs(largs);
  //   pp.setSizes(sizes);
    pp.readKernels();
    #if defined TIME 
        //std::cout << "Hello from inner time\n";
        auto start = std::chrono::high_resolution_clock::now();
    #endif
    // pp.createGraph();
#endif

#if defined TIME && !defined IRIS
    std::cout << "Hello from outer time\n";
    auto start = std::chrono::high_resolution_clock::now();
#endif

#ifdef PR_MPI
    double start, end;
    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#endif

    // DEFAULT PARAMETERS
    int domainSize =512;//128; //64
    int boxSize = 64;//64; //32
    // double maxTime = 1.0;
    // int maxStep = 10;
    double maxTime = 0.25 / domainSize;
    int maxStep = 1;
    int outputInterval = 1;
    double gamma = 1.4;
    
    // PARSE COMMAND LINE
    InputArgs args;
    args.add("domainSize",     domainSize);
    args.add("boxSize",        boxSize);
    args.add("maxTime",        maxTime);
    args.add("maxStep",        maxStep);
    args.add("outputInterval", outputInterval);
    args.parse(argc, argv);
    args.print();
    n = boxSize; 
    m = boxSize;
    double dx = 1.0 / domainSize;
    double dt = 0.25 / domainSize;
    
    if (procID() == 0)
    {
        std::cout << "dx: " << dx << " | dt: " << dt << std::endl;
    }
     
    // INITIALIZE TIMERS
    PR_TIMER_SETFILE(to_string(domainSize) + ".DIM" + to_string(DIM) + ".LevelEuler.time.table");
    PR_TIMERS("main");
    
    // INITIALIZE DOMAIN
    auto domain = Box::Cube(domainSize);
    array<bool,DIM> per;
    per.fill(true);
    ProblemDomain pd(domain,per);
    DisjointBoxLayout layout(pd,Point::Ones(boxSize));

    typedef BoxOp_Euler<double> OP;

    // INITIALIZE DATA
    LevelBoxData<double, OP::numState()> U(layout, OP::ghost());
    // U.initConvolve(f_initialize, dx, gamma);
    Operator::initConvolve(U, f_initialize, dx, gamma);

    // DO INTEGRATION
    LevelRK4<BoxOp_Euler, double> integrator(layout, dx);
    double time = 0.0;
#ifdef PR_HDF5
    HDF5Handler h5;
#endif
    std::vector<std::string> varnames(OP::numState());
    varnames[0] = "rho";
    for (int ii = 1; ii <= DIM; ii++)
    {
        varnames[ii] = ("rho*v"+std::to_string(ii-1));
    }
    varnames[DIM+1] = "rho*E";
#ifdef PR_HDF5
    h5.writeLevel(varnames, dx, U, "U_0");
#endif

    for (int k = 0; ((k < maxStep) && (time < maxTime)); k++)
    // for (int k = 0; k < 1; k++)
    {
        integrator.advance(U, dt, time);
        if ((k+1) % outputInterval == 0)
        {
#ifdef PR_HDF5
            h5.writeLevel(varnames, dx, U, "U_%i", k+1);
#endif
        }
        time += dt;
        // std::cout << k << " " << maxStep << " " << time << " " << maxTime << std::endl;
    }

    // pp.transform();

#ifdef PR_MPI
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    end = MPI_Wtime();
    MPI_Finalize();
    std::cout << "the time is " << end - start << std::endl;
#endif

#if defined TIME
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "The total time is " << duration.count() << std::endl;
#endif
    
#if defined IRIS
  iris_finalize();
#endif
}
