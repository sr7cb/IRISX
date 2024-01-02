#include <iostream>
#include "Proto.H"
#include "include/BoxOp_Euler.hpp"
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


int main() {

    // DEFAULT PARAMETERS
    int domainSize = 256;
    int boxSize = 128;
    double maxTime = 1.0;
    int maxStep = 10;
    int outputInterval = 1;
    double gamma = 1.4;
    
    // PARSE COMMAND LINE
    // InputArgs args;
    // args.add("domainSize",     domainSize);
    // args.add("boxSize",        boxSize);
    // args.add("maxTime",        maxTime);
    // args.add("maxStep",        maxStep);
    // args.add("outputInterval", outputInterval);
    // args.parse(argc, argv);
    // args.print();

    double dx = 1.0 / domainSize;
    double dt = 0.25 / domainSize;
    
    // if (procID() == 0)
    // {
    //     std::cout << "dx: " << dx << " | dt: " << dt << std::endl;
    // }
     
    // // INITIALIZE TIMERS
    // PR_TIMER_SETFILE(to_string(domainSize) + ".DIM" + to_string(DIM) + ".LevelEuler.time.table");
    // PR_TIMERS("main");
    
    // INITIALIZE DOMAIN
    auto domain = Box::Cube(domainSize);
    array<bool,DIM> per;
    per.fill(true);
    ProblemDomain pd(domain,per);
    DisjointBoxLayout layout(pd,Point::Ones(boxSize));

    typedef BoxOp_Euler<double> OP;

    // INITIALIZE DATA
    LevelBoxData<double, OP::numState()> U(layout, OP::ghost());
    Operator::initConvolve(U, f_initialize, dx, gamma);
    LevelBoxData<double, OP::numState()> LU(layout, OP::ghost());
    LevelOp<BoxOp_Euler, double> integrator(layout, dx);
    

    integrator(LU,U);

    return 0;
}