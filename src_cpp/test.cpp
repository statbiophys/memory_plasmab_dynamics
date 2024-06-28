#include "src/noise_plaw_poisson.h"
#include <iostream>


//g++ -o test test.cpp -lgsl -lcblas -lm -std=c++17

int main(int argc, char *argv[]) {
    
    plaw_poiss_pars pars = plaw_poiss_pars(2.0, 1e-6, 1e4);
    std::cout << integr_plaw_poiss(pars, 0, 10000) << std::endl;
    std::cout << integr_plaw_poiss_log(pars, 0, 10000) << std::endl;
    std::cout << integr_plaw_poiss_adapt(pars, 0, 10000, 1e-6, 0.01) << std::endl;
    
    return 0;
}