#ifndef UTILS
#define UTILS


// Utilities and libraries for all the other functions


#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>


using func = std::function<double(double)>;
using str = std::string;
using vecui = std::vector<unsigned int>;
using vecd = std::vector<double>;
using vec2d = std::vector<vecd>;


double plaw_norm(double alpha, double fmin){
    if (alpha < 0){
        std::cout << "Warning: negative alpha!\n";
        return 0;
    }
    if (alpha == 1)
        return - log(fmin);
    return (1 - pow(fmin, 1-alpha)) / (1 - alpha);
}

double plaw(double f, double alpha, double fmin){
    return pow(f, -alpha) / plaw_norm(alpha, fmin);
}

double plaw_average(double alpha, double fmin){
    if (alpha < 0){
        std::cout << "Warning: negative alpha!\n";
        return 0;
    }
    if (alpha == 2)
        return - log(fmin) / plaw_norm(alpha, fmin);
    return (1 - pow(fmin, 2-alpha)) / (2 - alpha) / plaw_norm(alpha, fmin);
}

double plaw_m2(double alpha, double fmin){
    if (alpha < 0){
        std::cout << "Warning: negative alpha!\n";
        return 0;
    }
    if (alpha == 3)
        return - log(fmin) / plaw_norm(alpha, fmin);
    return (1 - pow(fmin, 3-alpha)) / (3 - alpha) / plaw_norm(alpha, fmin);
}


// Finds root of func(x) between a and b with error eps
double bisection(func fn, double a, double b, double eps)
{
    double fa = fn(a), fb = fn(b);
    if (fa * fb >= 0)
    {
        std::cout << "You have not assumed right a and b in bisection:\n";
        std::cout << "f(a)=" << fa << ", f(b)=" << fb << std::endl << std::endl;
        return fa;
    }
 
    double c = a;
    while ((b-a) >= eps)
    {
        // Find middle point
        c = (a+b)/2;
        double fc = fn(c);
        // Check if middle point is root
        if (fc == 0.0) break;
        // Decide the side to repeat the steps
        else { 
            if (fc*fa < 0)
                b = c;
            else{
                a = c;
                fa = fc;
            }
        }
    }
    return c;
}


#endif
