#ifndef TRAPEZOID
#define TRAPEZOID


// Functions for solving integrals

#include"utils.h"


    
class Trapezoid {
    
    // Generic class based on the trapezoid method

    private:
        func f;
    
    public:
        Trapezoid(func &f){
            this->f = f;
        }

        //~Trapezoid(){ delete f; }

        // Standard trapezoid integration with linear binning
        double integrate(double xmin, double xmax, int n_eval_points){

            float dx = (xmax - xmin) / (float)n_eval_points;
            float x = xmin;
            float f_val = f(xmin);
            double integral = 0; 

            for(int i=1; i<n_eval_points; i++){
                float x_next = x + dx;
                float f_val_next = f(x_next);
                integral += (f_val + f_val_next);
                f_val = f_val_next;
                x = x_next;
            }

            return integral / 2.0 * dx;
        }
        
        // Trapezoid integration with logarithmic binning
        double integrate_log(double xmin, double xmax, int n_eval_points){

            double dx_exp = pow(xmax / xmin, 1 / (double)n_eval_points);
            float x = xmin;
            float f_val = f(xmin);
            double integral = 0; 

            for(int i=1; i<n_eval_points; i++){
                float x_next = x*dx_exp;
                float f_val_next = f(x_next);
                integral += (f_val + f_val_next)*(x_next - x);
                f_val = f_val_next;
                x = x_next;
            }

            return integral / 2.0;
        }
};



#endif