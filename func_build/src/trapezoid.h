#ifndef TRAPEZOID
#define TRAPEZOID


// Functions for solving integrals

#include"utils.h"


    
class Trapezoid {
    
    // Generic class based on the trapezoid method

    private:
        func f;
    
    public:
        float n_new_points;

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

        // Integration of a positive function with a single maximum in the integration range.
        double integrate_peaked(double xmin, double xmax, double x_fmax, int n_eval_points, 
                                int min_n_eval_points=100, float f_zero_th=1e-6, float log_x_tolerance=0.01){

            double fmax = f(x_fmax);
            double zero_th = fmax * f_zero_th; // th below which the function is considered zero
            func f_zero = [zero_th, this](double x) { return this->f(x) - zero_th; };
            double xmin_new = xmin;
            if (f(xmin) < zero_th){
                xmin_new = bisection(f_zero, xmin, x_fmax, log_x_tolerance);
                xmin_new = std::max(xmin, xmin_new - log_x_tolerance);
            }
            double xmax_new = xmax;
            if (f(xmax) < zero_th){
                xmax_new = bisection(f_zero, x_fmax, xmax, log_x_tolerance);
                xmax_new = std::min(xmax, xmax_new + log_x_tolerance);
            }

            n_new_points = std::max(min_n_eval_points, (int)(n_eval_points * (xmax_new - xmin_new) / (xmax - xmin)));
            return integrate(xmin_new, xmax_new, n_new_points);
        }

        double integrate_peaked_log(double xmin, double xmax, double x_fmax, int n_eval_points, 
                                    int min_n_eval_points=100, float f_zero_th=1e-6, float log_x_tolerance=0.01){

            double fmax = f(x_fmax);
            double zero_th = fmax * f_zero_th; // th below which the function is considered zero
            func f_zero = [zero_th, this](double x) { return this->f(exp(x)) - zero_th; };
            double xmin_new = xmin;
            double log_xmin = log(xmin);
            if (f(xmin) < zero_th){
                xmin_new = bisection(f_zero, log_xmin, log(x_fmax), log_x_tolerance);
                xmin_new = exp(std::max(log_xmin, xmin_new - log_x_tolerance));
            }
            double xmax_new = xmax;
            double log_xmax = log(xmax);
            if (f(xmax) < zero_th){
                xmax_new = bisection(f_zero, log(x_fmax), log_xmax, log_x_tolerance);
                xmax_new = exp(std::min(log_xmax, xmax_new + log_x_tolerance));
            }

            n_new_points = std::max(min_n_eval_points, (int)(n_eval_points * (log(xmax_new) - log(xmin_new)) / (log_xmax - log_xmin)));
            return integrate_log(xmin_new, xmax_new, n_new_points);
        }
};



#endif
