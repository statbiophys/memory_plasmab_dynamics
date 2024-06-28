#ifndef PLAWPOISSON
#define PLAWPOISSON


// Function for the integral and likelihood computation of the noise model: power law + poisson


#include"trapezoid.h"


struct plaw_poiss_pars {

    plaw_poiss_pars(double alpha, double fmin, vecui& Ms){
        this->alpha = alpha;
        this->fmin = fmin;
        this->Ms = Ms;
    }

    double alpha;
    double fmin;
    vecui Ms;

    // operator overloading is necessary if we want to use the variable as a key 
    // of a map (not necessary for just solving the integrals)
    bool operator==(const plaw_poiss_pars &pars) const{
        for (unsigned int i=0; i<Ms.size(); i++)
            if (Ms[i] != pars.Ms[i])
                return false;
        return alpha == pars.alpha && fmin == pars.fmin;
    }

    bool operator<(const plaw_poiss_pars &pars) const{
        if (alpha < pars.alpha)
            return true;
        if (alpha == pars.alpha && fmin < pars.fmin)
            return true;
        if (alpha == pars.alpha && fmin == pars.fmin){
            double sum1 = 0;
            for (unsigned int M : Ms) sum1 += M;
            double sum2 = 0;
            for (unsigned int M : pars.Ms) sum2 += M;
            if (sum1 < sum2) return true;
        }
        return false;
    }
};


// 
// double plaw_poiss(double f, unsigned int n, const plaw_poiss_pars& pars) {
//     return pow(f, -pars.alpha) * gsl_ran_poisson_pdf(n, f*pars.M);
// }

// Non-normalized integrand of the power-law poisson
double plaw_poiss(double f, const vecui& ns, const plaw_poiss_pars& pars) {
    double res = pow(f, -pars.alpha);
    for (unsigned int i=0; i<pars.Ms.size(); i++){
        res *= gsl_ran_poisson_pdf(ns[i], f*pars.Ms[i]);
    }
    return  res;
}


// SOLVING INTEGRALS FOR THE POWER-LAW POISSON FUNCTION

// // Integral of the plaw poisson integrand, trapezoid method with linear frequency binning
// double integr_plaw_poiss(const plaw_poiss_pars &pars, int n, int n_points) {
//     // Converting the function to a single double argument function thorugh lambda
//     func f_aux = [pars, n](double f) { return plaw_poiss(f, n, pars); };
//     Trapezoid tr = Trapezoid(f_aux);
//     double norm = plaw_norm(pars.alpha, pars.fmin);
//     return tr.integrate(pars.fmin, 1.0, n_points) / norm;
// }


// double integr_plaw_poiss_log(const plaw_poiss_pars &pars, int n, int n_points) {
//     // Converting the function to a single double argument function thorugh lambda
//     func f_aux = [pars, n](double f) { return plaw_poiss(f, n, pars); };
//     Trapezoid tr = Trapezoid(f_aux);
//     double norm = plaw_norm(pars.alpha, pars.fmin);
//     return tr.integrate_log(pars.fmin, 1.0, n_points) / norm;
// }


// Integrating the plaw poisson with an adaptive domain, most efficient way
double integr_plaw_poiss_adapt(const plaw_poiss_pars &pars, const vecui& ns, int n_points, float f_zero_th, float log_x_tolerance) {

    // Finding the approximate position of function maximum (required for setting different hyperparams of the integration)
    double x_fmax = 0; 
    for (unsigned int i=0; i<pars.Ms.size(); i++)
        x_fmax += ns[i] / (float)pars.Ms[i];
    x_fmax = x_fmax / ns.size();
    double norm = plaw_norm(pars.alpha, pars.fmin);
    if (x_fmax < pars.fmin){ // Parameter inconsistency
        func f_aux = [pars, ns](double f) { return plaw_poiss(f, ns, pars); };
        Trapezoid tr = Trapezoid(f_aux);
        return tr.integrate_log(pars.fmin, 1.0, n_points) / norm;
    }

    double fmax = plaw_poiss(x_fmax, ns, pars);
    double zero_th = fmax*f_zero_th; // th below function is zero for integral computation
    double log_xmin = log(pars.fmin);
    func f_exp = [pars, ns, zero_th](double f) { return plaw_poiss(exp(f), ns, pars) - zero_th; };
    double xmin = pars.fmin;
    if (plaw_poiss(xmin, ns, pars) < zero_th){
        xmin = bisection(f_exp, log_xmin, log(x_fmax), log_x_tolerance);
        xmin = exp(std::max(log_xmin, xmin - log_x_tolerance));
    }
    double xmax = 1;
    if (plaw_poiss(xmax, ns, pars) < zero_th){
        xmax = bisection(f_exp, log(x_fmax), 0, log_x_tolerance);
        xmax = exp(std::min(0.0, xmax + log_x_tolerance));
    }
        
    //std::cout << xmin << "\t"  << x_fmax << "\t" << xmax << std::endl;

    func f_aux = [pars, ns](double f) { return plaw_poiss(f, ns, pars); };
    int new_n_points = std::max(100.0, n_points * (log(xmax) - log(xmin)) / (-log_xmin));
    //std::cout << f_aux(xmin) << "\t" << f_aux(xmax) << std::endl;
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(xmin, xmax, new_n_points) / norm;
}



/* 
struct like_storage{

    // results of the computation of a constrained log likelihood of a given set of parameters

    like_storage(double log_like, double constraint){
        this->log_like = log_like;
        this->constraint = constraint;
    }

    double log_like;
    double constraint;
};

// Map of a set of parameters to the already computed values of the likelihood
using plaw_poiss_map = std::map<plaw_poiss_pars, like_storage>;

// Class for computing the likelihood. It does not decrease the performance
// compared to the python computation with c++ integrals
class Like_plaw_poiss {

    private:

        // Likelihood parameters
        int N_obs; // Number of unique sequences
        const vecui ns_unique; // Number of unique experimental counts
        const vecui ns_count; // Multiplicity of the experimental counts
        plaw_poiss_map storage_map; 

        // Integration parameters
        int n_points = 10000;
        float f_zero_th = 1e-6;
        float log_x_tolerance = 0.01;

         // Explicit computation of the likelihood and the constraint
        like_storage compute_like(const plaw_poiss_pars &pars){

            double like_0 = integr_plaw_poiss_adapt(pars, 0, n_points, f_zero_th, log_x_tolerance);
            like_0 = std::min(like_0, 1 - 1e-20);
            double N = N_obs / (1 - like_0);
            double constr = plaw_average(pars.alpha, pars.fmin) * N - 1;

            double log_like = 0;
            for(unsigned int i=0; i<ns_count.size(); i++){
                double like_n = integr_plaw_poiss_adapt(pars, ns_unique[i], n_points, f_zero_th, log_x_tolerance);
                log_like += log(like_n) * ns_count[i];
            }

            return like_storage(log_like - N_obs * log(1 - like_0), constr);
        }

        // Check if the computation has been already performed and return the result.
        like_storage get_result(const plaw_poiss_pars &pars){
            plaw_poiss_map::iterator it = storage_map.find(pars);
            like_storage result = compute_like(pars);
            if (it == storage_map.end()){
                
                storage_map.insert(std::pair<plaw_poiss_pars, like_storage>(pars, result));
                return result;
            }
            else{
                //return result;
                return storage_map.at(pars);
            }
        } 

    public:

        Like_plaw_poiss(const vecui &ns_unique, const vecui &ns_count):
        ns_count(ns_count), ns_unique(ns_unique) {
            storage_map = plaw_poiss_map();
            N_obs = 0;
            for(unsigned int i=0; i<ns_count.size(); i++) N_obs += ns_count[i];
        }

        double get_loglike(const plaw_poiss_pars &pars){
            like_storage result = get_result(pars);
            return result.log_like;
        }

        double get_constraint(const plaw_poiss_pars &pars){
            like_storage result = get_result(pars);
            return result.constraint;
        } 
};

*/


#endif