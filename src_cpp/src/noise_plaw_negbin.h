#ifndef PLAWNEGBIN
#define PLAWNEGBIN


// Function for the integralof the noise model: power law + negative binomial


#include"trapezoid.h"


struct plaw_negbin_pars {

    plaw_negbin_pars(double alpha, double fmin, double a, double b, vecui& Ms){
        this->alpha = alpha;
        this->fmin = fmin;
        this->a = a;
        this->b = b;
        this->Ms = Ms;
    }

    double alpha;
    double fmin;
    double a;
    double b;
    vecui Ms;

    bool operator==(const plaw_negbin_pars &pars) const{
        for (unsigned int i=0; i<Ms.size(); i++)
            if (Ms[i] != pars.Ms[i])
                return false;
        return alpha == pars.alpha && fmin == pars.fmin && a == pars.a && b == pars.b;
    }

    bool operator<(const plaw_negbin_pars &pars) const{
        if (alpha < pars.alpha)
            return true;
        if (alpha == pars.alpha && fmin < pars.fmin)
            return true;
        if (alpha == pars.alpha && fmin == pars.fmin && a < pars.a)
            return true;
        if (alpha == pars.alpha && fmin == pars.fmin && a == pars.a && b < pars.b)
            return true;
        if (alpha == pars.alpha && fmin == pars.fmin && a == pars.a && b == pars.b){
            double sum1 = 0;
            for (unsigned int M : Ms) sum1 += M;
            double sum2 = 0;
            for (unsigned int M : pars.Ms) sum2 += M;
            if (sum1 < sum2) return true;
        }
        return false;
    }
};


// Non-normalized integrand of the power-law neg-bin
// Clone counts are labeled with k to not confuse them with the neg-bin argument n
double plaw_negbin(double f, const vecui& ks, const plaw_negbin_pars &pars) {
    double res = pow(f, -pars.alpha);
    for (unsigned int i=0; i<pars.Ms.size(); i++){
        // Writing the parameters p and n from a and b used for the inference
        double mean = f*pars.Ms[i];
        double var = mean + pars.a * pow(mean, pars.b);
        double p = mean/var;
        double n = mean*mean/(var-mean);
        res *= gsl_ran_negative_binomial_pdf(ks[i], p, n);
    }
    return res;
}


// Integrating the plaw negative binomial with an adaptive domain, most efficient way
double integr_plaw_negbin_adapt(const plaw_negbin_pars &pars, const vecui& ns, int n_points, float f_zero_th, float log_x_tolerance) {

    // Finding the approximate position of function maximum (required for setting different hyperparams of the integration)
    double x_fmax = 0; 
    for (unsigned int i=0; i<pars.Ms.size(); i++)
        x_fmax += ns[i] / (float)pars.Ms[i];
    x_fmax = x_fmax / ns.size() ;
    double norm = plaw_norm(pars.alpha, pars.fmin);
    func f_aux = [pars, ns](double f) { return plaw_negbin(f, ns, pars); };
    if (x_fmax < pars.fmin){ // Parameter inconsistency, integrate with log-binning
        Trapezoid tr = Trapezoid(f_aux);
        return tr.integrate_log(pars.fmin, 1.0, n_points) / norm;
    }

    // Adapting the integration domain by excluding zero region at the boundaries
    double fmax = plaw_negbin(x_fmax, ns, pars);
    double zero_th = fmax * f_zero_th; // th below which function is zero for integral computation
    double log_xmin = log(pars.fmin);
    // The search of zero region from the right and from the left is done with a bisection method in log-space
    func f_exp = [pars, ns, zero_th](double f) { return plaw_negbin(exp(f), ns, pars) - zero_th; };
    double xmin = pars.fmin;
    if (plaw_negbin(xmin, ns, pars) < zero_th){
        xmin = bisection(f_exp, log_xmin, log(x_fmax), log_x_tolerance);
        xmin = exp(std::max(log_xmin, xmin - log_x_tolerance));
    }
    double xmax = 1;
    if (plaw_negbin(xmax, ns, pars) < zero_th){
        xmax = bisection(f_exp, log(x_fmax), 0, log_x_tolerance);
        xmax = exp(std::min(0.0, xmax + log_x_tolerance));
    }

    // Logarithmic trapezoid integration in the adapted domain
    int new_n_points = std::max(100.0, n_points * (log(xmax) - log(xmin)) / (-log_xmin));
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(xmin, xmax, new_n_points) / norm;
}


#endif