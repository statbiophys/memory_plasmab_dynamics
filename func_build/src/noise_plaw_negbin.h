#ifndef PLAWNEGBIN
#define PLAWNEGBIN


// Function for the integralof the noise model: power law + negative binomial


#include"trapezoid.h"
#include <gsl/gsl_sf_psi.h>
#include <tuple>


struct plaw_negbin_pars {

    plaw_negbin_pars(double beta, double fmin, double a, double b, vecd& Ms){
        this->beta = beta;
        this->fmin = fmin;
        this->a = a;
        this->b = b;
        this->Ms = Ms;
    }

    double beta;
    double fmin;
    double a;
    double b;
    vecd Ms;

    // operator overloading is necessary if we want to use the variable as a key 
    // of a map (not necessary for just solving the integrals)
    bool operator==(const plaw_negbin_pars &pars) const{
        for (unsigned int i=0; i<Ms.size(); i++)
            if (Ms[i] != pars.Ms[i])
                return false;
        return beta == pars.beta && fmin == pars.fmin && a == pars.a && b == pars.b;
    }

    bool operator<(const plaw_negbin_pars &pars) const{
        if (beta < pars.beta)
            return true;
        if (beta == pars.beta && fmin < pars.fmin)
            return true;
        if (beta == pars.beta && fmin == pars.fmin && a < pars.a)
            return true;
        if (beta == pars.beta && fmin == pars.fmin && a == pars.a && b < pars.b)
            return true;
        if (beta == pars.beta && fmin == pars.fmin && a == pars.a && b == pars.b){
            double sum1 = 0;
            for (unsigned int M : Ms) sum1 += M;
            double sum2 = 0;
            for (unsigned int M : pars.Ms) sum2 += M;
            if (sum1 < sum2) return true;
        }
        return false;
    }
};


std::tuple<double, double, double, double> get_negbin_pars(double a, double b, double f, double M){
    double mean = f*M;
    double var = mean + a * pow(mean, b);
    double p = mean/var;
    double r = mean*mean/(var-mean);
    return {mean, var, p, r};
}

// Auxiliary func to compute product of negative binomials
double negbin_prod(double f, const vecui& ns, const plaw_negbin_pars &pars){
    double res = 1;
    for (unsigned int i=0; i<pars.Ms.size(); i++){
        // Writing the parameters p and n from a and b used for the inference
        double mean, var, p, r;
        std::tie(mean, var, p, r) = get_negbin_pars(pars.a, pars.b, f, pars.Ms[i]);
        res *= gsl_ran_negative_binomial_pdf(ns[i], p, r);
    }
    return  res;
}

// Non-normalized integrand of the power-law neg-bin
double plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars &pars) {
    return pow(f, -pars.beta) * negbin_prod(f, ns, pars);
}


// SOLVING INTEGRALS FOR THE POWER-LAW NEGBIN FUNCTION

// Integrating the plaw negative binomial with an adaptive domain, most efficient way
double integr_plaw_negbin_adapt(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000, float f_zero_th=1e-6, float log_x_tolerance=0.01) {

    // Finding the approximate position of function maximum (required for setting different hyperparams of the integration)
    double x_fmax = 0; 
    for (unsigned int i=0; i<pars.Ms.size(); i++)
        x_fmax += ns[i] / pars.Ms[i];
    x_fmax = x_fmax / ns.size() ;
    double norm = plaw_norm(pars.beta, pars.fmin);
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


// DERIVATIVES FOR THE HESSIAN


// Aux derivative: negbin parameter r w.r.t. negbin average mu
double dr_dmu_f(double mu, double a, double b){
    return pow(mu, 1-b) * (2-b) / a;
}

// Aux derivative: negbin parameter r w.r.t. negbin a
double dr_da_f(double mu, double a, double b){
    return -pow(mu,2-b) / a / a;
}

// Aux derivative: negbin parameter r w.r.t. negbin b
double dr_db_f(double mu, double a, double b){
    return -pow(mu, 2-b) * log(mu) / a;
}

// Aux second derivative: negbin parameter r w.r.t. negbin average mu
double d2r_dmu_f(double mu, double a, double b){
    return pow(mu, -b) * (1-b) * (2-b) / a;
}
// Aux second derivative: negbin parameter r w.r.t. negbin a
double d2r_da_f(double mu, double a, double b){
    return 2 * pow(mu, 2-b) / pow(a, 3);
}

// Aux second derivative: negbin parameter r w.r.t. negbin b
double d2r_db_f(double mu, double a, double b){
    return pow(mu, 2-b) * pow(log(mu), 2) / a;
}

// Aux second mixed derivative: negbin parameter r w.r.t. a and mu
double d2r_dmu_a_f(double mu, double a, double b){
    return - (2-b) * pow(mu, 1-b) / a / a;
}

// Aux second mixed derivative: negbin parameter r w.r.t. b and mu
double d2r_dmu_b_f(double mu, double a, double b){
    return - pow(mu, 1-b) / a * (1 + (2-b)*log(mu));
}

// Aux second mixed derivative: negbin parameter r w.r.t. a and b
double d2r_da_b_f(double mu, double a, double b){
    return pow(mu, 2-b) * log(mu) / a / a;
}

// Aux derivative: negbin parameter p w.r.t. negbin average mu
double dp_dmu_f(double mu, double var, double a, double b){
    return a * pow(mu, b) * (1-b) / pow(var, 2);
}

// Aux derivative: negbin parameter p w.r.t. negbin a
double dp_da_f(double mu, double var, double b){
    return -pow(mu, (b+1)) / var / var;
}

// Aux derivative: negbin parameter p w.r.t. negbin b
double dp_db_f(double mu, double var, double a, double b){
    return -a * pow(mu, b+1) * log(mu) / var / var;
}

// Aux second derivative: negbin parameter p w.r.t. negbin average mu
double d2p_dmu_f(double mu, double var, double a, double b){
    return a * (1-b) * pow(mu, b) * (b - 2 - a * b * pow(mu, b-1)) / pow(var, 3);
}

// Aux second derivative: negbin parameter p w.r.t. negbin a
double d2p_da_f(double mu, double var, double b){
    return 2 * pow(mu, 2*b+1) / pow(var, 3);
}

// Aux second derivative: negbin parameter p w.r.t. negbin b
double d2p_db_f(double mu, double var, double a, double b){
    return a * pow(log(mu), 2) * pow(mu, b+1) * (2 * a * pow(mu, b) / var - 1) / var / var;
}

// Aux second mixed derivative: negbin parameter p w.r.t. a and mu
double d2p_dmu_a_f(double mu, double var, double a, double b){
    return pow(mu, b) * (b-1) * (a * pow(mu, b) - mu) / pow(var, 3);
}

// Aux second mixed derivative: negbin parameter p w.r.t. b and mu
double d2p_dmu_b_f(double mu, double var, double a, double b){
    return a * pow(mu, b) / var / var * ((1-b)*log(mu)*(mu - a*pow(mu, b))/var - 1);
}

// Aux second mixed derivative: negbin parameter p w.r.t. a and b
double d2p_da_b_f(double mu, double var, double a, double b){
    return pow(mu, b+1) * log(mu) * (a*pow(mu, b) - mu) / pow(var, 3);
}
    
// Aux derivative: log of negbin w.r.t. p
double dlog_nb_dp_f(double n, double r, double p){
    return r/p - n/(1-p);
}

// Aux second derivative: log of negbin w.r.t. p
double d2log_nb_dp_f(double n, double r, double p){
    return -(r/p/p + n/(1-p)/(1-p));
}

// Aux derivative: log of negbin w.r.t. r
double dlog_nb_dr_f(double n, double r, double p){
    return log(p) + gsl_sf_psi(n+r) - gsl_sf_psi(r);
}

// Aux second derivative: log of negbin w.r.t. r
double d2log_nb_dr_f(double n, double r){
    return gsl_sf_psi_1(n+r) - gsl_sf_psi_1(r);
}

// Aux second mixed derivative: log of negbin w.r.t. r and p
double d2log_nb_dpdr_f(double p){
    return 1/p;
}

// Aux derivative: log of negbin w.r.t. M
double dlog_nb_dM_f(double f, const plaw_negbin_pars& pars, unsigned int n, double mu, double var, double r, double p){
    double dr_dM = f * dr_dmu_f(mu, pars.a, pars.b);
    double dp_dM = f * dp_dmu_f(mu, var, pars.a, pars.b);
    return dp_dM * dlog_nb_dp_f(n, r, p) + dr_dM * dlog_nb_dr_f(n, r, p);
}

// Aux derivative: log of negbin w.r.t. c
double dlog_nb_dc_f(double f, const plaw_negbin_pars& pars, const vecui& ns){
    double dlog_nb_dc = 0;
    for (unsigned int i=0; i<ns.size(); i++){
        double mu, var, p, r;
        std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, f, pars.Ms[i]);
        dlog_nb_dc += pars.Ms[i] * dlog_nb_dM_f(f, pars, ns[i], mu, var, r, p);
    }
    return dlog_nb_dc;
}

// Aux derivative: log of negbin w.r.t. a
double dlog_nb_da_f(double f, const vecui& ns, const vecd& Ms, double a, double b){
    double dlog_nb_da = 0;
    for (unsigned int i=0; i<ns.size(); i++){
        double mu, var, p, r;
        std::tie(mu, var, p, r) = get_negbin_pars(a, b, f, Ms[i]);
        double dr_da = dr_da_f(mu, a, b);
        double dp_da = dp_da_f(mu, var, b);
        dlog_nb_da += dp_da * dlog_nb_dp_f(ns[i], r, p) + dr_da * dlog_nb_dr_f(ns[i], r, p);
    }
    return dlog_nb_da;
}

// Aux derivative: log of negbin w.r.t. b
double dlog_nb_db_f(double f, const vecui& ns, const vecd& Ms, double a, double b){
    double dlog_nb_db = 0;
    for (unsigned int i=0; i<ns.size(); i++){
        double mu, var, p, r;
        std::tie(mu, var, p, r) = get_negbin_pars(a, b, f, Ms[i]);
        double dr_db = dr_db_f(mu, a, b);
        double dp_db = dp_db_f(mu, var, a, b);
        dlog_nb_db += dp_db * dlog_nb_dp_f(ns[i], r, p) + dr_db * dlog_nb_dr_f(ns[i], r, p);
    }
    return dlog_nb_db;
}

// First derivative w.r.t. beta of the integrand
double dalp_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars& pars) {
    double x = pars.fmin/f;
    double res = pow(x, pars.beta) * (1 + (pars.beta-1) * log(x)) / pars.fmin;
    return res * negbin_prod(f, ns, pars);
}

// First derivative w.r.t. beta
double integr_dalp_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000){
    func f_aux = [pars, ns](double f) { return dalp_plaw_negbin(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// Second derivative w.r.t. beta of the integrand
double d2alp_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars& pars) {
    double x = pars.fmin/f;
    double res = pow(x, pars.beta) * log(x) * ((pars.beta-1) * log(x) + 2) / pars.fmin;
    return res * negbin_prod(f, ns, pars);
}

// Second derivative w.r.t. beta
double integr_d2alp_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000){
    func f_aux = [pars, ns](double f) { return d2alp_plaw_negbin(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// First derivative w.r.t. fmin
double integr_dfmin_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000, float f_zero_th=1e-6, float log_x_tolerance=0.01){
    double Ps = negbin_prod(pars.fmin, ns, pars);
    double Pn = integr_plaw_negbin_adapt(pars, ns, n_points, f_zero_th, log_x_tolerance);
    return (pars.beta-1)/pars.fmin * (Pn - Ps);
}

// Second derivative w.r.t. fmin
double integr_d2fmin_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000, float f_zero_th=1e-6, float log_x_tolerance=0.01){
    double Ps = negbin_prod(pars.fmin, ns, pars);
    double Pn = integr_plaw_negbin_adapt(pars, ns, n_points, f_zero_th, log_x_tolerance);
    double dnb_df = 0;
    for (unsigned int i=0; i<ns.size(); i++){
        double mu, var, p, r;
        std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, pars.fmin, pars.Ms[i]);
        double dr_df = pars.Ms[i] * dr_dmu_f(mu, pars.a, pars.b);
        double dp_df = pars.Ms[i] * dp_dmu_f(mu, var, pars.a, pars.b);
        dnb_df += dp_df * dlog_nb_dp_f(ns[i], r, p) + dr_df * dlog_nb_dr_f(ns[i], r, p);
    }
    return (pars.beta-1)/pars.fmin * ((pars.beta-2)*(Pn - Ps)/pars.fmin - dnb_df*Ps);
}

// First derivative w.r.t. M (integrand)
double dM_plaw_negbin(double f, const plaw_negbin_pars &pars, const vecui& ns, int der_i){
    double mu, var, p, r;
    std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, f, pars.Ms[der_i]);
    double dlog_nb_dM = dlog_nb_dM_f(f, pars, ns[der_i], mu, var, r, p);
    return plaw(f, pars.beta, pars.fmin) * negbin_prod(f, ns, pars) * dlog_nb_dM;
}

// First derivative w.r.t. M  
double integr_dM_plaw_negbin(const plaw_negbin_pars& pars, const vecui& ns, int der_i, int n_points=10000) {
    func f_aux = [pars, ns, der_i](double f) { return dM_plaw_negbin(f, pars, ns, der_i); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
} 

// Second derivative w.r.t. M of the integrand
double d2M_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars& pars, int der_i, int der_j) {
    double mu, var, p, r;
    std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, f, pars.Ms[der_i]);
    double dr_dM = f * dr_dmu_f(mu, pars.a, pars.b);
    double dp_dM = f * dp_dmu_f(mu, var, pars.a, pars.b);
    double dlog_nb_dM = dlog_nb_dM_f(f, pars, ns[der_i], mu, var, r, p);
    double coef;
    if (der_i == der_j){   
        coef = dlog_nb_dM*dlog_nb_dM;
        coef += f*f * d2r_dmu_f(mu, pars.a, pars.b) * dlog_nb_dr_f(ns[der_i], r, p);
        coef += dr_dM*dr_dM * d2log_nb_dr_f(ns[der_i], r);
        coef += 2 * dr_dM*dp_dM * d2log_nb_dpdr_f(p);
        coef += f*f * d2p_dmu_f(mu, var, pars.a, pars.b) * dlog_nb_dp_f(ns[der_i], r, p);
        coef += dp_dM*dp_dM * d2log_nb_dp_f(ns[der_i], r, p);
    }
    else{
        std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, f, pars.Ms[der_j]);
        coef = dlog_nb_dM * dlog_nb_dM_f(f, pars, ns[der_j], mu, var, r, p);
    }
    return plaw(f, pars.beta, pars.fmin) * negbin_prod(f, ns, pars) * coef;
} 

// Second derivative w.r.t. M
double integr_d2M_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int der_i, int der_j, int n_points=10000){
    func f_aux = [pars, ns, der_i, der_j](double f) { return d2M_plaw_negbin(f, ns, pars, der_i, der_j); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// First derivative w.r.t. c (integrand)
double dc_plaw_negbin(double f, const plaw_negbin_pars &pars, const vecui& ns){
    return plaw(f, pars.beta, pars.fmin) * negbin_prod(f, ns, pars) * dlog_nb_dc_f(f, pars, ns);
}

// First derivative w.r.t. c  
double integr_dc_plaw_negbin(const plaw_negbin_pars& pars, const vecui& ns, int n_points=10000) {
    func f_aux = [pars, ns](double f) { return dc_plaw_negbin(f, pars, ns); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
} 

// Second derivative w.r.t. c of the integrand
double d2c_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars& pars) {
    double dlog_nb_dc = 0, d2_coef = 0;
    for (unsigned int i=0; i<ns.size(); i++){
        double mu, var, p, r;
        std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, f, pars.Ms[i]);
        double dr_dM = f * dr_dmu_f(mu, pars.a, pars.b);
        double dp_dM = f * dp_dmu_f(mu, var, pars.a, pars.b);
        dlog_nb_dc += pars.Ms[i] * dlog_nb_dM_f(f, pars, ns[i], mu, var, r, p);
        double coef = f*f * d2r_dmu_f(mu, pars.a, pars.b) * dlog_nb_dr_f(ns[i], r, p);
        coef += dr_dM*dr_dM * d2log_nb_dr_f(ns[i], r);
        coef += 2 * dr_dM*dp_dM * d2log_nb_dpdr_f(p);
        coef += f*f * d2p_dmu_f(mu, var, pars.a, pars.b) * dlog_nb_dp_f(ns[i], r, p);
        coef += dp_dM*dp_dM * d2log_nb_dp_f(ns[i], r, p);
        d2_coef += pars.Ms[i]*pars.Ms[i] * coef;
    }
    return plaw(f, pars.beta, pars.fmin) * negbin_prod(f, ns, pars) * (dlog_nb_dc*dlog_nb_dc + dlog_nb_dc + d2_coef);
} 

// Second derivative w.r.t. c
double integr_d2c_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000){
    func f_aux = [pars, ns](double f) { return d2c_plaw_negbin(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// First derivative w.r.t. a (integrand)
double da_plaw_negbin(double f, const plaw_negbin_pars &pars, const vecui& ns){
    return plaw(f, pars.beta, pars.fmin) * negbin_prod(f, ns, pars) * dlog_nb_da_f(f, ns, pars.Ms, pars.a, pars.b);
}

// First derivative w.r.t. a  
double integr_da_plaw_negbin(const plaw_negbin_pars& pars, const vecui& ns, int n_points=10000) {
    func f_aux = [pars, ns](double f) { return da_plaw_negbin(f, pars, ns); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
} 

// Second derivative w.r.t. a of the integrand
double d2a_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars& pars) {
    double d2log_nb_da = 0;
    for (unsigned int i=0; i<ns.size(); i++){
        double mu, var, p, r;
        std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, f, pars.Ms[i]);
        double dr_da = dr_da_f(mu, pars.a, pars.b);
        double d2r_da = d2r_da_f(mu, pars.a, pars.b);
        double dp_da = dp_da_f(mu, var, pars.b);
        double d2p_da = d2p_da_f(mu, var, pars.b);
        d2log_nb_da += d2p_da * dlog_nb_dp_f(ns[i], r, p);
        d2log_nb_da += dp_da*dp_da * d2log_nb_dp_f(ns[i], r, p);
        d2log_nb_da += 2 * dp_da*dr_da * d2log_nb_dpdr_f(p);
        d2log_nb_da += dr_da*dr_da * d2log_nb_dr_f(ns[i], r);
        d2log_nb_da += d2r_da * dlog_nb_dr_f(ns[i], r, p);
    }
    double dlog_nb_da = dlog_nb_da_f(f, ns, pars.Ms, pars.a, pars.b);
    return plaw(f, pars.beta, pars.fmin) * negbin_prod(f, ns, pars) * (dlog_nb_da*dlog_nb_da + d2log_nb_da);
} 

// Second derivative w.r.t. a
double integr_d2a_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000){
    func f_aux = [pars, ns](double f) { return d2a_plaw_negbin(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// First derivative w.r.t. b (integrand)
double db_plaw_negbin(double f, const plaw_negbin_pars &pars, const vecui& ns){
    return plaw(f, pars.beta, pars.fmin) * negbin_prod(f, ns, pars) * dlog_nb_db_f(f, ns, pars.Ms, pars.a, pars.b);
}

// First derivative w.r.t.b  
double integr_db_plaw_negbin(const plaw_negbin_pars& pars, const vecui& ns, int n_points=10000) {
    func f_aux = [pars, ns](double f) { return db_plaw_negbin(f, pars, ns); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
} 

// Second derivative w.r.t. b of the integrand
double d2b_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars& pars) {
    double dlog_nb_db = 0, d2log_nb_db = 0;
    for (unsigned int i=0; i<ns.size(); i++){
        double mu, var, p, r;
        std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, f, pars.Ms[i]);
        double dr_db = dr_db_f(mu, pars.a, pars.b);
        double d2r_db = d2r_db_f(mu, pars.a, pars.b);
        double dp_db = dp_db_f(mu, var, pars.a, pars.b);
        double d2p_db = d2p_db_f(mu, var, pars.a, pars.b);
        dlog_nb_db += dp_db * dlog_nb_dp_f(ns[i], r, p) + dr_db * dlog_nb_dr_f(ns[i], r, p);
        d2log_nb_db += d2p_db * dlog_nb_dp_f(ns[i], r, p);
        d2log_nb_db += dp_db*dp_db * d2log_nb_dp_f(ns[i], r, p);
        d2log_nb_db += 2 * dp_db*dr_db * d2log_nb_dpdr_f(p);
        d2log_nb_db += dr_db*dr_db * d2log_nb_dr_f(ns[i], r);
        d2log_nb_db += d2r_db * dlog_nb_dr_f(ns[i], r, p);
    }
    return plaw(f, pars.beta, pars.fmin) * negbin_prod(f, ns, pars) * (dlog_nb_db*dlog_nb_db + d2log_nb_db);
} 

// Second derivative w.r.t. b
double integr_d2b_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000){
    func f_aux = [pars, ns](double f) { return d2b_plaw_negbin(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// Second mixed derivative w.r.t. beta and fmin
double integr_d2alp_fmin_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000, float f_zero_th=1e-6, float log_x_tolerance=0.01){
    double Pn = integr_plaw_negbin_adapt(pars, ns, n_points, f_zero_th, log_x_tolerance);
    double Ps = negbin_prod(pars.fmin, ns, pars);
    func f_aux = [pars, ns](double f) { return dalp_plaw_negbin(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    double da_P_n = tr.integrate_log(pars.fmin, 1.0, n_points);
    return (Pn - Ps + (pars.beta-1)*da_P_n) / pars.fmin;
}

// Second mixed derivative w.r.t. beta and M (integrand)
double d2alp_M_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars &pars, int der_i){
    double mu, var, p, r;
    std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, f, pars.Ms[der_i]);
    double dlog_nb_dM = dlog_nb_dM_f(f, pars, ns[der_i], mu, var, r, p);
    return dalp_plaw_negbin(f, ns, pars) * dlog_nb_dM;
}

// Second mixed derivative w.r.t. beta and M
double integr_d2alp_M_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int der_i, int n_points=10000){
    func f_aux = [pars, ns, der_i](double f) { return d2alp_M_plaw_negbin(f, ns, pars, der_i); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// Second mixed derivative w.r.t. beta and c (integrand)
double d2alp_c_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars &pars){
    return dalp_plaw_negbin(f, ns, pars) * dlog_nb_dc_f(f, pars, ns);
}

// Second mixed derivative w.r.t. beta and c
double integr_d2alp_c_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000){
    func f_aux = [pars, ns](double f) { return d2alp_c_plaw_negbin(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// Second mixed derivative w.r.t. beta and a (integrand)
double d2alp_a_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars &pars){
    return dalp_plaw_negbin(f, ns, pars) * dlog_nb_da_f(f, ns, pars.Ms, pars.a, pars.b);
}

// Second mixed derivative w.r.t. beta and a
double integr_d2alp_a_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000){
    func f_aux = [pars, ns](double f) { return d2alp_a_plaw_negbin(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}
    
// Second mixed derivative w.r.t. beta and b (integrand)
double d2alp_b_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars &pars){
    return dalp_plaw_negbin(f, ns, pars) * dlog_nb_db_f(f, ns, pars.Ms, pars.a, pars.b);
}

// Second mixed derivative w.r.t. beta and b
double integr_d2alp_b_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000){
    func f_aux = [pars, ns](double f) { return d2alp_b_plaw_negbin(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// Second mixed derivative w.r.t. fmin and M
double integr_d2fmin_M_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int der_i, int n_points=10000){
    double Ps = negbin_prod(pars.fmin, ns, pars);
    double dM_P_n = integr_dM_plaw_negbin(pars, ns, der_i, n_points);
    double mu, var, p, r;
    std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, pars.fmin, pars.Ms[der_i]);
    return  (pars.beta-1)*(dM_P_n - Ps * dlog_nb_dM_f(pars.fmin, pars, ns[der_i], mu, var, r, p)) / pars.fmin;
}

// Second mixed derivative w.r.t. fmin and c
double integr_d2fmin_c_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000){
    double Ps = negbin_prod(pars.fmin, ns, pars);
    double dc_Pn = integr_dc_plaw_negbin(pars, ns, n_points);
    return (pars.beta - 1)*(dc_Pn - Ps * dlog_nb_dc_f(pars.fmin, pars, ns)) / pars.fmin;
}

// Second mixed derivative w.r.t. fmin and a
double integr_d2fmin_a_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000){
    double Ps = negbin_prod(pars.fmin, ns, pars);
    double da_P_n = integr_da_plaw_negbin(pars, ns, n_points);
    double dlog_nb_da = dlog_nb_da_f(pars.fmin, ns, pars.Ms, pars.a, pars.b);
    return  (pars.beta-1)*(da_P_n - Ps * dlog_nb_da) / pars.fmin;
}

// Second mixed derivative w.r.t. fmin and b
double integr_d2fmin_b_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000){
    double Ps = negbin_prod(pars.fmin, ns, pars);
    double db_P_n = integr_db_plaw_negbin(pars, ns, n_points);
    double dlog_nb_db = dlog_nb_db_f(pars.fmin, ns, pars.Ms, pars.a, pars.b);
    return  (pars.beta-1)*(db_P_n - Ps * dlog_nb_db) / pars.fmin;
}

// Second mixed derivative w.r.t. M and a (integrand)
double d2M_a_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars &pars, int der_i){

    double mu, var, p, r;
    std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, f, pars.Ms[der_i]);
    double dlog_nb_dM = dlog_nb_dM_f(f, pars, ns[der_i], mu, var, r, p);
    
    double dr_da = dr_da_f(mu, pars.a, pars.b);
    double dr_dmu = dr_dmu_f(mu, pars.a, pars.b);
    double dp_da = dp_da_f(mu, var, pars.b);
    double dp_dmu = dp_dmu_f(mu, var, pars.a, pars.b);
    
    double d2log_nb_dM_a = d2r_dmu_a_f(mu, pars.a, pars.b) * dlog_nb_dr_f(ns[der_i], r, p);
    d2log_nb_dM_a += dr_da * (dp_dmu * d2log_nb_dpdr_f(p) + dr_dmu * d2log_nb_dr_f(ns[der_i], r));
    d2log_nb_dM_a += d2p_dmu_a_f(mu, var, pars.a, pars.b) * dlog_nb_dp_f(ns[der_i], r, p);
    d2log_nb_dM_a += dp_da * (dp_dmu * d2log_nb_dp_f(ns[der_i], r, p) + dr_dmu * d2log_nb_dpdr_f(p));

    double Ps = negbin_prod(f, ns, pars);
    return plaw(f, pars.beta, pars.fmin) * Ps * (d2log_nb_dM_a*f + dlog_nb_da_f(f, ns, pars.Ms, pars.a, pars.b) * dlog_nb_dM);
}

// Second mixed derivative w.r.t. M and a
double integr_d2M_a_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int der_i, int n_points=10000){
    func f_aux = [pars, ns, der_i](double f) { return d2M_a_plaw_negbin(f, ns, pars, der_i); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// Second mixed derivative w.r.t. c and a (integrand)
double d2c_a_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars &pars){
    double dlog_nb_da = dlog_nb_da_f(f, ns, pars.Ms, pars.a, pars.b);
    double dlog_nb_dc = dlog_nb_dc_f(f, pars, ns);
    double Ps = negbin_prod(f, ns, pars);
    double d2_coef = 0;
    for (unsigned int i=0; i<ns.size(); i++){
        double mu, var, p, r;
        std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, f, pars.Ms[i]);
        double dr_da = dr_da_f(mu, pars.a, pars.b);
        double dr_dmu = dr_dmu_f(mu, pars.a, pars.b);
        double dp_da = dp_da_f(mu, var, pars.b);
        double dp_dmu = dp_dmu_f(mu, var, pars.a, pars.b);
        double coef = d2r_dmu_a_f(mu, pars.a, pars.b) * dlog_nb_dr_f(ns[i], r, p);
        coef += dr_da * (dp_dmu * d2log_nb_dpdr_f(p) + dr_dmu * d2log_nb_dr_f(ns[i], r));
        coef += d2p_dmu_a_f(mu, var, pars.a, pars.b) * dlog_nb_dp_f(ns[i], r, p);
        coef += dp_da * (dp_dmu * d2log_nb_dp_f(ns[i], r, p) + dr_dmu * d2log_nb_dpdr_f(p));
        d2_coef += coef * pars.Ms[i];
    }
    return plaw(f, pars.beta, pars.fmin) * Ps * (d2_coef * f + dlog_nb_da*dlog_nb_dc);
}

// Second mixed derivative w.r.t. c and a
double integr_d2c_a_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000){
    func f_aux = [pars, ns](double f) { return d2c_a_plaw_negbin(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// Second mixed derivative w.r.t. M and b (integrand)
double d2M_b_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars &pars, int der_i){

    double mu, var, p, r;
    std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, f, pars.Ms[der_i]);
    double dlog_nb_dM = dlog_nb_dM_f(f, pars, ns[der_i], mu, var, r, p);
    
    double dr_db = dr_db_f(mu, pars.a, pars.b);
    double dr_dmu = dr_dmu_f(mu, pars.a, pars.b);
    double dp_db = dp_db_f(mu, var, pars.a, pars.b);
    double dp_dmu = dp_dmu_f(mu, var, pars.a, pars.b);
    
    double d2log_nb_dM_b = d2r_dmu_b_f(mu, pars.a, pars.b) * dlog_nb_dr_f(ns[der_i], r, p);
    d2log_nb_dM_b += dr_db * (dp_dmu * d2log_nb_dpdr_f(p) + dr_dmu * d2log_nb_dr_f(ns[der_i], r));
    d2log_nb_dM_b += d2p_dmu_b_f(mu, var, pars.a, pars.b) * dlog_nb_dp_f(ns[der_i], r, p);
    d2log_nb_dM_b += dp_db * (dp_dmu * d2log_nb_dp_f(ns[der_i], r, p) + dr_dmu * d2log_nb_dpdr_f(p));

    double Ps = negbin_prod(f, ns, pars);
    return plaw(f, pars.beta, pars.fmin) * Ps * (d2log_nb_dM_b*f + dlog_nb_db_f(f, ns, pars.Ms, pars.a, pars.b) * dlog_nb_dM);
}

// Second mixed derivative w.r.t. M and b
double integr_d2M_b_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int der_i, int n_points=10000){
    func f_aux = [pars, ns, der_i](double f) { return d2M_b_plaw_negbin(f, ns, pars, der_i); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// Second mixed derivative w.r.t. c and b (integrand)
double d2c_b_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars &pars){
    double dlog_nb_db = dlog_nb_db_f(f, ns, pars.Ms, pars.a, pars.b);
    double dlog_nb_dc = dlog_nb_dc_f(f, pars, ns);
    double Ps = negbin_prod(f, ns, pars);
    double d2_coef = 0;
    for (unsigned int i=0; i<ns.size(); i++){
        double mu, var, p, r;
        std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, f, pars.Ms[i]);
        double dr_db = dr_db_f(mu, pars.a, pars.b);
        double dr_dmu = dr_dmu_f(mu, pars.a, pars.b);
        double dp_db = dp_db_f(mu, var, pars.a, pars.b);
        double dp_dmu = dp_dmu_f(mu, var, pars.a, pars.b);
        double coef = d2r_dmu_b_f(mu, pars.a, pars.b) * dlog_nb_dr_f(ns[i], r, p);
        coef += dr_db * (dp_dmu * d2log_nb_dpdr_f(p) + dr_dmu * d2log_nb_dr_f(ns[i], r));
        coef += d2p_dmu_b_f(mu, var, pars.a, pars.b) * dlog_nb_dp_f(ns[i], r, p);
        coef += dp_db * (dp_dmu * d2log_nb_dp_f(ns[i], r, p) + dr_dmu * d2log_nb_dpdr_f(p));
        d2_coef += coef * pars.Ms[i];
    }
    return plaw(f, pars.beta, pars.fmin) * Ps * (d2_coef * f + dlog_nb_db*dlog_nb_dc);
}

// Second mixed derivative w.r.t. c and b
double integr_d2c_b_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000){
    func f_aux = [pars, ns](double f) { return d2c_b_plaw_negbin(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// Second mixed derivative w.r.t. a and b (integrand)
double d2a_b_plaw_negbin(double f, const vecui& ns, const plaw_negbin_pars &pars){

    double d2log_nb_da_b = 0;
    for (unsigned int i=0; i<ns.size(); i++){
        double mu, var, p, r;
        std::tie(mu, var, p, r) = get_negbin_pars(pars.a, pars.b, f, pars.Ms[i]);
        double dr_db = dr_db_f(mu, pars.a, pars.b);
        double dr_da = dr_da_f(mu, pars.a, pars.b);
        double dp_db = dp_db_f(mu, var, pars.a, pars.b);
        double dp_da = dp_da_f(mu, var, pars.b);
        d2log_nb_da_b += d2r_da_b_f(mu, pars.a, pars.b) * dlog_nb_dr_f(ns[i], r, p);
        d2log_nb_da_b += dr_db * (dp_da * d2log_nb_dpdr_f(p) + dr_da * d2log_nb_dr_f(ns[i], r));
        d2log_nb_da_b += d2p_da_b_f(mu, var, pars.a, pars.b) * dlog_nb_dp_f(ns[i], r, p);
        d2log_nb_da_b += dp_db * (dp_da * d2log_nb_dp_f(ns[i], r, p) + dr_da * d2log_nb_dpdr_f(p));
    }

    double dlog_nb_da = dlog_nb_da_f(f, ns, pars.Ms, pars.a, pars.b);
    double dlog_nb_db = dlog_nb_db_f(f, ns, pars.Ms, pars.a, pars.b);
    return plaw(f, pars.beta, pars.fmin) * negbin_prod(f, ns, pars) * (d2log_nb_da_b + dlog_nb_db * dlog_nb_da);
}

// Second mixed derivative w.r.t. a and b
double integr_d2a_b_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, int n_points=10000){
    func f_aux = [pars, ns](double f) { return d2a_b_plaw_negbin(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

    
vecd jac_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, bool logfmin){

    double Pn = integr_plaw_negbin_adapt(pars, ns);
    unsigned int ns_sum  = 0;
    int z_sign = 1;
    for (unsigned int i=0; i<ns.size(); i++) ns_sum += ns[i];
    if (ns_sum == 0){
        Pn = 1-Pn;
        z_sign = -1;
    }
    vecd jac = vecd(5, 0);
    jac[0] = z_sign * integr_dalp_plaw_negbin(pars, ns) / Pn;
    jac[1] = z_sign * integr_dfmin_plaw_negbin(pars, ns) / Pn;
    jac[2] = z_sign * integr_da_plaw_negbin(pars, ns) / Pn;
    jac[3] = z_sign * integr_db_plaw_negbin(pars, ns) / Pn;
    jac[4] = z_sign * integr_dc_plaw_negbin(pars, ns) / Pn;

    if (logfmin)
        jac[1] = jac[1] * pars.fmin * log(10);

    return jac;
}


// Hessian of a logarithmic plaw-negbin integral explicitely computed. In case of sum(ns) == 0, 
// it computes the Hessian of log(1-P(0)).
// logpars=true returns the hessian of the transformed variables log fmin and log M.
vec2d hess_plaw_negbin(const plaw_negbin_pars &pars, const vecui& ns, bool logpars){
    
    double Pn = integr_plaw_negbin_adapt(pars, ns);
    int z_sign = 1; // special case of sum(ns) = 0
    unsigned int ns_sum  = 0;
    for (unsigned int i=0; i<ns.size(); i++) ns_sum += ns[i];
    if (ns_sum == 0){
        Pn = 1-Pn;
        z_sign = -1;
    }

    // First derivatives
    double dalp = integr_dalp_plaw_negbin(pars, ns);
    double df = integr_dfmin_plaw_negbin(pars, ns);
    double da = integr_da_plaw_negbin(pars, ns);
    double db = integr_db_plaw_negbin(pars, ns);
    double dc = integr_dc_plaw_negbin(pars, ns);
    
    // Matrix of the second derivatives of the log of Pn
    vec2d d2log_mat = vec2d(5, vecd(5, 0));
    // d2 log Pn = ( d2_Pn - d_Pn^2 / Pn ) / Pn
    d2log_mat[0][0] = (z_sign * integr_d2alp_plaw_negbin(pars, ns) - dalp*dalp / Pn) / Pn;
    d2log_mat[0][1] = (z_sign * integr_d2alp_fmin_plaw_negbin(pars, ns) - dalp*df / Pn) / Pn;
    d2log_mat[0][2] = (z_sign * integr_d2alp_a_plaw_negbin(pars, ns) - dalp*da / Pn) / Pn;
    d2log_mat[0][3] = (z_sign * integr_d2alp_b_plaw_negbin(pars, ns) - dalp*db / Pn) / Pn;
    d2log_mat[0][4] = (z_sign * integr_d2alp_c_plaw_negbin(pars, ns) - dalp*dc / Pn) / Pn;
    d2log_mat[1][1] = (z_sign * integr_d2fmin_plaw_negbin(pars, ns) - df*df / Pn) / Pn;
    d2log_mat[1][2] = (z_sign * integr_d2fmin_a_plaw_negbin(pars, ns) - df*da / Pn) / Pn;
    d2log_mat[1][3] = (z_sign * integr_d2fmin_b_plaw_negbin(pars, ns) - df*db / Pn) / Pn;
    d2log_mat[1][4] = (z_sign * integr_d2fmin_c_plaw_negbin(pars, ns) - df*dc / Pn) / Pn;
    d2log_mat[2][2] = (z_sign * integr_d2a_plaw_negbin(pars, ns) - da*da / Pn) / Pn;
    d2log_mat[2][3] = (z_sign * integr_d2a_b_plaw_negbin(pars, ns) - da*db / Pn) / Pn;
    d2log_mat[2][4] = (z_sign * integr_d2c_a_plaw_negbin(pars, ns) - da*dc / Pn) / Pn;
    d2log_mat[3][3] = (z_sign * integr_d2b_plaw_negbin(pars, ns) - db*db / Pn) / Pn;
    d2log_mat[3][4] = (z_sign * integr_d2c_b_plaw_negbin(pars, ns) - db*dc / Pn) / Pn;
    d2log_mat[4][4] = (z_sign * integr_d2c_plaw_negbin(pars, ns) - dc*dc / Pn) / Pn;
    
    // Variable transformation: log10 fmin
    if (logpars){
        double log10 = log(10);
        d2log_mat[0][1] = d2log_mat[0][1] * pars.fmin * log10;
        d2log_mat[1][1] = log10*log10 * pars.fmin * (z_sign * df / Pn + pars.fmin * d2log_mat[1][1]);
        d2log_mat[1][2] = d2log_mat[1][2] * pars.fmin * log10;
        d2log_mat[1][3] = d2log_mat[1][3] * pars.fmin * log10;
        d2log_mat[1][4] = d2log_mat[1][3] * pars.fmin * log10;
    }

    // Copying the symmetric terms
    for (unsigned int i=0; i<5; i++){
        for (unsigned int j=i+1; j<5; j++)
            d2log_mat[j][i] = d2log_mat[i][j];
    }
    return d2log_mat;
}


// Hessian of a logarithmic plaw-negbin integral explicitely computed. In case of sum(ns) == 0, 
// it computes the Hessian of log(1-P(0)).
// logpars=true returns the hessian of the transformed variables log fmin and log M.
vec2d hess_plaw_negbin_M(const plaw_negbin_pars &pars, const vecui& ns, bool logpars){
    
    unsigned int L = pars.Ms.size();
    double Pn = integr_plaw_negbin_adapt(pars, ns);
    int z_sign = 1; // special case of sum(ns) = 0
    unsigned int ns_sum  = 0;
    for (unsigned int i=0; i<L; i++) ns_sum += ns[i];
    if (ns_sum == 0){
        Pn = 1-Pn;
        z_sign = -1;
    }

    // First derivatives
    double dalp = integr_dalp_plaw_negbin(pars, ns);
    double df = integr_dfmin_plaw_negbin(pars, ns);
    double da = integr_da_plaw_negbin(pars, ns);
    double db = integr_db_plaw_negbin(pars, ns);
    vecd dMs = vecd(L);
    for (unsigned int i=0; i<L; i++) dMs[i] = integr_dM_plaw_negbin(pars, ns, i);
    
    // Matrix of the second derivatives of the log of Pn
    vec2d d2log_mat = vec2d(4+L, vecd(4+L, 0));
    // d2 log Pn = ( d2_Pn - d_Pn^2 / Pn ) / Pn
    d2log_mat[0][0] = (z_sign * integr_d2alp_plaw_negbin(pars, ns) - dalp*dalp / Pn) / Pn;
    d2log_mat[1][1] = (z_sign * integr_d2fmin_plaw_negbin(pars, ns) - df*df / Pn) / Pn;
    d2log_mat[2][2] = (z_sign * integr_d2a_plaw_negbin(pars, ns) - da*da / Pn) / Pn;
    d2log_mat[3][3] = (z_sign * integr_d2b_plaw_negbin(pars, ns) - db*db / Pn) / Pn;
    d2log_mat[0][1] = (z_sign * integr_d2alp_fmin_plaw_negbin(pars, ns) - dalp*df / Pn) / Pn;
    d2log_mat[0][2] = (z_sign * integr_d2alp_a_plaw_negbin(pars, ns) - dalp*da / Pn) / Pn;
    d2log_mat[0][3] = (z_sign * integr_d2alp_b_plaw_negbin(pars, ns) - dalp*db / Pn) / Pn;
    d2log_mat[1][2] = (z_sign * integr_d2fmin_a_plaw_negbin(pars, ns) - df*da / Pn) / Pn;
    d2log_mat[1][3] = (z_sign * integr_d2fmin_b_plaw_negbin(pars, ns) - df*db / Pn) / Pn;
    d2log_mat[2][3] = (z_sign * integr_d2a_b_plaw_negbin(pars, ns) - da*db / Pn) / Pn;
    for (unsigned int i=0; i<L; i++){
        d2log_mat[0][4+i] = (z_sign * integr_d2alp_M_plaw_negbin(pars, ns, i) - dalp*dMs[i] / Pn) / Pn;
        d2log_mat[1][4+i] = (z_sign * integr_d2fmin_M_plaw_negbin(pars, ns, i) - df*dMs[i] / Pn) / Pn;
        d2log_mat[2][4+i] = (z_sign * integr_d2M_a_plaw_negbin(pars, ns, i) - da*dMs[i] / Pn) / Pn;
        d2log_mat[3][4+i] = (z_sign * integr_d2M_b_plaw_negbin(pars, ns, i) - db*dMs[i] / Pn) / Pn;
        for (unsigned int j=i; j<L; j++)
            d2log_mat[i+4][j+4] = (z_sign * integr_d2M_plaw_negbin(pars, ns, i, j) - dMs[i]*dMs[j] / Pn) / Pn;
    }
    
    // Variable transformation: log10 fmin, log10 M
    if (logpars){
        double log10 = log(10);
        d2log_mat[0][1] = d2log_mat[0][1] * pars.fmin * log10;
        d2log_mat[1][1] = log10*log10 * pars.fmin * (z_sign * df / Pn + pars.fmin * d2log_mat[1][1]);
        d2log_mat[1][2] = d2log_mat[1][2] * pars.fmin * log10;
        d2log_mat[1][3] = d2log_mat[1][3] * pars.fmin * log10;
        for (unsigned int i=0; i<L; i++){
            d2log_mat[0][4+i] = d2log_mat[0][4+i] * pars.Ms[i] * log10;
            d2log_mat[1][4+i] = d2log_mat[1][4+i] * pars.fmin * pars.Ms[i] * log10*log10;
            d2log_mat[2][4+i] = d2log_mat[2][4+i] * pars.Ms[i] * log10;
            d2log_mat[3][4+i] = d2log_mat[3][4+i] * pars.Ms[i] * log10;
            d2log_mat[4+i][4+i] = log10*log10 * pars.Ms[i] * (z_sign * dMs[i] / Pn + pars.Ms[i] * d2log_mat[4+i][4+i]);
            for (unsigned int j=i+1; j<L; j++)
                d2log_mat[4+i][4+j] = d2log_mat[4+i][4+j] * pars.Ms[i] * pars.Ms[j] * log10*log10;
        }
    }

    // Copying the symmetric terms
    for (unsigned int i=0; i<L+4; i++){
        for (unsigned int j=i+1; j<L+4; j++)
            d2log_mat[j][i] = d2log_mat[i][j];
    }
    return d2log_mat;
}


#endif
