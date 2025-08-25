#ifndef PLAWPOISSON
#define PLAWPOISSON


// Function for the integral and likelihood computation of the noise model: power law + poisson


#include"trapezoid.h"


struct plaw_poiss_pars {

    plaw_poiss_pars(double beta, double fmin, vecd& Ms){
        this->beta = beta;
        this->fmin = fmin;
        this->Ms = Ms;
    }

    double beta;
    double fmin;
    vecd Ms;

    // operator overloading is necessary if we want to use the variable as a key 
    // of a map (not necessary for just solving the integrals)
    bool operator==(const plaw_poiss_pars &pars) const{
        for (unsigned int i=0; i<Ms.size(); i++)
            if (Ms[i] != pars.Ms[i])
                return false;
        return beta == pars.beta && fmin == pars.fmin;
    }

    bool operator<(const plaw_poiss_pars &pars) const{
        if (beta < pars.beta)
            return true;
        if (beta == pars.beta && fmin < pars.fmin)
            return true;
        if (beta == pars.beta && fmin == pars.fmin){
            double sum1 = 0;
            for (unsigned int M : Ms) sum1 += M;
            double sum2 = 0;
            for (unsigned int M : pars.Ms) sum2 += M;
            if (sum1 < sum2) return true;
        }
        return false;
    }
};

// Auxiliary func to compute product of poissinian in the plaw_poisson integral
double poisson_prod(double f, const vecui& ns, const vecd& Ms){
    double res = 1;
    for (unsigned int i=0; i<Ms.size(); i++){
        res *= gsl_ran_poisson_pdf(ns[i], f*Ms[i]);
    }
    return  res;
}

// Non-normalized integrand of the power-law poisson
double plaw_poiss(double f, const vecui& ns, const plaw_poiss_pars& pars) {
    return pow(f, -pars.beta) * poisson_prod(f, ns, pars.Ms);
}


// SOLVING INTEGRALS FOR THE POWER-LAW POISSON FUNCTION

// Integrating the plaw poisson with an adaptive domain, most efficient way
double integr_plaw_poiss_adapt(const plaw_poiss_pars &pars, const vecui& ns, int n_points=10000, float f_zero_th=1e-6, float log_x_tolerance=0.01) {

    // Finding the approximate position of function maximum (required for setting different hyperparams of the integration)
    double x_fmax = 0; 
    for (unsigned int i=0; i<pars.Ms.size(); i++)
        x_fmax += ns[i] / (float)pars.Ms[i];
    x_fmax = x_fmax / ns.size();
    double norm = plaw_norm(pars.beta, pars.fmin);
    if (x_fmax < pars.fmin){ // Parameter inconsistency
        func f_aux = [pars, ns](double f) { return plaw_poiss(f, ns, pars); };
        Trapezoid tr = Trapezoid(f_aux);
        return tr.integrate_log(pars.fmin, 1.0, n_points) / norm;
    }

    double fmax = plaw_poiss(x_fmax, ns, pars);
    double zero_th = fmax*f_zero_th; // th below which the function is considered zero
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

    func f_aux = [pars, ns](double f) { return plaw_poiss(f, ns, pars); };
    int new_n_points = std::max(100.0, n_points * (log(xmax) - log(xmin)) / (-log_xmin));
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(xmin, xmax, new_n_points) / norm;
}


// DERIVATIVES FOR THE HESSIAN

// First derivative w.r.t. beta of the integrand
double da_plaw_poiss(double f, const vecui& ns, const plaw_poiss_pars& pars) {
    double x = pars.fmin/f;
    double res = pow(x, pars.beta) * (1 + (pars.beta-1) * log(x)) / pars.fmin;
    return res * poisson_prod(f, ns, pars.Ms);
}

// First derivative w.r.t. beta
double integr_da_plaw_poiss(const plaw_poiss_pars &pars, const vecui& ns, int n_points=10000){
    func f_aux = [pars, ns](double f) { return da_plaw_poiss(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// Second derivative w.r.t. beta of the integrand
double d2a_plaw_poiss(double f, const vecui& ns, const plaw_poiss_pars& pars) {
    double x = pars.fmin/f;
    double res = pow(x, pars.beta) * log(x) * ((pars.beta-1) * log(x) + 2) / pars.fmin;
    return res * poisson_prod(f, ns, pars.Ms);
}

// Second derivative w.r.t. beta
double integr_d2a_plaw_poiss(const plaw_poiss_pars &pars, const vecui& ns, int n_points=10000){
    func f_aux = [pars, ns](double f) { return d2a_plaw_poiss(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// First derivative w.r.t. fmin
double integr_dfmin_plaw_poiss(const plaw_poiss_pars &pars, const vecui& ns, int n_points=10000, float f_zero_th=1e-6, float log_x_tolerance=0.01){
    double Ps = poisson_prod(pars.fmin, ns, pars.Ms);
    double Pn = integr_plaw_poiss_adapt(pars, ns, n_points, f_zero_th, log_x_tolerance);
    return (pars.beta-1)/pars.fmin * (Pn - Ps);
}

// Second derivative w.r.t. fmin
double integr_d2fmin_plaw_poiss(const plaw_poiss_pars &pars, const vecui& ns, int n_points=10000, float f_zero_th=1e-6, float log_x_tolerance=0.01){
    double Ps = poisson_prod(pars.fmin, ns, pars.Ms);
    double Pn = integr_plaw_poiss_adapt(pars, ns, n_points, f_zero_th, log_x_tolerance);
    double aux = 0;
    for (unsigned int i=0; i<ns.size(); i++) aux += (ns[i] - pars.fmin*pars.Ms[i]);
    return (pars.beta-1)/pars.fmin/pars.fmin * ((pars.beta-2)*(Pn - Ps) - aux*Ps);
}

// First derivative w.r.t. M
double integr_dM_plaw_poiss(const plaw_poiss_pars& pars, const vecui& ns, int der_i, int n_points=10000) {
    func f_aux = [pars, ns, der_i](double f) { 
        return plaw_poiss(f, ns, pars)/plaw_norm(pars.beta, pars.fmin) * (ns[der_i]/(double)pars.Ms[der_i] - f); 
    };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
} 

// Second derivative w.r.t. M of the integrand
double d2M_plaw_poiss(double f, const vecui& ns, const plaw_poiss_pars& pars, int der_i, int der_j) {
    double Ps = poisson_prod(f, ns, pars.Ms);
    double plaw = pow(f, -pars.beta) / plaw_norm(pars.beta, pars.fmin);
    double coef;
    if (der_i == der_j)
        coef = pow(ns[der_i]/(double)pars.Ms[der_i] - f, 2) - ns[der_i]/(double)pars.Ms[der_i]/(double)pars.Ms[der_i];
    else
        coef = (ns[der_i]/(double)pars.Ms[der_i] - f) * (ns[der_j]/(double)pars.Ms[der_j] - f);
    return plaw * Ps * coef;
} 

// Second derivative w.r.t. M
double integr_d2M_plaw_poiss(const plaw_poiss_pars &pars, const vecui& ns, int der_i, int der_j, int n_points=10000){
    func f_aux = [pars, ns, der_i, der_j](double f) { return d2M_plaw_poiss(f, ns, pars, der_i, der_j); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// First derivative w.r.t. c
double integr_dc_plaw_poiss(const plaw_poiss_pars& pars, const vecui& ns, int n_points=10000) {
    double ns_sum = 0, Ms_sum = 0;
    for (unsigned int i=0; i<ns.size(); i++){
        ns_sum += ns[i];
        Ms_sum += pars.Ms[i];
    }
    double n = plaw_norm(pars.beta, pars.fmin);
    func f_aux = [pars, ns, n, ns_sum, Ms_sum](double f) { 
        return plaw_poiss(f, ns, pars) / n * (ns_sum - f*Ms_sum);
    };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// Second derivative w.r.t. c
double integr_d2c_plaw_poiss(const plaw_poiss_pars& pars, const vecui& ns, int n_points=10000) {
    double ns_sum = 0, Ms_sum = 0;
    for (unsigned int i=0; i<ns.size(); i++){
        ns_sum += ns[i];
        Ms_sum += pars.Ms[i];
    }
    double n = plaw_norm(pars.beta, pars.fmin);
    func f_aux = [pars, ns, n, ns_sum, Ms_sum](double f) { 
        return plaw_poiss(f, ns, pars) / n * (pow(ns_sum - f*Ms_sum, 2) - f * Ms_sum); 
    };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}

// Second mixed derivative w.r.t. beta and fmin
double integr_d2afmin_plaw_poiss(const plaw_poiss_pars &pars, const vecui& ns, int n_points=10000, float f_zero_th=1e-6, float log_x_tolerance=0.01){
    double Pn = integr_plaw_poiss_adapt(pars, ns, n_points, f_zero_th, log_x_tolerance);
    double Ps = poisson_prod(pars.fmin, ns, pars.Ms);
    func f_aux = [pars, ns](double f) { return da_plaw_poiss(f, ns, pars); };
    Trapezoid tr = Trapezoid(f_aux);
    double da_P_n = tr.integrate_log(pars.fmin, 1.0, n_points);
    return (Pn - Ps + (pars.beta-1)*da_P_n) / pars.fmin;
}

// Second mixed derivative w.r.t. beta and c
double integr_d2ac_plaw_poiss(const plaw_poiss_pars &pars, const vecui& ns, int n_points=10000){
    double ns_sum = 0, Ms_sum = 0;
    for (unsigned int i=0; i<ns.size(); i++){
        ns_sum += ns[i];
        Ms_sum += pars.Ms[i];
    }
    func f_aux = [pars, ns, ns_sum, Ms_sum](double f) { return da_plaw_poiss(f, ns, pars) * (ns_sum - f*Ms_sum); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}
    
// Second mixed derivative w.r.t. fmin and c
double integr_d2fminc_plaw_poiss(const plaw_poiss_pars &pars, const vecui& ns, int n_points=10000){
    double ns_sum = 0, Ms_sum = 0;
    for (unsigned int i=0; i<ns.size(); i++){
        ns_sum += ns[i];
        Ms_sum += pars.Ms[i];
    }
    double Ps = poisson_prod(pars.fmin, ns, pars.Ms);
    double dc_P_n = integr_dc_plaw_poiss(pars, ns, n_points);
    return (pars.beta-1)*(dc_P_n - Ps * (ns_sum - pars.fmin*Ms_sum)) / pars.fmin;
}

// Second mixed derivative w.r.t. beta and M
double integr_d2aM_plaw_poiss(const plaw_poiss_pars &pars, const vecui& ns, int der_i, int n_points=10000){
    func f_aux = [pars, ns, der_i](double f) { return da_plaw_poiss(f, ns, pars) * (ns[der_i]/(double)pars.Ms[der_i] - f); };
    Trapezoid tr = Trapezoid(f_aux);
    return tr.integrate_log(pars.fmin, 1.0, n_points);
}
    
// Second mixed derivative w.r.t. fmin and M
double integr_d2fminM_plaw_poiss(const plaw_poiss_pars &pars, const vecui& ns, int der_i, int n_points=10000){
    double Ps = poisson_prod(pars.fmin, ns, pars.Ms);
    func f_aux = [pars, ns, der_i](double f) { 
        return plaw_poiss(f, ns, pars)/plaw_norm(pars.beta, pars.fmin) * (ns[der_i]/(double)pars.Ms[der_i] - f); 
    };
    Trapezoid tr = Trapezoid(f_aux);
    double dM_P_n = tr.integrate_log(pars.fmin, 1.0, n_points);
    return (pars.beta-1)*(dM_P_n - Ps * (ns[der_i]/(double)pars.Ms[der_i] - pars.fmin)) / pars.fmin;
}


vecd jac_plaw_poisson(const plaw_poiss_pars &pars, const vecui& ns, bool logfmin){

    double Pn = integr_plaw_poiss_adapt(pars, ns);
    unsigned int ns_sum  = 0;
    int z_sign = 1;
    for (unsigned int i=0; i<ns.size(); i++) ns_sum += ns[i];
    if (ns_sum == 0){
        Pn = 1-Pn;
        z_sign = -1;
    }
    vecd jac = vecd(3, 0);
    jac[0] = z_sign * integr_da_plaw_poiss(pars, ns) / Pn;
    jac[1] = z_sign * integr_dfmin_plaw_poiss(pars, ns) / Pn;
    jac[2] = z_sign * integr_dc_plaw_poiss(pars, ns) / Pn;

    if (logfmin)
        jac[1] = jac[1] * pars.fmin * log(10);

    return jac;
}


// Hessian of a logarithmic plaw-poisson integral explicitely computed. In case of sum(ns) == 0, 
// it compute the Hessian of log(1-P(0)).
// logfmin=true returns the hessian of the transformed variables log10 fmin.
vec2d hess_plaw_poiss_c(const plaw_poiss_pars &pars, const vecui& ns, bool logfmin, int n_points){
    
    unsigned int L = pars.Ms.size();
    double Pn = integr_plaw_poiss_adapt(pars, ns, n_points);
    int z_sign = 1; // special case of sum(ns) = 0
    unsigned int ns_sum  = 0;
    for (unsigned int i=0; i<L; i++) ns_sum += ns[i];
    if (ns_sum == 0){
        Pn = 1-Pn;
        z_sign = -1;
    }

    // First derivatives
    double da = integr_da_plaw_poiss(pars, ns, n_points);
    double df = integr_dfmin_plaw_poiss(pars, ns, n_points);
    double dc = integr_dc_plaw_poiss(pars, ns, n_points);

    // Matrix of the second derivatives of the log of Pn
    vec2d d2log_mat = vec2d(3, vecd(3, 0));
    // d2 log Pn = ( d2_Pn - d_Pn^2 / Pn ) / Pn
    d2log_mat[0][0] = (z_sign * integr_d2a_plaw_poiss(pars, ns, n_points) - da*da / Pn) / Pn;
    d2log_mat[0][1] = (z_sign * integr_d2afmin_plaw_poiss(pars, ns, n_points) - da*df / Pn) / Pn;
    d2log_mat[0][2] = (z_sign * integr_d2ac_plaw_poiss(pars, ns, n_points) - da*dc / Pn) / Pn;
    d2log_mat[1][1] = (z_sign * integr_d2fmin_plaw_poiss(pars, ns, n_points) - df*df / Pn) / Pn;
    d2log_mat[1][2] = (z_sign * integr_d2fminc_plaw_poiss(pars, ns, n_points) - df*dc / Pn) / Pn;
    d2log_mat[2][2] = (z_sign * integr_d2c_plaw_poiss(pars, ns, n_points) - dc*dc / Pn) / Pn;
    
    // Variable transformation: log10 fmin
    if (logfmin){
        double log10 = log(10);
        d2log_mat[0][1] = d2log_mat[0][1] * pars.fmin * log10;
        d2log_mat[1][1] = log10*log10 * pars.fmin * (z_sign * df / Pn + pars.fmin * d2log_mat[1][1]);
        d2log_mat[1][2] = d2log_mat[1][2] * pars.fmin * log10;
    }

    // Copying the symmetric terms
    for (unsigned int i=0; i<3; i++){
        for (unsigned int j=i+1; j<3; j++)
            d2log_mat[j][i] = d2log_mat[i][j];
    }

    return d2log_mat;
}

// Hessian of a logarithmic plaw-poisson integral explicitely computed. In case of sum(ns) == 0, 
// it compute the Hessian of log(1-P(0)).
// logpars=true returns the hessian of the transformed variables log fmin and log M.
vec2d hess_plaw_poiss_M(const plaw_poiss_pars &pars, const vecui& ns, bool logpars){
    
    unsigned int L = pars.Ms.size();
    double Pn = integr_plaw_poiss_adapt(pars, ns);
    int z_sign = 1; // special case of sum(ns) = 0
    unsigned int ns_sum  = 0;
    for (unsigned int i=0; i<L; i++) ns_sum += ns[i];
    if (ns_sum == 0){
        Pn = 1-Pn;
        z_sign = -1;
    }

    // First derivatives
    double da = integr_da_plaw_poiss(pars, ns);
    double df = integr_dfmin_plaw_poiss(pars, ns);
    vecd dMs = vecd(L);
    for (unsigned int i=0; i<L; i++) dMs[i] = integr_dM_plaw_poiss(pars, ns, i);
    
    // Matrix of the second derivatives of the log of Pn
    vec2d d2log_mat = vec2d(2+L, vecd(2+L, 0));
    // d2 log Pn = ( d2_Pn - d_Pn^2 / Pn ) / Pn
    d2log_mat[0][0] = (z_sign * integr_d2a_plaw_poiss(pars, ns) - da*da / Pn) / Pn;
    d2log_mat[1][1] = (z_sign * integr_d2fmin_plaw_poiss(pars, ns) - df*df / Pn) / Pn;
    d2log_mat[0][1] = (z_sign * integr_d2afmin_plaw_poiss(pars, ns) - da*df / Pn) / Pn;
    for (unsigned int i=0; i<L; i++){
        d2log_mat[0][2+i] = (z_sign * integr_d2aM_plaw_poiss(pars, ns, i) - da*dMs[i] / Pn) / Pn;
        d2log_mat[1][2+i] = (z_sign * integr_d2fminM_plaw_poiss(pars, ns, i) - df*dMs[i] / Pn) / Pn;
        for (unsigned int j=i; j<L; j++)
            d2log_mat[i+2][j+2] = (z_sign * integr_d2M_plaw_poiss(pars, ns, i, j) - dMs[i]*dMs[j] / Pn) / Pn;
    }
    
    // Variable transformation: log10 fmin, log10 M
    if (logpars){
        double log10 = log(10);
        d2log_mat[0][1] = d2log_mat[0][1] * pars.fmin * log10;
        d2log_mat[1][1] = log10*log10 * pars.fmin * (z_sign * df / Pn + pars.fmin * d2log_mat[1][1]);
        for (unsigned int i=0; i<L; i++){
            d2log_mat[0][2+i] = d2log_mat[0][2+i] * pars.Ms[i] * log10;
            d2log_mat[1][2+i] = d2log_mat[1][2+i] * pars.fmin * pars.Ms[i] * log10*log10;
            d2log_mat[2+i][2+i] = log10*log10 * pars.Ms[i] * (z_sign * dMs[i] / Pn + pars.Ms[i] * d2log_mat[2+i][2+i]);
            for (unsigned int j=i+1; j<L; j++)
                d2log_mat[2+i][2+j] = d2log_mat[2+i][2+j] * pars.Ms[i] * pars.Ms[j] * log10*log10;
        }
    }

    // Copying the symmetric terms
    for (unsigned int i=0; i<L+2; i++){
        for (unsigned int j=i+1; j<L+2; j++)
            d2log_mat[j][i] = d2log_mat[i][j];
    }
    return d2log_mat;
}


#endif
