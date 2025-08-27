#ifndef GBMLIKE
#define GBMLIKE


// Functions for computing the likelihood of samples of propagating geometric brownian motion


//#include"trapezoid.h"
//#include "gsl/gsl_sf_erf.h"


struct gbm_likeMC_pars {

    gbm_likeMC_pars(double tau, double theta, double M_tot, double n0){
        this->tau = tau;
        this->theta = theta;
        this->M_tot = M_tot;
        this->n0 = n0;
        alpha = 2 * theta / tau; // The power law exponent is 1 + alpha!

    }

    double tau;
    double theta;
    double alpha;
    double n0;
    double M_tot;
};


double gen_one_gbm_traj(const double tau, const double theta, double x0, unsigned int n_steps, double dt, const gsl_rng *r) {

    double dmu = -dt / tau;
    double dtheta = sqrt(dt / theta);
    double x = x0;
    for(unsigned int ti = 0; ti < n_steps; ti++){
        x += dmu + gsl_ran_gaussian_ziggurat(r, dtheta);
        if (x <= 0){
            x = 0;
            break;
        }
    }
    return x;
}


vecd gen_gbm_traj(const double tau, const double theta, vecd &xs, double delta_t, double dt) {

    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    unsigned int n_steps = delta_t / dt;
    for (unsigned int i = 0; i < xs.size(); i++)
        xs[i] = gen_one_gbm_traj(tau, theta, xs[i], n_steps, dt, r);
    return xs;
}


vecd gen_gbm_create_traj(const double tau, const double theta, const double n0, const double s, vecd &xs, double delta_t, double dt) {

    //int N_cells = 0;
    //for (unsigned int i = 0; i < xs.size(); i++) N_cells += exp(xs[i]);

    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    unsigned int n_steps = delta_t / dt;
    for (unsigned int i = 0; i < xs.size(); i++)
        xs[i] = gen_one_gbm_traj(tau, theta, xs[i], n_steps, dt, r);

    // Creation of new clones
    if (n0 > 0){
        unsigned int n_new_clones = gsl_ran_poisson(r, s * delta_t);
        //std::cout << s << " " << N_cells << " " << (1/tau - 1/theta/2.0) << std::endl;
        vecd x_new = vecd(n_new_clones);
        for (unsigned int i = 0; i < n_new_clones; i++){
            double life_time = gsl_ran_flat(r, 0, delta_t);
            unsigned int n_steps = life_time / dt;
            x_new[i] = gen_one_gbm_traj(tau, theta, log(n0), n_steps, dt, r);
        }
        xs.insert(std::end(xs), std::begin(x_new), std::end(x_new));
    }

    return xs;
}

#endif
