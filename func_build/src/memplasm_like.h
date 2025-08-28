#ifndef MEMPLASM
#define MEMPLASM


class memplasm_pars {

    public:

    memplasm_pars(double tau_m, double theta_m, double rho, double tau_p, double theta_p, double n0=0){
        this->tau_m = tau_m;
        this->theta_m = theta_m;
        this->rho = rho;
        this->tau_p = tau_p;
        this->theta_p = theta_p;
        this->n0 = n0;

        x0 = log(n0);
    }

    double tau_m;
    double theta_m;
    double rho;
    double tau_p;
    double theta_p;
    double n0;
    double x0;
};




vecd gen_one_memplasm_traj(const memplasm_pars &pars, double x, double n, unsigned int n_steps, double dt, const gsl_rng *r) {

    double dmu_m = -dt / pars.tau_m;
    double dsigma_m = sqrt(dt / pars.theta_m);
    double dsigma_p = sqrt(dt / pars.theta_p);
    bool just_extincted = false;

    for(unsigned int ti = 0; ti < n_steps; ti++){ // Iter over time
            
        if (n < 1){
            if (just_extincted){ // extinction
                just_extincted = false;
                n = 0;
            }
            // while extincted, prob of reintroduction given by exp(x) * rho * dt
            // we deterministically reintroduce after the cumulation of that prob reaches 1
            n += exp(x) * pars.rho * dt;
            if (n >= 1) just_extincted = true;
        }
        else
            n += (exp(x) * pars.rho - n / pars.tau_p) * dt + gsl_ran_gaussian_ziggurat(r, n * dsigma_p);

        x += dmu_m + gsl_ran_gaussian_ziggurat(r, dsigma_m);

        if (x < 0 && n < 1){ // Complete extinction
            n = 0; 
            x = -1;
            break;
        }
    }

    if (x < 0) x = -1;
    if (n < 0) n = 0; // We keep values smaller than 1 because they can be important as init cond of subsequent process
    vecd result = {x, n};
    return result;
}


// Generation of memory and plasmablast trajectories given:
//
// pars: memplasm_pars, if proper values of N_cells and n0 are specified, the source rate will be larger than zero
//       and new memory clones will be created
//
// x0_m: initial condition of memory cell log counts. This sets also how many trajectories are generated
//
// n0_p: initial conditions of plasmablast counts. If vector is of length 0 or different from x0_m, stationary init 
//       counts are generated
// 
// time, dt, seed: total time of the trajectory, temporal resolution of the process, random seed.
//
// It returns the final values of the memory cell log counts and plasmablast counts. If new cells are generated
vec2d gen_memplasm_traj(const memplasm_pars &pars, const vecd &x0_m, const vecd &n0_p, double time, double dt, unsigned int seed = 0) {

    unsigned int R = x0_m.size();
    bool plasm_init = true;
    if (n0_p.size() != R) plasm_init = false;

    vecd x_m = vecd(R); // Log number of memory to return at final time
    vecd n_p = vecd(R); // Number of plasmablasts to return at final time

    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, seed);

    unsigned int n_steps = time / dt;
    for (unsigned int i = 0; i < R; i++){ // Iter over clones

        // Initial conditions
        double x = x0_m[i];
        double n;
        if (plasm_init) n = n0_p[i];
        else n = exp(x) * pars.rho * pars.tau_p; // Initial condition for stat dist
        vecd result = gen_one_memplasm_traj(pars, x, n, n_steps, dt, r);
        x_m[i] = result[0];
        n_p[i] = result[1];
    }

    // Creation of new clones
    if (pars.n0 > 0){
        double s = R / log(pars.n0) / pars.tau_m;
        unsigned int n_new_clones = gsl_ran_poisson(r, s * time);
        std::cout << n_new_clones << std::endl;
        vecd x_new_m = vecd(n_new_clones);
        vecd n_new_p = vecd(n_new_clones);
        for (unsigned int i = 0; i < n_new_clones; i++){
            double life_time = gsl_ran_flat(r, 0, time);
            n_steps = life_time / dt;
            double n = 1;
            vecd result = gen_one_memplasm_traj(pars, pars.x0, n, n_steps, dt, r);
            x_new_m[i] = result[0];
            n_new_p[i] = result[1];
        }
        x_m.insert(std::end(x_m), std::begin(x_new_m), std::end(x_new_m));
        n_p.insert(std::end(n_p), std::begin(n_new_p), std::end(n_new_p));
    }
    
    vec2d result = {x_m, n_p};
    return result;
}


#endif
