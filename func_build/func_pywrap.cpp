#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "src/noise_plaw_poisson.h"
#include "src/noise_plaw_negbin.h"
#include "src/gbm_like.h"
#include "src/memplasm_like.h"


namespace py = pybind11;


// execute from terminal 
// cmake CMakeLists.txt
// cmake --build .


PYBIND11_MODULE(like_func, m) {

    m.doc() = "list of functions for the likelihood computation";


    // Parameter classes

    py::class_<plaw_poiss_pars>(m, "plaw_poiss_pars", "container of power-law poisson integral parameters")
        .def(py::init<float, float, vecd&>(), py::arg("beta"), py::arg("fmin"), py::arg("Ms"))
        .def_readwrite("beta", &plaw_poiss_pars::beta, "power law exponent")
        .def_readwrite("fmin", &plaw_poiss_pars::fmin, "minimal frequency")
        .def_readwrite("Ms", &plaw_poiss_pars::Ms, "total counts");

    py::class_<plaw_negbin_pars>(m, "plaw_negbin_pars", "container of power-law negative binomial integral parameters")
        .def(py::init<float, float, float, float, vecd&>(), py::arg("beta"), py::arg("fmin"), py::arg("a"), py::arg("b"), py::arg("Ms"))
        .def_readwrite("beta", &plaw_negbin_pars::beta, "power law exponent")
        .def_readwrite("fmin", &plaw_negbin_pars::fmin, "minimal frequency")
        .def_readwrite("a", &plaw_negbin_pars::a, "negative binomial param a, from var = mean + a * mean ** b")
        .def_readwrite("b", &plaw_negbin_pars::b, "negative binomial param b, from var = mean + a * mean ** b")
        .def_readwrite("Ms", &plaw_negbin_pars::Ms, "total counts for each sample");
    
    py::class_<gbm_likeMC_pars>(m, "gbm_likeMC_pars", "container of geometric brownian motion parameters")
        .def(py::init<double, double, double, double>(), py::arg("tau"), py::arg("theta"), py::arg("M_tot"), py::arg("n0"))
        .def_readwrite("tau", &gbm_likeMC_pars::tau, "decay time of the GBM")
        .def_readwrite("theta", &gbm_likeMC_pars::theta, "standard deviation of log counts of a GBM")
        .def_readwrite("alpha", &gbm_likeMC_pars::alpha, "Exponent of the cumulative")
        .def_readwrite("M_tot", &gbm_likeMC_pars::M_tot, "")
        .def_readwrite("n0", &gbm_likeMC_pars::n0, "");
        
    py::class_<memplasm_pars>(m, "memplasm_pars", "container of parameters")
        .def(py::init<double, double, double, double, double, double>(), 
            py::arg("tau_m"), py::arg("theta_m"), py::arg("rho"), py::arg("tau_p"), py::arg("theta_p"), py::arg("n0")=0)
        .def_readwrite("tau_m", &memplasm_pars::tau_m, "exp decay time of the memory cells")
        .def_readwrite("theta_m", &memplasm_pars::theta_m, "inverse standard deviation of log counts of the memory")
        .def_readwrite("rho", &memplasm_pars::rho, "differentiation rate from memory to plasmablasts")
        .def_readwrite("tau_p", &memplasm_pars::tau_p, "exp decay time of the plasmablast")
        .def_readwrite("theta_p", &memplasm_pars::theta_p, "inverse standard deviation of log counts of the plasmablasts")
        .def_readwrite("n0", &memplasm_pars::n0, "size of clone at reintroduction");


    // Power law things

    m.def("plaw_norm", &plaw_norm, 
          py::arg("beta"), py::arg("fmin"),
          "Normalization of the power law distribution");

    m.def("plaw_average", &plaw_average, 
          py::arg("beta"), py::arg("fmin"),
          "Average value of the power law distribution");

    
    // Power law-poisson integrals and related

    m.def("plaw_poiss", &plaw_poiss, 
          py::arg("freq"),  py::arg("ns"), py::arg("plaw_poiss_pars"),
          "Integrand of a Poisson sampling noise on power law distribution");

    m.def("integr_plaw_poiss", &integr_plaw_poiss_adapt,
        py::arg("plaw_poiss_pars"), py::arg("ns"), py::arg("n_points")=10000, py::arg("f_zero_th")=1e-6, py::arg("log_x_tolerance")=0.01,
        "Integrate the power-law poisson integral with logarithmic binning and adaptive domain");

    m.def("jac_log_plaw_poiss", &jac_plaw_poisson,
        py::arg("plaw_poiss_pars"), py::arg("ns"), py::arg("logfmin")=true,
        "Jacobian of the log plaw-poisson with respect to beta, fmin, c");

    m.def("hess_log_integr_plaw_poiss", &hess_plaw_poiss_c,
        py::arg("plaw_poiss_pars"), py::arg("ns"), py::arg("logfmin")=true, py::arg("n_points")=50000,
        "Hessian of the logarithm of the plaw-poisson integral computed with xplicit expression of the derivatives. \
        In case of sum(ns) == 0, it computes the Hessian of log(1-P(0)). \
        logfmin=true returns the hessian of the transformed variables log10 fmin. \
        The variable order is beta, fmin, c.");


    // Power law-negbin integrals and related

    m.def("plaw_negbin", &plaw_negbin, 
          py::arg("freq"),  py::arg("ns"), py::arg("plaw_negbin_pars"),
          "Integrand of a negative binomial sampling noise on power law distribution");

    m.def("integr_plaw_negbin", &integr_plaw_negbin_adapt,
        py::arg("plaw_negbin_pars"), py::arg("ns"), py::arg("n_points")=10000, py::arg("f_zero_th")=1e-6, py::arg("log_x_tolerance")=0.01,
        "integrate the power-law negative binomial integral with logarithmic binning and adaptive domain");

    m.def("jac_log_plaw_negbin", &jac_plaw_negbin,
        py::arg("plaw_negbin_pars"), py::arg("ns"), py::arg("logfmin")=true,
        "Jacobian of the log plaw-negbin with respect to beta, fmin, a, b, c");

    m.def("hess_log_integr_plaw_negbin", &hess_plaw_negbin,
        py::arg("plaw_negbin_pars"), py::arg("ns"), py::arg("logpars")=true,
        "Hessian of the logarithm of the plaw-negbin integral. Explicit expression of the derivatives. In case of sum(ns) == 0, it computes the Hessian of log(1-P(0)). logpars=true returns the hessian of the transformed variables log10 fmin and log10 M.");

    m.def("hess_log_integr_plaw_negbin", &hess_plaw_negbin_M,
        py::arg("plaw_negbin_pars"), py::arg("ns"), py::arg("logfmin")=true,
        "Hessian of the logarithm of the plaw-negbin integral computed with xplicit expression of the derivatives. \
        In case of sum(ns) == 0, it computes the Hessian of log(1-P(0)). \
        logfmin=true returns the hessian of the transformed variables log10 fmin. \
        The variable order is beta, fmin, a, b, c.");
        
        
    // Stochastic trajectories generation
    
    m.def("gen_gbm_traj", &gen_gbm_traj, 
          py::arg("tau"), py::arg("theta"), py::arg("xs"), py::arg("delta_t"), py::arg("dt"),
          "Generates the samples of a Brownian motion with absorbing conditions in 0.");

    m.def("gen_gbm_create_traj", &gen_gbm_create_traj, 
          py::arg("tau"), py::arg("theta"), py::arg("n0"), py::arg("s"), py::arg("xs"), py::arg("delta_t"), py::arg("dt"),
          "Generates the samples of a Brownian motion with absorbing conditions in 0. Uniform creation of new clones of size n0 to \
          preserve stationarity.");
          
    m.def("gen_memplasm_traj", &gen_memplasm_traj, 
          py::arg("pars"), py::arg("x0s"), py::arg("n0s"), py::arg("time"), py::arg("dt"), py::arg("seed"),
          "Generates the samples of memory-plasmablast dynamics  with absorbing conditions in 0 given initial conditions of \
          memory log-counts, x0s, and plasmablast counts, n0s. \
          It returns a 2d vector that stores the memory log-counts and the plasmablast counts at the last time point. ");

}
