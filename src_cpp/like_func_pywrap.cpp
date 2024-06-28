#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "src/noise_plaw_poisson.h"
#include "src/noise_plaw_negbin.h"

namespace py = pybind11;


// execute from terminal 
// cmake CMakeLists.txt
// cmake --build .

PYBIND11_MODULE(like_func, m) {

    m.doc() = "list of functions for the likelihood computation";

    py::class_<plaw_poiss_pars>(m, "plaw_poiss_pars", "container of power-law poisson integral parameters")
        .def(py::init<float, float, vecui&>(), py::arg("alpha"), py::arg("fmin"), py::arg("Ms"))
        .def_readwrite("alpha", &plaw_poiss_pars::alpha, "power law exponent")
        .def_readwrite("fmin", &plaw_poiss_pars::fmin, "minimal frequency")
        .def_readwrite("Ms", &plaw_poiss_pars::Ms, "total counts");

    py::class_<plaw_negbin_pars>(m, "plaw_negbin_pars", "container of power-law negative binomial integral parameters")
        .def(py::init<float, float, float, float, vecui&>(), py::arg("alpha"), py::arg("fmin"), py::arg("a"), py::arg("b"), py::arg("Ms"))
        .def_readwrite("alpha", &plaw_negbin_pars::alpha, "power law exponent")
        .def_readwrite("fmin", &plaw_negbin_pars::fmin, "minimal frequency")
        .def_readwrite("a", &plaw_negbin_pars::a, "negative binomial param a, from var = mean + a * mean ** b")
        .def_readwrite("b", &plaw_negbin_pars::b, "negative binomial param b, from var = mean + a * mean ** b")
        .def_readwrite("Ms", &plaw_negbin_pars::Ms, "total counts for each sample");

    m.def("plaw_norm", &plaw_norm, 
          py::arg("alpha"), py::arg("fmin"),
          "Normalization of the power law distribution");

    m.def("plaw_average", &plaw_average, 
          py::arg("alpha"), py::arg("fmin"),
          "Average value of the power law distribution");

    m.def("plaw_poiss", &plaw_poiss, 
          py::arg("freq"),  py::arg("ns"), py::arg("plaw_poiss_pars"),
          "Integrand of a Poisson sampling noise on power law distribution");

    m.def("plaw_negbin", &plaw_negbin, 
          py::arg("freq"),  py::arg("ns"), py::arg("plaw_poiss_pars"),
          "Integrand of a negative binomial sampling noise on power law distribution");

    m.def("integr_plaw_poiss", &integr_plaw_poiss_adapt,
        py::arg("plaw_poiss_pars"), py::arg("ns"), py::arg("n_points")=10000, py::arg("f_zero_th")=1e-6, py::arg("log_x_tolerance")=0.01,
        "integrate the power-law poisson integral with logarithmic binning and adaptive domain");

    m.def("integr_plaw_negbin", &integr_plaw_negbin_adapt,
        py::arg("plaw_negbin_pars"), py::arg("ns"), py::arg("n_points")=10000, py::arg("f_zero_th")=1e-6, py::arg("log_x_tolerance")=0.01,
        "integrate the power-law negative binomial integral with logarithmic binning and adaptive domain");

    // py::class_<Like_plaw_poiss>(m, "Like_plaw_poiss", "Likelihood computator for a power-law poisson noise model")
    //     .def(py::init<vecui, vecui>(), py::arg("ns_unique"), py::arg("ns_count"))
    //     .def("get_loglike", &Like_plaw_poiss::get_loglike, py::arg("plaw_poiss_pars"))
    //     .def("get_constraint", &Like_plaw_poiss::get_constraint, py::arg("plaw_poiss_pars"));
}