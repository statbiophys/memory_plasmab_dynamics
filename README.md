## Data analysis and inference in the paper *Quantifying the dynamics of memory B cells and plasmablasts in healthy individuals*

### Folder structure:
 - **Phad data**: it contains all the scripts to download the dataset of https://www.nature.com/articles/s41590-022-01230-1, process the B-cell reperotire data of memory cells and plasmablasts, and infer the different models discussed in out paper.
 - **Mikelov data**: similar type of analysis of **Phad data** but for https://elifesciences.org/articles/79254.
 - **func_py**: phyton functions used in the analysis.
 - **func_build**: c++ functions that are wrapped in python scripts.

### Software required
- **Cell Ranger** (version 7.1.0): for the single-cell data analysis of the Phad dataset.
- **Change-O toolkit** (https://changeo.readthedocs.io/en/stable/): for the Ig sequence alignment using IgBlast (version 1.22.0) of both the datasets.
- **Hilary** (https://github.com/statbiophys/hilary) for the clonal family assignment of both datasets.
- **gcc compiler** (version 11.4.0): for the c++ scripts.
- **gsl libraries** (https://www.gnu.org/software/gsl/) for the c++ scripts.
- **Python 3.8.8** and its standard libraries.
- **pybind11** (https://pybind11.readthedocs.io/en/stable/basics.html) to wrap the cpp functions into python scripts

For any question or further clarification contact me at andrea.mazzolini.90@gmail.com
