# OOP Boolean Networks RNA miRNA

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This is a project to implement a Boolean Network system and related behaviors such as differential equations to calculate RNA abundances and miRNA effects on RNA abundances.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is reference to classes, or descriptive
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         oop_boolean_networks_rna_mirna and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── environment.yml   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── oop_boolean_networks_rna_mirna   <- Source code for use in this project.
    │
    ├── __init__.py
    |
    ├── abn_mir_helper_functions.py
    |
    ├── abn_mir_plotting_functions.py
    |
    ├── abundant_boolean_networks_with_micro_rna.py
    |
    ├── abundant_boolean_networks.py
    |
    ├── bn_graph_methods.py
    |
    ├── bn_mir_helper_functions_V1.py
    |
    ├── boolean_networks.py
    |
    ├── helper.py
    |
    ├── micro_rna_2.py
    |
    ├── mirna_demo_methods.py
    |
    ├── nx_docs.py
    |
    ├── plotting_helper_methods.py
    |
    ├── plotting_revised.py
    |
    ├── plotting.py
    |
    ├── seq_pert_as_py.py
    |
    ├── sequential_perturbations_methods.py
    |
    ├── widgets_dev.ipynb
```

--------

Figures with same axes as some of the figures in $\text{Kauffman}$ $1969$, from `BN.ipynb`


<p align="center">
  <img src="https://github.com/AlanBeem/oop_boolean_networks_rna_mirna/blob/main/reports/figures/bn/re_kauffman_1.png" alt="Alt text" />
</p>


<p align="center">
  <img src="https://github.com/AlanBeem/oop_boolean_networks_rna_mirna/blob/main/reports/figures/bn/re_kauffman_2.png" alt="Alt text" />
</p>

<p align="center">
  <img src="https://github.com/AlanBeem/oop_boolean_networks_rna_mirna/blob/main/reports/figures/bn/re_kauffman_3.png" alt="Alt text" />
</p>
