[![Documentation Status](https://readthedocs.org/projects/pysce/badge/?version=latest)](https://pysce.readthedocs.io/en/latest/?badge=latest)

pySCE: Single Cell Entropy Scoring in Python
=============================================

![image](https://user-images.githubusercontent.com/7418190/148014006-1c522756-05b2-4e6f-9ab5-dec627405b57.png)

## Overview
<p align="justify">
  pySCE is a python package for calculating transcriptional entropy in single cell RNA-Seq data. Implementation was inspired by the original <a href="https://github.com/aet21/SCENT">Single Cell Entropy</a> package introduced by <a href="https://www.nature.com/articles/ncomms15599">Teschendorff et. al</a>. pySCE expands the throughput capabilities of entropy scoring as a computational tool for pseudotime analysis of single cell data by leveraging tensorflow for major compute operations, allowing GPUs as well as google TPUs to be used for analysis. This shift, coupled with the computational efficiency of python scientific computing libraries (numpy, scipy, etc..) enable significant performance improvements that allow for exact entropy calculation for hundreds of thousands of cells in minimal time.
</p>
  
