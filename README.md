[![Documentation Status](https://readthedocs.org/projects/pysce/badge/?version=latest)](https://pysce.readthedocs.io/en/latest/?badge=latest)

pySCE: Single Cell Entropy Scoring in Python
=============================================

pySCE is a python package for calculating transcriptional entropy in single cell RNA-Seq data. Implementation was inspired by the original [Single Cell Entropy](https://github.com/aet21/SCENT) package introduced by [Teschendorff et. al](https://www.nature.com/articles/ncomms15599). pySCE expands the throughput capabilities of entropy scoring as a computational tool for pseudotime analysis of single cell data by leveraging tensorflow for major compute operations, allowing GPUs as well as google TPUs to be used for analysis. This shift, coupled with the computational efficiency of python scientific computing libraries (numpy, scipy, etc..) enable significant performance improvements that allow for exact entropy calculation for hundreds of thousands of cells in minimal time.

