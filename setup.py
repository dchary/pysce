####################################################################################################
# # Copyright (C) 2024-Present - Daniel Charytonowicz - All Rights Reserved
# # Contact: daniel.charytonowicz@icahn.mssm.edu
# ###################################################################################################

from setuptools import setup, find_packages
from pkg_resources import parse_requirements
import pathlib

# Read requirements text file to get required packages
with pathlib.Path('requirements.txt').open() as reqsfile:
    reqs = [str(req) for req in parse_requirements(reqsfile)]

# Setup package with params
setup(name='pysce',
      version='0.1.0',
      description='pySCE: Single Cell Entropy Scoring in Python',
      url='https://github.com/dchary/pysce',
      author='Daniel Charytonowicz',
      author_email='daniel.charytonowicz@icahn.mssm.edu',
      license='GNU GPLv3',
      install_requires=reqs,
      packages=find_packages(),
      package_data={'pysce': ['data/ppi_scent_pp_sparse.h5ad']},
      include_package_data = True,
      zip_safe=False)