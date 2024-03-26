####################################################################################################
# # Copyright (C) 2024-Present - Daniel Charytonowicz - All Rights Reserved
# # Written by: Daniel Charytonowicz
# # Contact: daniel.charytonowicz@icahn.mssm.edu
# ###################################################################################################

from typing import Union, Tuple, Iterable

from anndata import AnnData, read_h5ad
import scipy.sparse.linalg as la
from scipy.sparse import issparse

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import os


def load_ppi(
    ppi : str
) -> AnnData:
    """\
    Load PPI
    
    Loads a protein-protein interaction dataset based on string identifier.
    
    Params
    -------
    ppi
        Name of protein-protein interaction dataset to use. Loaded
        in format 'ppi_{ppi}_pp_sparse.h5ad'
    
    Returns
    -------
    ppi : anndata.AnnData
        The ppi matrix as a sparse annotated dataset
    """
    
    # Read filepath from local context and then load the corresponding PPI
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"data/ppi_{ppi}_pp_sparse.h5ad")
    return read_h5ad(filepath)


def preprocess(
    data : AnnData,
    ppi : AnnData,
) -> Tuple[AnnData, AnnData]:
    """\
    Data Input and PPI Preprocessing
    
    Converts provided PPI into a fully-connected subgraph required for
    entropy calculation, and re-indexes both PPI and data to match
    shared gene symbols.
    
    Params
    -------
    data
        The annotated data matrix.
    ppi
        The PPI matrix as an annotated data object.
    
    Returns
    -------
    data : anndata.AnnData
        The annotated data matrix with reduced gene symbols to match
        gene symbol intersection with PPI
    ppi : anndata.AnnData
        The protein-protein interaction matrix with genes that
        represent the largest connected subgraph, and those that
        intersect with the input data matrix.
    """
    
    # Find common gene intersection between the two objects
    common_genes = pd.Index(np.intersect1d(data.var_names, ppi.var_names))
    
    # Ensure that there is sufficient overlap
    assert len(common_genes) > 1000, "Low number of genes overlapping reference PPI and target data"
    
    # Re-index keeping only common genes
    ppi = ppi[common_genes, common_genes]
    data = data[:, common_genes]
    
    # use networkX to find the largest connected subgraph
    gr = nx.Graph(zip(ppi.X.tocoo().row, ppi.X.tocoo().col))
    gr = nx.relabel_nodes(gr, dict(enumerate(ppi.var_names)))
    largest_clust_genes = sorted(max(nx.connected_components(gr), key=len))
    
    # Re-index keeping only largest cluster genes
    ppi = ppi[largest_clust_genes, largest_clust_genes]
    data = data[:, largest_clust_genes].copy()
    
    # Intvert data
    if data.X.max() < 30:
        print("Data is log-scaled, inverting")
        data.X = np.expm1(data.X)
    
    print("Normalizing data")
    sc.pp.normalize_total(data, 1e4)
    sc.pp.log1p(data)

    return data, ppi


def calc_max_entropy(
    ppi : AnnData
) -> float:
    """\
    Calculate maximum entropy.
    
    Calculates maximal entropy given a PPI matrix as a
    sparse anndata object. Calculated by taking the
    log of the right real eigenvalue.
    
    Params
    -------
    ppi
        The PPI matrix as an annotated data object.
    
    Returns
    -------
    max_entropy : float
        Maximum entropy of the given ppi matrix
    """
    
    # Calculate maximum entropy as log of right real eigenvalue
    eig_val, _ = la.eigs(ppi.X, k=1, which='LM')    
    return np.log(float(eig_val.real))


def chunk(
    obj : Iterable,
    chunksize : int = None
) -> Iterable:
    """\
    Iterable chunking
    
    Chunks an iterable (obj) into blocks of chunksize, 
    if the length of the last yielded value of this
    generaotr is smaller than the chunksize, then
    it will return a smaller chunk.
    
    Params
    -------
    obj
        Iterable-like object that can be chunked and
        has a length
    chunksize
        size of chunks to return
        
    Returns
    -------
    chunk : Iterable
        chunk of the underlying iterable
        
    """
    
    # Get chunksize, if not default to length of object
    chunksize = chunksize if chunksize else len(obj)
    
    # Return block of iterable object in chunks
    for i in range(0, len(obj) if not issparse(obj) else obj.shape[0], chunksize):
        yield obj[i : i + chunksize]
