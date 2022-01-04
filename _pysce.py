from anndata import AnnData, read_h5ad
from typing import Union, Optional
import tensorflow as tf
import pandas as pd
import numpy as np
import gc

from ._utils import load_ppi, preprocess, calc_max_entropy, get_strategy, get_batch_size
from ._utils import spmatrix_to_sparsetensor, create_dataset, batch_compute_entropy, run_entropy_compute_strategy


def score_entropy(
    data : Union[AnnData, pd.DataFrame],
    ppi : str = 'scent',
    system : str = 'AUTO',
    use_raw : bool = True,
    batch_size : Optional[int] = None,
    batch_size_mem_overhead : float = 0.3,
    key_added : str = 'entropy',
    tpu_kwargs : dict = None,
    serialize_data_matrix : bool = False,
    artifact_genes = ('RPS','RPL','MT'),
    inplace : bool = True
) -> Optional[AnnData]:
    """\
    Scores entropy for cells.
    
    Scores entropy for cells using the approach proposed by [Tessendorf17],
    using an updated implementation that takes advantage of GPU acceleration
    and python scientific computing packages.
    
    Params
    -------
    data
        The annotated data matrix. Optionally can pass a DataFrame of
        rows (cells) x columns (genes) with gene symbols.
    ppi
        Name of protein-protein interaction dataset to use, currently we only 
        support four different ones, scent, inbio, biogrid, and huri.
    system
        Which system to use for computation, either CPU, GPU, TPU, or 'AUTO'.
    use_raw
        Use raw representation of all genes for entropy calculation (reccomended).
    batch_size
        Number of cells to compute entropy for in batch. If None, is calculated
        automatically using a heuristic function.
    batch_size_mem_overhead
        Percent of free memory on system device (expressed as fraction) to keep
        free when automatically calcuating batch_size. If batch_size provided,
        this is not used.
    key_added
        'adata.obs' key under which to add entropy results
    tpu_kwargs
        dict of args (tpu/zone/project) needed to resolve TPU (optional),
        only required if using a TPU system.
    serialize_data_matrix
        override automatic check and serialize entire matrix into a tensor
        instead of reading from generator.
    artificant_genes
        tuple of gene name stems to ignore from PPI and data, if None, skip
    inplace
        return passed adata if True, otherwise just append obs and return
        None. if passed data is a DataFrame, ignored and values are just
        returned directly.
        
    Returns
    -------
    'adata.obs[key_added]'
        Array of dim (number of samples) that contains normalized entropy
        scores for each cell.
        
    """
    
    # Check inputs validity
    assert isinstance(data, (AnnData, pd.DataFrame)), "Input data must be AnnData or DataFrame object."
    assert ppi in ['scent','inbio','biogrid','huri'], "PPI must be scent/inbio/bigrid/huri."
    assert system in ['CPU', 'GPU', 'TPU','AUTO'], "System must be CPU/GPU/TPU/AUTO."
    assert batch_size >= 1 if batch_size else True, "Batch size must be >= 1 or None (auto-calculate)"
    
    # Create a reference to the original data object
    data_orig = data
    
    # If passing a dataframe, convert to anndata object
    is_adata = isinstance(data, AnnData)
    data = data if is_adata else AnnData(data)
    
    # If raw exists and is to be used, pass
    data = data.raw.to_adata() if (use_raw and data.raw) else data
    
    # Remove artifact genes
    data = data[:, data.var[~data.var.index.str.startswith(artifact_genes)].index] if artifact_genes else data
    
    # Load PPI using provided name
    ppi = load_ppi(ppi)
    
    # Preprocess data, PPI, and calculate maximum entropy
    data, ppi = preprocess(data, ppi)
    
    # Calculate maximum entropy as log of right real eigenvalue
    max_entropy = calc_max_entropy(ppi)
    
    # Get a tensorflow training strategy, devices, and system ID
    strategy, num_devices, system = get_strategy(system, tpu_kwargs)
    
    # Calculate optimal batch size for the given system being used
    batch_size = batch_size if batch_size else get_batch_size(ppi.shape, system)
    
    # Create dataset, apply batch size and distrbute across compute strategy
    dataset, dataset_host = create_dataset(data, system, serialize_data_matrix)
    dataset = dataset.batch(num_devices * batch_size)
    dataset = strategy.experimental_distribute_dataset(dataset)
    
    # Convert ppi and max_entropy to tensors
    ppi_tensor = tf.convert_to_tensor(ppi.X.todense(), dtype = tf.float32)
    max_entropy_tensor = tf.convert_to_tensor(max_entropy, dtype = tf.float32)
    
    # Run entropy scoring
    entropy_scores = run_entropy_compute_strategy(dataset, ppi_tensor, strategy, 
                            max_entropy_tensor, data.shape[0], num_devices, batch_size)
    
    # Post-run memory cleanup routine
    del dataset, ppi, max_entropy, strategy, batch_size, num_devices
    del ppi_tensor, max_entropy_tensor
    dataset_host.cleanup()
    del dataset_host
    gc.collect()
    
    # Add scores to obs at key_added if original data is adata, otherwise return
    if is_adata: 
        data_orig.obs[key_added] = entropy_scores
        return None if inplace else data_orig
    else:
        return entropy_scores