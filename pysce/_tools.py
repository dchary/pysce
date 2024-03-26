####################################################################################################
# # Copyright (C) 2023-Present - Daniel Charytonowicz - All Rights Reserved
# # Written by: Daniel Charytonowicz
# # Contact: daniel.charytonowicz@icahn.mssm.edu
# ###################################################################################################

from typing import Union, Tuple, Optional

from ._utils import load_ppi, preprocess, calc_max_entropy, chunk
from ._dataloaders import EntropyDataset
from ._modules import EntropyModule

from torch.utils.data import Dataset, DataLoader

from anndata import AnnData
import pandas as pd
import numpy as np
import torch

from tqdm.auto import tqdm
    


def score(
    data : Union[AnnData, pd.DataFrame],
    ppi : str = 'scent',
    use_raw : bool = True,
    batch_size : Optional[int] = 8,
    key_added : str = 'entropy',
    artifact_genes = ('RPS','RPL','MT'),
    inplace : bool = True,
    layer : Optional[str] = None,
) -> Optional[AnnData]:
    """
    Score Entropy
    
    Scores entropy for each cell in annotated dataset.
    
    Params
    -------
    data
        Annotated dataset comprising a preprocessed dataset to compute entropy on.
        Can also be a dataframe.
    ppi
        The ppi network to use, currently only 'scent' is supported.
    use_raw
        Whether to use 'adata.raw' for counts source.
    batch_size
        Size of batch to compute entropy with, default is 8 cells. If you experiences
        out of memory errors, try decreasing this value. Increase this value to increase
        GPU utilization, but note memory issues may arise.
    key_added
        The key in 'adata.obs' to add with entropy values computed.
    artifact_genes
        Genes to exclude from entropy calculation.
    inplace
        Compute changes to data inplace
    layer
        Whether to get counts source from a layer in data.
        
    Returns
    -------
    data or None if in-place
    
    """
    
    # Check inputs validity
    assert isinstance(data, (AnnData, pd.DataFrame)), "Input data must be AnnData or DataFrame object."
    assert ppi == 'scent', "PPI must be scent"
    assert batch_size >= 1 if batch_size else True, "Batch size must be >= 1 or None (auto-calculate)"
    
    # Create a reference to the original data object
    data_orig = data
    
    # If passing a dataframe, convert to anndata object
    is_adata = isinstance(data, AnnData)
    data = data if is_adata else AnnData(data)
    
    # If raw exists and is to be used, pass
    if not layer:
        data = data.raw.to_adata() if (use_raw and data.raw) else data
    else:
        data = AnnData(data.layers[layer], obs = data.obs, var = data.var)
    
    # Remove artifact genes
    data = data[:, data.var[~data.var.index.str.startswith(artifact_genes)].index] if artifact_genes else data

    # Load PPI using provided name
    ppi = load_ppi(ppi)
    
    # Preprocess data, PPI, and calculate maximum entropy
    data, ppi = preprocess(data, ppi)
    
    # Calculate maximum entropy as log of right real eigenvalue
    max_entropy = calc_max_entropy(ppi)
    
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get dataset
    dataset = EntropyDataset(data)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size = batch_size)
    
    # Create entropy module
    module = EntropyModule()

    # Assign to devices if available
    module = torch.nn.DataParallel(module, device_ids=[0])
    
    # Make PPI tensor
    ppi_tensor = torch.tensor(ppi.X.toarray()).to(device)
    
    # Buffer to hold scores
    entropy_scores = []

    # Create a progressbar to monitor progress
    with tqdm(total=len(dataloader), unit='batch') as pbar:

        pbar.set_description(f"Scoring Entropy")

        # Iterate each batch in dataloader
        for i, batch in enumerate(dataloader):

            # Push batch into target device
            batch = batch.to(device)

            # Compute entropy
            entropy = module(batch, ppi_tensor, max_entropy)

            # Add to results buffer
            entropy_scores.append(entropy.cpu().numpy())

            # Update progressbar
            pbar.update()
    
    # Merge scores
    entropy_scores = np.concatenate(entropy_scores)
    
    # Add key
    data_orig.obs[key_added] = entropy_scores
    
    # Return if not inplace
    if not inplace:
        return data_orig
