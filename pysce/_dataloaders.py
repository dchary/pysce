####################################################################################################
# # Copyright (C) 2024-Present - Daniel Charytonowicz - All Rights Reserved
# # Written by: Daniel Charytonowicz
# # Contact: daniel.charytonowicz@icahn.mssm.edu
# ###################################################################################################

import torch
from torch.utils.data import Dataset
from anndata import AnnData
from scipy.sparse import issparse
import numpy as np

class EntropyDataset(Dataset):
    """
    EntropyDataset
    fafaf
    A Dataloader for anndata objects
    
    Params
    -------
    adata
        Annotated dataset comprising a preprocessed dataset to compute entropy on.
    """
    
    def __init__(self, adata : AnnData):
        self.data = adata.X
                
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        
        # Get vector at index
        exp = self.data[idx]
        
        # convert from sparse to dense if desired
        exp = exp.toarray() if issparse(exp) else exp
        
        # Flatten
        exp = exp.flatten()
        
        # Cast to tensor and yield
        return torch.tensor(exp)
        
