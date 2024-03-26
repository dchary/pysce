####################################################################################################
# # Copyright (C) 2024-Present - Daniel Charytonowicz - All Rights Reserved
# # Written by: Daniel Charytonowicz
# # Contact: daniel.charytonowicz@icahn.mssm.edu
# ###################################################################################################

import torch

class EntropyModule(torch.nn.Module):
    """\
    
    Entropy Scoring Module
    
    This is the base module that contains an entropy scoring forward function.
    
    Params
    -------
    
    """
    def __init__(self):
        super().__init__()
        
    @torch.compile
    def forward(self, 
        exp : torch.tensor, 
        ppi : torch.tensor,
        max_entropy : float, 
        expand : bool = False
        ) -> torch.tensor:
        """\
        Forward Function To Calculate Entropy Score
        
        """
        
        # If batch size is 1 and expression is a vector then expand dimensions
        exp = exp.unsqueeze(0) if expand else exp

        # Transpose expression tensor to shape (genes, batch_size)
        exp = exp.t()

        # Calculate stationary distribution used later when computing
        # the final Markov chain entropy
        m_exp = exp * torch.matmul(ppi, exp)
        m_exp = m_exp / m_exp.sum(0)

        # Map cell expression onto the PPI
        pm = ppi * (exp.unsqueeze(0).permute([2, 1, 0]) * exp.unsqueeze(0).permute([2, 0, 1]))

        # Normalize each column to create a markov chain
        pm = torch.nan_to_num(pm / pm.sum(1, keepdim = True))

        # Calculate entropy score
        entropy = -torch.sum(torch.nan_to_num(pm.log() * pm), 1, keepdim = True)
        entropy = torch.reshape(entropy, shape = [entropy.shape[0], entropy.shape[2]]).t()
        entropy = torch.sum(m_exp * entropy, 0, keepdim = True)

        # Normalize entropy
        entropy = torch.squeeze(entropy / max_entropy)

        return entropy
