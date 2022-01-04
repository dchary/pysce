from tensorflow.distribute import OneDeviceStrategy, MirroredStrategy, TPUStrategy, Strategy
from tensorflow.config import list_logical_devices, experimental_connect_to_cluster
from tensorflow.config import PhysicalDevice, LogicalDevice
from tensorflow.compat.v1 import get_default_graph
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.tpu.experimental import initialize_tpu_system
from tensorflow.data import Dataset
from tensorflow import SparseTensor
from tensorflow.data import Dataset
from tensorflow.sparse import to_dense
from tensorflow import float32, TensorShape

from progressbar import progressbar
from itertools import cycle
from more_itertools import collapse
from anndata import AnnData, read_h5ad
from typing import Union, Tuple
import subprocess as sp
import scipy.sparse.linalg as la
from scipy.sparse import spmatrix, issparse
import tensorflow as tf
import networkx as nx
import pandas as pd
import numpy as np
import psutil
import math
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
                            f"datasets/ppi_{ppi}_pp_sparse.h5ad")
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
    
    # Re-index keeping only common genes
    ppi = ppi[common_genes, common_genes]
    data = data[:, common_genes]
    
    # use networkX to find the largest connected subgraph
    gr = nx.Graph(zip(ppi.X.tocoo().row, ppi.X.tocoo().col))
    gr = nx.relabel_nodes(gr, dict(enumerate(ppi.var_names)))
    largest_clust_genes = sorted(max(nx.connected_components(gr), key=len))
    
    # Re-index keeping only largest cluster genes
    ppi = ppi[largest_clust_genes, largest_clust_genes]
    data = data[:, largest_clust_genes]

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
    
    
def get_strategy(
    system : str = 'CPU', 
    tpu_kwargs : dict = None,
) -> Tuple[Strategy, int, str]:
    """\
    Gets Tensorflow Strategy.
    
    Instantiate the appropriate tensorflow compute strategy given the
    desired input.
    
    Params
    -------
    system
        name of system to compute on, CPU/GPU/TPU only
    tpu_kwargs
        dict of args (tpu/zone/project) needed to resolve TPU (optional)
    
    Returns
    -------
    strategy : tensorflow.distribute.Strategy
        The strategy to use
    num_devices : int
        Number of devices in strategy, useful for batch size calculation
    system : str
        Name of system, useful when system is AUTO and we need to know
        what has been decided as the optimal device to use.
    """
    
    # If automatic, attempt to use a GPU, if absent will default to CPU
    if system == 'AUTO':
        return get_strategy('GPU')
    
    # If using CPU, distribute to single device across all cores
    if system == 'CPU':
        strategy = OneDeviceStrategy(device = "/CPU:0")
        num_devices = 1
        
    # If using GPU, find all logical devices and distrubte across each
    elif system == 'GPU':
        logical_gpus = list_logical_devices('GPU')
        logical_gpus = [gpu.name.lstrip('/device:') for gpu in logical_gpus]
        
        # If no GPUs found, default back to CPU strategy.
        if len(logical_gpus) == 0: return get_strategy('CPU')
        
        strategy = MirroredStrategy(logical_gpus)
        num_devices = len(logical_gpus)
        
    # If using TPU, connect using provided kwargs
    elif system == 'TPU':
        resolver = TPUClusterResolver(**tpu_kwargs)
        experimental_connect_to_cluster(resolver)
        initialize_tpu_system(resolver)
        
        strategy = TPUStrategy(resolver)
        num_devices = 8
    
    # If any other provided device, raise valueErorr
    else:
        raise ValueError("System must be CPU/GPU/TPU")
    
    # Return the requested strategy
    return strategy, num_devices, system


def get_meminfo(
    device : Union[PhysicalDevice, LogicalDevice],
    query : str = 'free'
) -> int:
    """\
    Get memory information for CPU or GPU (requires nvidia-smi to be installed)
    
    Params
    -------
    device
        A LogicalDevice or PhysicalDevice object corresponding to a GPU or CPU.
    query
        A string either total/free/used corresponding to the memory
        information we want returned.
        
    Returns
    -------
    meminfo_values
        Amount of memory in bytes.
    """
    
    # Ensure data input is correct
    assert isinstance(device, (PhysicalDevice, LogicalDevice)), "device must be Logical or PhysicalDevice."
    assert device.device_type in ['GPU', 'CPU'], "device must be a GPU or CPU"
    assert query in ['total', 'free', 'used'], "Unsupported memory query, must be total / free / used."
    
    # If device is a CPU, default back to psutil
    if device.device_type == 'CPU':
        vmem = psutil.virtual_memory()
        return {'total' : vmem.total, 'free' : vmem.free, 'used' : vmem.active}.get(query)
    
    # Extract the device id
    device_id = device.name.rsplit(":",1)[-1]
    
    # Run nvidia-smi in subprocess and extract memory info desired
    command = f"nvidia-smi -i {device_id} --query-gpu=memory.{query} --format=csv"
    meminfo = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    meminfo_values = [int(x.split()[0]) for i, x in enumerate(meminfo)][0] * (1024**2)
    
    return meminfo_values


def get_batch_size(
    ppi_shape : Tuple[int,int],
    system : str,
    memory_overhead_ratio : float = 0.3
) -> int:
    """\
    Compute optimal batch size
    
    Computes optimal batch size for given system settings and available memory.
    If running on CPU, estimates maximal memory use during runtime and sets
    batch size to keep memory peak at level with some overhead remaining, defaults
    to 30% overhead of currently free memory, calculated at runtime.
    
    Params
    -------
    ppi_shape
        shape of the PPI being used after preprocessing, primary determinant of
        memory usage requirements.
    system
        desired system for calculations, will affect which memory limit is needed
        to be read.
    memory_overhead_ratio
        Percentage of free memory at program runtime to keep unallocated, should
        typically be between 20 - 30% as memory usage can be unpredictable.
    
    Returns
    -------
    batch_size
        Optimal batch size as an integer to use per device. Returns the lowest
        next power of 2 value.
    """
    
    # Ensure overhead is between 0 and 1
    assert 0.0 < memory_overhead_ratio < 1.0, "Memory overhead must be between 0 and 1"
    assert system in ['GPU', 'CPU'], "system must be a GPU or CPU"

    # get all logical devices corresponding to the desired system
    devices = filter(lambda d : d.device_type == system, list_logical_devices())
    
    # Get system memory, which corresonds to the device with the least amount of free memory.
    sysmem = min(map(lambda x : get_meminfo(x, 'used' if system == 'GPU' else 'free'), devices))
    
    # Get the lowest power of 2 value
    batch_size = math.floor((((1 - memory_overhead_ratio)) * sysmem) 
                                // (4 * 4 * (ppi_shape[0] * ppi_shape[1])))
    
    return batch_size


def spmatrix_to_sparsetensor(
    data : spmatrix
) -> SparseTensor:
    
    """\
    Converts spmatrix to SparseTensor
    
    Params
    -------
    data
        spmatrix to convert to tensor
    
    Returns
    -------
    sparsedata : tf.SparseTensor
        a sparse tensor to return
    """
    
    # ensure sparse matrix
    assert isinstance(data, spmatrix), "Data must be spmatrix"
    
    # Convert to sparse tensor
    data_coo = data.tocoo()
    indices = np.vstack([data_coo.row, data_coo.col]).T
    sparsedata = SparseTensor(indices, data_coo.data, data_coo.shape)
    
    return sparsedata


def dataset_generator(
    arr : Union[np.ndarray, spmatrix],
) -> object:
    """\
    Create a Dataset Generator from underlying array-like, sparse or dense
    
    Params
    -------
    arr
        array-like object, sparse or dense
    
    Returns
    -------
    _gen
        a generator object that will yield rows of arr as dense rows
        
    """
    
    # Check if arr is sparse
    is_sparse_arr = issparse(arr)
    
    # Define generator object
    def _gen():
        for row in arr:
            yield np.asarray(row.todense()).flatten() if is_sparse_arr else row
            
    return _gen


class TFDatasetHost(object):
    """
    Wrapper class for TensorFlow Dataset
    
    Currently tf.data.Dataset has the tendency to experience memory leaks when running
    in a persistent environment (i.e. jupyter notebook). This wrapper [kkimdev2020]
    serves to ensure proper garbage collection after each call.
    
    Not fully documented below.
    
    """
    def __init__(self):
        self.py_func_set_to_cleanup = set()
        
    def from_tensor_slices(self, tensors):
        if not hasattr(get_default_graph(), '_py_funcs_used_in_graph'):
            get_default_graph()._py_funcs_used_in_graph = []
        py_func_set_before = set(get_default_graph()._py_funcs_used_in_graph)
        result = Dataset.from_tensor_slices(tensors)
        py_func_set_after = set(get_default_graph()._py_funcs_used_in_graph) - py_func_set_before
        self.py_func_set_to_cleanup |= py_func_set_after
        return result

    def from_generator(self, generator, output_types, output_shapes=None, args=None):
        if not hasattr(get_default_graph(), '_py_funcs_used_in_graph'):
            get_default_graph()._py_funcs_used_in_graph = []
        py_func_set_before = set(get_default_graph()._py_funcs_used_in_graph)
        result = Dataset.from_generator(generator, output_types, output_shapes, args)
        py_func_set_after = set(get_default_graph()._py_funcs_used_in_graph) - py_func_set_before
        self.py_func_set_to_cleanup |= py_func_set_after
        return result
  
    def cleanup(self):
        """
        Should be called after done using generator
        """
        new_py_funcs = set(get_default_graph()._py_funcs_used_in_graph) - self.py_func_set_to_cleanup
        get_default_graph()._py_funcs_used_in_graph = list(new_py_funcs)
        self.py_func_set_to_cleanup = set()

def create_dataset(
    data : AnnData,
    system : str,
    serialize_data_matrix : bool = False
) -> Dataset:
    """\
    Create a TF Dataset
    
    Creates a tf.data.Dataset from underlying preprocessed anndata object.
    Depending on the system configuration, will either use a generator
    to dynamically pull data (more memory efficient), or if TPU, will
    convert entire dataset into a SparseTensor, and index it from there.
    
    Params
    -------
    data
        Preprocessed annotated data object
    system
        System we are using for computation of entropy
    serialize_data_matrix
        override automatic check and serialize entire matrix
        even if using GPU
        
    Returns
    -------
    dataset : tf.data.Dataset
        a dataset object that iterates the entire data array.
        
    dataset_host : TFDatasetHost
        a host wrapper that can be used to properly garbage collect
    """
    
    # Create a TFDatasetHost wrapper
    dataset_host = TFDatasetHost()
    
    # If system is TPU, it cannot accept a generator as input, so we must
    # serialize the entire matrix into a sparse tensor, and feed that
    # directly. This approach requires the entire sparse tensor to be
    # copied in memory.
    if (system == 'TPU') or serialize_data_matrix:
        spdata = spmatrix_to_sparsetensor(data.X) if issparse(data.X) else data.X
        dataset = dataset_host.from_tensor_slices(spdata)
        if issparse(data.X): dataset = dataset.map(lambda t : to_dense(t))

    # in all other cases we can just feed a generator directly from the data
    # object itself.
    else:
        dataset = dataset_host.from_generator(dataset_generator(data.X),
            output_types = float32, output_shapes = TensorShape([data.shape[1]]))
    
    return dataset, dataset_host


'''
Executes with an additional batch dimension to speed up processing
'''
@tf.function
def batch_compute_entropy(
    exp : tf.Tensor,
    ppi : tf.Tensor,
    max_entropy : tf.Tensor,
    expand : bool = False
) -> tf.Tensor:
    """\
    Compute Transcriptome Entropy
    
    Get entropy score for a cell or batch of cells.
    
    Params
    -------
    exp
        expression vector obs x genes where obs is the batch size
        and genes is the number of genes in the vector.
    ppi
        a matrix of size genes x genes representing the PPI adjacency
        matrix
    max_entropy
        a singular tf.constant representing the maximum possible entropy
        for the given ppi
    expand
        whether or not to expand the batch dimension, used in cases
        where the batch size is 1.
        
    Returns
    -------
    entropy
        normalized entropy value per cell
    
    """

    # If batch size is 1 and expression is a vector then expand dimensions
    exp = tf.expand_dims(exp, axis = 0) if expand else exp

    # Transpose
    exp = tf.transpose(exp)
    
    # Calculate stationary distribution used later when computing
    # the final markov chain entropy, start by multiplying expression
    # by the ppi matrix, and normalizing by row, per batch
    m_exp = tf.multiply(exp, tf.raw_ops.BatchMatMulV2(x = ppi, y = exp))
    m_exp = tf.divide(m_exp, tf.reduce_sum(m_exp, axis = 0, keepdims = True))

    # Map cell expression onto the PPI
    pm = tf.multiply(ppi, tf.multiply(
            tf.transpose(tf.expand_dims(exp, axis = 0), [2, 1, 0]),
            tf.transpose(tf.expand_dims(exp, axis = 0), [2, 0, 1])))

    # Normalize each column to create a markov chain
    pm = tf.math.divide_no_nan(pm, tf.reduce_sum(pm, axis = 1, keepdims = True))
    
    # Calculate entropy score
    entropy = -tf.reduce_sum(tf.math.multiply_no_nan(tf.math.log(pm), pm), axis = 1, keepdims = True)
    entropy = tf.transpose(tf.reshape(entropy, shape = [entropy.shape[0], entropy.shape[2]]))
    entropy = tf.reduce_sum(tf.multiply(m_exp, entropy), axis = 0, keepdims = True)
    
    # Normalize entropy
    entropy = tf.squeeze(tf.divide(entropy, max_entropy))

    return entropy


def run_entropy_compute_strategy(
    dataset : tf.data.Dataset,
    ppi : tf.Tensor,
    strategy : Strategy,
    max_entropy : tf.Tensor,
    n_samples : int,
    num_devices : int,
    batch_size : int
) -> np.ndarray:
    """\
    Run Entropy Compute Strategy
    
    Function to run actual computation of entropy.
    
    Params
    -------
    dataset
        tf.data.Dataset to run entropy computation on
    ppi
        tf.Tensor representing the PPI matrix
    strategy
        tensorflow training strategy
    max_entropy
        tf.Tensor constant of max entropy for PPi network above
    n_samples
        number of samples in dataset
    num_devices
        number of devices computation is being done on
    batch_size
        batch size of computation
    
    Returns
    -------
    entropy_values
        array of entropy values    
    """
    
    # Create iterator
    iterator = iter(dataset)
    
    # Holds temporary results
    results = []
    
    # Iterate samples in our dataset and run entropy scoring per batch
    for i in progressbar(tf.range(math.ceil(n_samples / (num_devices * batch_size)))):
        
        # Runs strategy
        result = strategy.run(batch_compute_entropy, 
            args=(next(iterator), next(cycle([ppi])), next(cycle([max_entropy]))))
        
        # if we have multiple devices returning args, progressively
        # append each other, otherwise just append single device
        if num_devices > 1:
            for r in result.values: results.append(r.numpy())
        else:
            results.append(result.numpy())
    
    # Collapse and return results as array
    return np.array(list(collapse(results)))