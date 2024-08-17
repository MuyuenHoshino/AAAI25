"""
    File to load dataset based on user control from main file
"""

from data.molecules import MoleculeDataset
from data.SBMs import SBMsDataset
from data.random import randomDataset
from data.superpixels import SuperPixDataset
from data.TUs import TUsDataset


def LoadData(DATASET_NAME, num_nodes=0):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """    
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC':
        return MoleculeDataset(DATASET_NAME)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS: 
        return SBMsDataset(DATASET_NAME)

    
    if DATASET_NAME == 'MNIST':
        return SuperPixDataset(DATASET_NAME)


    TU_DATASETS = ['DD']
    if DATASET_NAME in TU_DATASETS: 
        return TUsDataset(DATASET_NAME)
    

    Random_DATASETS = ['random']
    if DATASET_NAME in Random_DATASETS: 
        return randomDataset(num_nodes)



