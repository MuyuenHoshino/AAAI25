"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.random_graph_regression.SGGN_net import SGGNNet
from nets.random_graph_regression.SAN_NodeLPE import SAN_NodeLPE



def gnn_model(MODEL_NAME, net_params):
    models = {
        'SAN': SAN_NodeLPE,
        'SGGN': SGGNNet
    }
        
    return models[MODEL_NAME](net_params)