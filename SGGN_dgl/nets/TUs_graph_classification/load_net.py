"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.TUs_graph_classification.SGGN_Net import SGGNNet, SGGNNet_withoutC


def gnn_model(MODEL_NAME, net_params):
    models = {
        'SGGN': SGGNNet,
        'SGGN-C': SGGNNet_withoutC
    }
        
    return models[MODEL_NAME](net_params)