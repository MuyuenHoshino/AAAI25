"""
    Utility file to select GraphNN model as
    selected by the user
"""



from nets.superpixels_graph_classification.SGGN_Net import SGGNNet



def gnn_model(MODEL_NAME, net_params):
    models = {
        'SGGN': SGGNNet
    }
        
    return models[MODEL_NAME](net_params)