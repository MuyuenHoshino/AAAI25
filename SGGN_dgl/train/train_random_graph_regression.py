"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

from train.metrics import MAE

from statistics import mean 

from thop import profile


# def train_epoch_flops(model, optimizer, device, data_loader, epoch, MODEL_NAME):
#     model.train()
#     epoch_loss = 0
#     epoch_train_mae = 0
#     nb_data = 0
#     gpu_mem = 0

#     flops_list = []

#     for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
#         batch_graphs = batch_graphs.to(device)
#         batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
#         batch_e = batch_graphs.edata['feat'].flatten().long().to(device)
#         batch_targets = batch_targets.to(device)
#         optimizer.zero_grad()

#         if MODEL_NAME == "SAN":
#             batch_EigVecs = batch_graphs.ndata['EigVecs'].to(device)
#             #random sign flipping
#             sign_flip = torch.rand(batch_EigVecs.size(1)).to(device)
#             sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            
#             batch_EigVals = batch_graphs.ndata['EigVals'].to(device)
#             batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_EigVecs, batch_EigVals)
#             flops, params = profile(model, inputs=(batch_graphs, batch_x, batch_e, batch_EigVecs, batch_EigVals ))
#             # print(flops)
#         else:
#             try:
#                 batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
#                 sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
#                 sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
#                 batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
#             except:
#                 batch_lap_pos_enc = None

#             batch_wl_pos_enc = None
#             batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
#             flops, params = profile(model, inputs=(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc ))
#             # print(flops)
#         flops_list.append(flops)
#         loss = model.loss(batch_scores, batch_targets)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.detach().item()
#         epoch_train_mae += MAE(batch_scores, batch_targets)
#         nb_data += batch_targets.size(0)
#     epoch_loss /= (iter + 1)
#     epoch_train_mae /= (iter + 1)
#     print("##############################")
#     print("mean FLOPs : ", mean(flops_list))
#     return epoch_loss, epoch_train_mae, optimizer

from deepspeed.profiling.flops_profiler import FlopsProfiler

def train_epoch_flops(model, optimizer, device, data_loader, epoch, MODEL_NAME):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0

    prof = FlopsProfiler(model)
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].flatten().long().to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()






        if MODEL_NAME == "SAN":
            batch_EigVecs = batch_graphs.ndata['EigVecs'].to(device)
            #random sign flipping
            sign_flip = torch.rand(batch_EigVecs.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            
            batch_EigVals = batch_graphs.ndata['EigVals'].to(device)



            prof.start_profile()




            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_EigVecs, batch_EigVals)



            prof.stop_profile()



            
        else:
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
                sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
            except:
                batch_lap_pos_enc = None

            batch_wl_pos_enc = None



            prof.start_profile()



            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
            # flops, params = profile(model, inputs=(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc ))



            prof.stop_profile()




        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)



    flops = prof.get_total_flops()

    prof.print_model_profile(profile_step=epoch)
    prof.end_profile()


    return epoch_loss, epoch_train_mae, optimizer



def train_epoch(model, optimizer, device, data_loader, epoch, MODEL_NAME):
    
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].flatten().long().to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()

        if MODEL_NAME == "SAN":
            batch_EigVecs = batch_graphs.ndata['EigVecs'].to(device)
            #random sign flipping
            sign_flip = torch.rand(batch_EigVecs.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            
            batch_EigVals = batch_graphs.ndata['EigVals'].to(device)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_EigVecs, batch_EigVals)

        else:
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
                sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
            except:
                batch_lap_pos_enc = None

            batch_wl_pos_enc = None
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)


        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)

    return epoch_loss, epoch_train_mae, optimizer




def evaluate_network(model, device, data_loader, epoch, MODEL_NAME):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            # batch_e = batch_graphs.edata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].flatten().long().to(device)
            batch_targets = batch_targets.to(device)
            if MODEL_NAME == "SAN":
                batch_EigVecs = batch_graphs.ndata['EigVecs'].to(device)
                #random sign flipping
                sign_flip = torch.rand(batch_EigVecs.size(1)).to(device)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                
                batch_EigVals = batch_graphs.ndata['EigVals'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_EigVecs, batch_EigVals)
            else:
                try:
                    batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
                    sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
                    sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                    batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
                except:
                    batch_lap_pos_enc = None

                batch_wl_pos_enc = None
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae

