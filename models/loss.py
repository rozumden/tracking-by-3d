import torch.nn as nn
import torch
from helpers.torch_helpers import *
from kornia import total_variation
import numpy as np
from segmentations import compute_weights

class FMOLoss(nn.Module):
    def __init__(self, config, ivertices, faces):
        super(FMOLoss, self).__init__()
        self.config = config
        if self.config["loss_laplacian_weight"] > 0:    
            self.lapl_loss = LaplacianLoss(ivertices, faces)

    def forward(self, renders, segments, input_batch, vertices, texture_maps, tdiff, qdiff):
        losses = {}
        losses_all = {}
        if self.config["loss_rgb_weight"] > 0:
            modelled_renders = torch.cat( (renders[:,:,:,:-1]*renders[:,:,:,-1:], renders[:,:,:,-1:]), 3).mean(2)
            segments_our = (renders[:,:,0,-1:] > 0).to(renders.dtype)
            track_segm_loss, t_all = fmo_model_loss(input_batch, modelled_renders, segments_our,self.config)
            losses["model"] = self.config["loss_rgb_weight"]*(track_segm_loss)
            losses_all["track_segm_loss"] = t_all
        if self.config["loss_iou_weight"] > 0:
            losses["silh"] = 0
            losses_all["silh"] = []
            denom = renders.shape[1]
            for frmi in range(renders.shape[1]):
                temp_loss = self.config["loss_iou_weight"]*fmo_loss(renders[:,frmi], segments[:,frmi,None])
                losses_all["silh"].append(temp_loss.tolist()[0])
                losses["silh"] = losses["silh"] + temp_loss / denom
        if self.config["predict_vertices"] and self.config["loss_laplacian_weight"] > 0:
            losses["lap"] = self.config["loss_laplacian_weight"]*self.lapl_loss(vertices)

        if self.config["loss_qt_weight"] > 0:
            if qdiff[qdiff > 0].shape[0] > 0:
                losses["qdiff"] = qdiff[qdiff > 0][-1].mean()
            else:
                losses["qdiff"] = qdiff[-1].mean()

            if tdiff[tdiff > 0].shape[0] > 0:
                losses["tdiff"] = tdiff[tdiff > 0][-1].mean()
            else:
                losses["tdiff"] = tdiff[-1].mean()

            losses["tdiff"] = self.config["loss_qt_weight"]*tdiff[-1]
            losses["qdiff"] = self.config["loss_qt_weight"]*qdiff[-1]

        if self.config["loss_dist_weight"] > 0:
            dists = (segments[:,:,0]*renders[:,:,0,-1])
            losses["dist"] = self.config["loss_dist_weight"]*(dists.sum((0,2,3))/renders[:,:,0,-1].sum((0,2,3))).mean()

        if self.config["loss_tv_weight"] > 0:
            texture_maps_rep = torch.cat((texture_maps[:,:,-1:],texture_maps,texture_maps[:,:,:1]),2)
            texture_maps_rep = torch.cat((texture_maps_rep[:,:,:,-1:],texture_maps_rep,texture_maps_rep[:,:,:,:1]),3)
            texture_maps_rep = torch.cat((texture_maps_rep[:,:,-1:],texture_maps_rep,texture_maps_rep[:,:,:1]),2)
            losses["tv"] = self.config["loss_tv_weight"]*total_variation(texture_maps_rep)/(3*self.config["texture_size"]**2) 

        loss = 0
        for ls in losses:
            loss += losses[ls]
        return losses_all, losses, loss

def fmo_loss(Yp, Y):
    YM = Y[:,:,-1:,:,:]
    YpM = Yp[:,:,-1:,:,:]
    YMb = ((YM+YpM) > 0).type(YpM.dtype)
    loss = iou_loss(YM, YpM)
    return loss

def iou_loss(YM, YpM):
    A_inter_B = YM * YpM
    A_union_B = (YM + YpM - A_inter_B)
    iou = 1 - (torch.sum(A_inter_B, [2,3,4]) / torch.sum(A_union_B, [2,3,4])).mean(1)
    return iou

def cauchy_loss(YpM, YM, YMb, scale=0.25):
    losses = nn.L1Loss(reduction='none')(YpM*YMb, YM*YMb)#**2
    cauchy_losses = (scale**2) * torch.log(1 + losses/(scale**2))
    # cauchy_losses = (scale**2) * torch.log(1 + losses**2)
    bloss = cauchy_losses.sum([1,2,3]) / (YMb.sum([1,2,3])*YpM.shape[1] + 0.01)
    return bloss


def batch_loss(YpM, YM, YMb, do_mult=True, weights = None):
    if do_mult:
        losses = nn.L1Loss(reduction='none')(YpM*YMb, YM*YMb)
    else:
        losses = nn.L1Loss(reduction='none')(YpM, YM)
    if weights is not None:
        losses = weights*losses
        YMb = weights
    if len(losses.shape) > 4:
        bloss = losses.sum([1,2,3,4]) / YMb.sum([1,2,3,4])
    else:
        bloss = losses.sum([1,2,3]) / (YMb.sum([1,2,3]) + 0.01)
    return bloss

    
def fmo_model_loss(input_batch, renders, segments, config):  
    Mask = segments[:,:,-1:]  
    if Mask is None:
        Mask = renders[:,:,-1:] > 0.05
    Mask = Mask.type(renders.dtype)
    model_loss = 0
    model_loss_all = []
    for frmi in range(input_batch.shape[1]):
        if config["features"] == 'deep':
            temp_loss = cauchy_loss(renders[:,frmi,:-1], input_batch[:,frmi], Mask[:,frmi])
        else:
            temp_loss = batch_loss(renders[:,frmi,:-1], input_batch[:,frmi], Mask[:,frmi])
        model_loss_all.append(temp_loss.tolist()[0])
        model_loss = model_loss + temp_loss / input_batch.shape[1]
    # losses = nn.L1Loss(reduction='none')(renders, input_batch[:,:3])
    # model_loss = losses.mean([1,2,3])
    return model_loss, model_loss_all


# Taken from
# https://github.com/ShichenLiu/SoftRas/blob/master/soft_renderer/losses.py

class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.shape[0]
        self.nf = faces.shape[0]
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.matmul(self.laplacian, x)
        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).mean(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x
        
