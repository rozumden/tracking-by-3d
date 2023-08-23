import torch
import torch.nn as nn
from kornia.geometry.conversions import angle_axis_to_rotation_matrix, quaternion_to_angle_axis, rotation_matrix_to_quaternion
from kornia.geometry.conversions import quaternion_to_rotation_matrix, rotation_matrix_to_angle_axis, angle_axis_to_quaternion, QuaternionCoeffOrder
from main_settings import *

def mesh_normalize(vertices):
    mesh_max = torch.max(vertices, dim=1, keepdim=True)[0]
    mesh_min = torch.min(vertices, dim=1, keepdim=True)[0]
    mesh_middle = (mesh_min + mesh_max) / 2
    vertices = vertices - mesh_middle
    bs = vertices.shape[0]
    mesh_biggest = torch.max(vertices.view(bs, -1), dim=1)[0]
    vertices = vertices / mesh_biggest.view(bs, 1, 1) # * 0.45
    return vertices

def comp_tran_diff(vect):
    vdiff = (vect[1:] - vect[:-1]).abs()
    vdiff[vdiff < 0.2] = 0
    # vdiff[:,-1] = 5*vdiff[:,-1]
    return torch.cat((0*vdiff[:1], vdiff), 0).norm(dim=1)

def comp_diff(vect):
    vdiff = vect[1:] - vect[:-1]
    v2diff = vdiff - torch.cat((vdiff[:1], vdiff[:-1]), 0)
    return torch.cat((0*v2diff[:1], v2diff), 0).norm(dim=1)

def comp_2diff(vdiff):
    v2diff = vdiff - torch.cat((vdiff[:1], vdiff[:-1]), 0)
    return torch.cat((0*v2diff[:1], v2diff), 0).abs()

def qnorm(q1):
    return q1/q1.norm()

def qmult(q1, q0): # q0, then q1, you get q3
    w0, x0, y0, z0 = q0[0]
    w1, x1, y1, z1 = q1[0]
    q3 = torch.cat(((-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0)[None,None], (x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0)[None,None],(-x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0)[None,None],(x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0)[None,None]),1)
    return q3

def qdist(q1, q2):
    return 1 - (q1*q2).sum()**2

def qdifference(q1, q2): # how to get from q1 to q2
    q1conj = -q1
    q1conj[0,0] = q1[0,0]
    q1inv = q1conj / q1.norm()
    diff = qmult(q2, q1inv)
    return diff

class Encoder(nn.Module):
    def __init__(self, config, ivertices, faces, face_features, width, height, n_feat):
        super(Encoder, self).__init__()
        self.config = config
        translation_init = torch.zeros(1,1,config["input_frames"],3)
        translation_init[:,:,0,2] = self.config["tran_init"]
        self.translation = nn.Parameter(translation_init)
        qinit = torch.zeros(1,config["input_frames"],4)#*0.005
        qinit[:,:,0] = 1.0
        init_angle = torch.Tensor(self.config["rot_init"])
        init_quat = angle_axis_to_quaternion(init_angle, order=QuaternionCoeffOrder.WXYZ)
        self.register_buffer('init_quat', init_quat)
        qinit[:,0,:] = init_quat.clone()
        # qinit = angle_axis_to_quaternion(init_angle, order=QuaternionCoeffOrder.WXYZ).repeat(1,config["input_frames"],1)
        self.quaternion = nn.Parameter(qinit)
        offsets = torch.zeros(1,1,config["input_frames"],7)
        offsets[:,:,:,3] = 1.0
        self.register_buffer('offsets', offsets)
        self.register_buffer('used_tran', translation_init.clone())
        self.register_buffer('used_quat', qinit.clone())
        if self.config["predict_vertices"]:
            self.vertices = nn.Parameter(torch.zeros(1,ivertices.shape[0],3))
        self.register_buffer('face_features', torch.from_numpy(face_features).unsqueeze(0).type(self.translation.dtype))
        self.texture_map = nn.Parameter(torch.ones(1,n_feat,self.config["texture_size"],self.config["texture_size"]))
        ivertices = torch.from_numpy(ivertices).unsqueeze(0).type(self.translation.dtype)
        ivertices = mesh_normalize(ivertices)
        self.register_buffer('ivertices', ivertices)
        self.aspect_ratio = height/width
        if self.config["project_coin"]:
            thr = 0.025
            x_coor = ivertices[:,:,0]
            x_coor[x_coor > thr] = thr
            x_coor[x_coor < -thr] = -thr
            self.register_buffer('x_coor', x_coor)

    def set_grad_mesh(self, req_grad):
        self.texture_map.requires_grad = req_grad
        if self.config["predict_vertices"]:
            self.vertices.requires_grad = req_grad

    def forward(self, opt_frames):
        # if self.config["motion_only_last"]:
        #     motion_frames = opt_frames[-self.config["inc_step"]:]
        # else:
        motion_frames = opt_frames
        if self.config["predict_vertices"]:
            vertices = self.ivertices + self.vertices
            if self.config["mesh_normalize"]:
                vertices = mesh_normalize(vertices)
            else:
                vertices = vertices - vertices.mean(1)[:,None,:] ## make center of mass in origin
            if self.config["project_coin"]:
                vertices[:,:,0] = self.x_coor
        else:
            vertices = self.ivertices
        if 0 in opt_frames:
            quaternion_all = [qnorm(self.quaternion[:,0])]
            translation_all = [self.translation[:,:,0]]
        else:
            quaternion_all = [qnorm(self.quaternion[:,0]).detach()]
            translation_all = [self.translation[:,:,0].detach()]
        diffs = []
        dists = [qdist(quaternion_all[-1],quaternion_all[-1]),qdist(quaternion_all[-1],quaternion_all[-1])]
        for frmi in range(1,opt_frames[-1]+1):
            # angle_axis0 = quaternion_to_angle_axis(self.quaternion[:,frmi],order=QuaternionCoeffOrder.WXYZ)#/4#/self.config["rotation_divide"]
            # quaternion0 = qnorm(angle_axis_to_quaternion(angle_axis0, order=QuaternionCoeffOrder.WXYZ))
            quaternion0 = qmult(qnorm(self.quaternion[:,frmi]), qnorm(self.offsets[:,0,frmi,3:]))
            translation0 = self.translation[:,:,frmi] + self.offsets[:,:,frmi,:3]
            if not frmi in opt_frames:
                quaternion0 = quaternion0.detach()
                translation0 = translation0.detach()
            diffs.append(qnorm(qdifference(quaternion_all[-1], quaternion0)))
            if len(diffs) > 1:
                dists.append(qdist(diffs[-2],diffs[-1]))
            quaternion_all.append(quaternion0)
            translation_all.append(translation0)
        quaternion = torch.stack(quaternion_all,1).contiguous()
        translation = torch.stack(translation_all,2).contiguous()
        wghts = (torch.Tensor(opt_frames) - torch.Tensor(opt_frames[:1] + opt_frames[:-1])).to(translation.device)
        # tdiff = wghts*comp_diff(translation[0,0])
        # qdiff = wghts*torch.stack(dists,0).contiguous()
        tdiff = wghts*comp_tran_diff(translation[0,0,opt_frames])
        key_dists = []
        for frmi in opt_frames[1:]:
            key_dists.append(qdist(quaternion[:,frmi-1], quaternion[:,frmi]))
        qdiff = wghts*(torch.stack([qdist(quaternion0, quaternion0)] + key_dists,0).contiguous())
        if self.config["features"] == 'deep':
            texture_map = self.texture_map
        else:
            texture_map = nn.Sigmoid()(self.texture_map)
        return translation[:,:,opt_frames], quaternion[:,opt_frames], vertices, texture_map, tdiff, qdiff

    def forward_normalize(self):
        exp = 0
        if self.config["connect_frames"]:
            exp = nn.Sigmoid()(self.exposure_fraction)
        thr = self.config["camera_distance"]-2
        thrn = thr*4
        translation_all = []
        quaternion_all = []
        for frmi in range(self.translation.shape[1]):
            translation = nn.Tanh()(self.translation[:,frmi,None,:])
            
            translation = translation.view(translation.shape[:2]+torch.Size([1,2,3]))
            translation_new = translation.clone()
            translation_new[:,:,:,:,2][translation[:,:,:,:,2] > 0] = translation[:,:,:,:,2][translation[:,:,:,:,2] > 0]*thr
            translation_new[:,:,:,:,2][translation[:,:,:,:,2] < 0] = translation[:,:,:,:,2][translation[:,:,:,:,2] < 0]*thrn
            translation_new[:,:,:,:,:2] = translation[:,:,:,:,:2]*( (self.config["camera_distance"]-translation_new[:,:,:,:,2:])/2 )
            translation = translation_new
            translation[:,:,:,:,1] = self.aspect_ratio*translation_new[:,:,:,:,1]

            if frmi > 0 and self.config["connect_frames"]:
                translation[:,:,:,1,:] = translation_all[-1][:,:,:,1,:] + (1+exp)* translation_all[-1][:,:,:,0,:]
 
            translation[:,:,:,0,:] = translation[:,:,:,0,:] - translation[:,:,:,1,:]

            quaternion = self.quaternion[:,frmi]
            quaternion = quaternion.view(quaternion.shape[:1]+torch.Size([1,2,4]))
                      
            translation_all.append(translation)
            quaternion_all.append(quaternion)

        translation = torch.stack(translation_all,2).contiguous()[:,:,:,0]
        quaternion = torch.stack(quaternion_all,1).contiguous()[:,:,0]
        if self.config["predict_vertices"]:
            vertices = self.ivertices + self.vertices
            if self.config["mesh_normalize"]:
                vertices = mesh_normalize(vertices)
            else:
                vertices = vertices - vertices.mean(1)[:,None,:] ## make center of mass in origin
        else:
            vertices = self.ivertices

        return translation, quaternion, vertices, self.texture_map, exp

