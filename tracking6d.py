import os
import torch
import numpy as np

from my_utils import *
from models.initial_mesh import generate_initial_mesh, generate_face_features
from models.kaolin_wrapper import load_obj, write_obj_mesh

from models.encoder import *
from models.rendering import *
from models.loss import *

from helpers.torch_helpers import write_renders
from helpers.write_results import WriteResults
from scipy.ndimage.filters import gaussian_filter
import copy
from segmentations import *
import time

sys.path.append('S2DNet')
from s2dnet import S2DNet

sys.path.append('RAFT/core')
from RAFT.core.raft import RAFT
from utils.utils import InputPadder

class Tracking6D():
    def __init__(self, config, device, write_folder, file0, bbox0, init_mask=None):
        self.write_folder = write_folder
        self.config = config.copy()
        self.config["fmo_steps"] = 1
        self.device = device
        torch.backends.cudnn.benchmark = True
        if type(bbox0) is dict:
            self.tracker = PrecomputedTracker(self.config["image_downsample"],self.config["max_width"],bbox0,self.config["grabcut"])
        else:
            if self.config["tracker_type"] is 'csrt':
                self.tracker = CSRTrack(self.config["image_downsample"],self.config["max_width"],self.config["grabcut"])
            elif self.config["tracker_type"] is 'ostrack':
                self.tracker = OSTracker(self.config["image_downsample"],self.config["max_width"],self.config["grabcut"])
            elif self.config["tracker_type"] is 'd3s':
                self.tracker = MyTracker(self.config["image_downsample"],self.config["max_width"],self.config["grabcut"])
        if self.config["features"] == 'deep':
            self.net = S2DNet(device=device,checkpoint_path=g_ext_folder).to(device)
            self.feat = lambda x: self.net(x[0])[0][None][:,:,:64]
        else:
            self.feat = lambda x: x
        if self.config["optical_flow"]:
            raft_config = RAFTConfig(g_raft_model)
            self.of = torch.nn.DataParallel(RAFT(raft_config))
            self.of.load_state_dict(torch.load(raft_config.model))
            self.of = self.of.module
            self.of.to(device)
            self.of.eval()
        self.images, self.segments, self.config["image_downsample"] = self.tracker.init_bbox(file0, bbox0, init_mask)
        self.images, self.segments = self.images[None].to(self.device), self.segments[None].to(self.device)
        self.images_feat = self.feat(self.images).detach()
        self.flow = np.zeros(self.images.shape[-2:]+(2,1), np.float32)
        shape = self.segments.shape
        prot = self.config["shapes"][0]
        if not config["init_shape"] is False:
            mesh = load_obj(config["init_shape"])
            ivertices = mesh.vertices.numpy()
            ivertices = ivertices - ivertices.mean(0)
            ivertices = ivertices / ivertices.max()
            faces = mesh.faces.numpy().copy()
            iface_features = generate_face_features(ivertices, faces)
        elif prot == 'sphere':
            ivertices, faces, iface_features = generate_initial_mesh(self.config["mesh_size"])
        else:
            mesh = load_obj(os.path.join('/cluster/home/denysr/src/ShapeFromBlur/prototypes',prot+'.obj'))
            ivertices = mesh.vertices.numpy()
            faces = mesh.faces.numpy().copy()
            iface_features = mesh.uvs[mesh.face_uvs_idx].numpy()
        self.faces = faces
        self.rendering = RenderingKaolin(self.config, self.faces, shape[-1], shape[-2]).to(self.device)
        self.encoder = Encoder(self.config, ivertices, faces, iface_features, shape[-1], shape[-2], self.images_feat.shape[2]).to(self.device)
        all_parameters = list(self.encoder.parameters())
        self.optimizer = torch.optim.Adam(all_parameters, lr = self.config["learning_rate"])
        self.encoder.train()
        self.loss_function = FMOLoss(self.config, ivertices, faces).to(self.device)
        if self.config["features"] == 'deep':
            config = self.config.copy()
            config["features"] = 'rgb'
            self.rgb_encoder = Encoder(config, ivertices, faces, iface_features, shape[-1], shape[-2], 3).to(self.device)
            rgb_parameters = list(self.rgb_encoder.parameters())[-1:]
            self.rgb_optimizer = torch.optim.Adam(rgb_parameters, lr = self.config["learning_rate"])
            self.rgb_encoder.train()
            config["loss_laplacian_weight"] = 0
            config["loss_tv_weight"] = 1.0
            config["loss_iou_weight"] = 0
            config["loss_dist_weight"] = 0
            config["loss_qt_weight"] = 0
            self.rgb_loss_function = FMOLoss(config, ivertices, faces).to(self.device)
        if self.config["verbose"]:
            print('Total params {}'.format(sum(p.numel() for p in self.encoder.parameters())))
        self.best_model = {}
        self.best_model["value"] = 100
        self.best_model["face_features"] = self.encoder.face_features.detach().clone()
        self.best_model["faces"] = faces
        self.best_model["encoder"] = None
        self.keyframes = [0]


    def run_tracking(self, files, bboxes, depths_dict=None):
        reswriter = WriteResults(self.write_folder, self.images.shape, files.shape[0])
        self.config["loss_rgb_weight"] = 0 # for initialization
        b0 = None
        image_prev = self.images
        for stepi in range(1,self.config["input_frames"]):
            image_raw, segment = self.tracker.next(files[stepi])
            image, segment = image_raw[None].to(self.device), segment[None].to(self.device)
            if b0 is not None:
                segment_clean = segment*0
                segment_clean[:,:,:,b0[0]:b0[1],b0[2]:b0[3]] = segment[:,:,:,b0[0]:b0[1],b0[2]:b0[3]]
                segment_clean[:,:,0] = segment[:,:,0]
                segment = segment_clean
            with torch.no_grad():
                image_feat = self.feat(image).detach()

                padder = InputPadder(image.shape)
                image_prev_pad, image_pad = padder.pad(image_prev[0], image[0])
                _, flow_up_pad = self.of(image_prev_pad, image_pad, iters=20, test_mode=True)
                flow_up = padder.unpad(flow_up_pad)
                flow_cur = flow_up[0].permute(1,2,0).cpu().numpy()
                self.flow[...,-1] = combine_flows(self.flow[...,-1], flow_cur)
                nmofs, rads = compute_norm_mean_of_array(self.flow, self.segments)
                self.flow = np.concatenate( (self.flow, np.zeros(flow_cur.shape, np.float32)[...,None]), 3)   

            self.images = torch.cat( (self.images, image), 1)     
            self.segments = torch.cat( (self.segments, segment), 1)        
            self.images_feat = torch.cat( (self.images_feat, image_feat), 1)        
            self.keyframes.append(stepi)
            start = time.time()
            b0 = get_bbox(self.segments)
            self.rendering = RenderingKaolin(self.config, self.faces, b0[3]-b0[2], b0[1]-b0[0]).to(self.device)
            self.encoder.offsets[:,:,stepi,:3] = (self.encoder.used_tran[:,:,stepi-1]+self.encoder.offsets[:,:,stepi-1,:3])
            self.encoder.offsets[:,0,stepi,3:] = qmult(qnorm(self.encoder.used_quat[:,stepi-1]),qnorm(self.encoder.offsets[:,0,stepi-1,3:]))

            self.apply(self.images_feat[:,:,:,b0[0]:b0[1],b0[2]:b0[3]], self.segments[:,:,:,b0[0]:b0[1],b0[2]:b0[3]], self.keyframes, b0)
            silh_losses = np.array(self.best_model["losses"]["silh"])

            print('Elapsed time in seconds: ', time.time() - start)
            if silh_losses[-1] < 0.8:
                self.encoder.used_tran[:,:,stepi] = self.encoder.translation[:,:,stepi].detach()
                self.encoder.used_quat[:,stepi] = self.encoder.quaternion[:,stepi].detach()
            
            # if self.config["write_results"]:
            renders = reswriter.write_results(stepi, self, silh_losses, b0, bboxes, segment, depths_dict)
            normTdist = compute_trandist(renders) / rads
            translation, quaternion, vertices, texture_maps, tdiff, qdiff = self.encoder(self.keyframes)
            angs = compute_angs(quaternion)

            image_prev = image
            keep_keyframes = (silh_losses <= 1e10) 
            keep_keyframes = (silh_losses < 0.8) # remove really bad ones (IoU < 0.2)
            keep_keyframes[np.argmin(silh_losses)] = True # keep the best (in case all are bad)
            of_th = 0.3
            rot_degree_th = 40
            norm_tran_th = 3
            small_rotation = angs.shape[0] > 1 and angs[-1] < rot_degree_th and angs[-2] < rot_degree_th
            small_translation = normTdist.shape[0] > 1 and  normTdist[-1] < norm_tran_th and normTdist[-2] < norm_tran_th
            # if nmofs.shape[0] > 1 and nmofs[-1] < of_th and nmofs[-2] < of_th:
            if nmofs.shape[0] > 1 and (small_rotation and small_translation):
                keep_keyframes[-1] = True
                keep_keyframes[-2] = False
                keep_keyframes[-3] = True
            self.flow = filter_flows(self.flow, keep_keyframes)

            self.keyframes = (np.array(self.keyframes)[keep_keyframes]).tolist()
            self.images = self.images[:,keep_keyframes]
            self.images_feat = self.images_feat[:,keep_keyframes]
            self.segments = self.segments[:,keep_keyframes]
            if len(self.keyframes) > self.config["max_keyframes"]:
                self.keyframes = self.keyframes[-self.config["max_keyframes"]:]
                self.images = self.images[:,-self.config["max_keyframes"]:]
                self.images_feat = self.images_feat[:,-self.config["max_keyframes"]:]
                self.segments = self.segments[:,-self.config["max_keyframes"]:]
        reswriter.close()
        return self.best_model

    def apply(self, input_batch, segments, opt_frames = None, bounds = None):
        if self.config["write_results"]:
            save_image(input_batch[0,:,:3],os.path.join(self.write_folder,'im.png'), nrow=self.config["max_keyframes"]+1)
            save_image(torch.cat((input_batch[0,:,:3],segments[0,:,[1]]),1),os.path.join(self.write_folder,'segments.png'), nrow=self.config["max_keyframes"]+1)
        
        self.best_model["value"] = 100
        self.best_model["losses"] = None
        iters_without_change = 0
        for epoch in range(self.config["iterations"]):
            translation, quaternion, vertices, texture_maps, tdiff, qdiff = self.encoder(opt_frames)
            renders = self.rendering(translation, quaternion, vertices, self.encoder.face_features, texture_maps)
            losses_all, losses, jloss = self.loss_function(renders, segments, input_batch, vertices, texture_maps, tdiff, qdiff)

            if "model" in losses:
                model_loss = losses["model"].mean().item()
            else:
                model_loss = losses["silh"].mean().item()
            if self.config["verbose"] and epoch % 20 == 0:
                print("Epoch {:4d}".format(epoch+1), end =" ")
                for ls in losses:
                    print(", {} {:.3f}".format(ls,losses[ls].mean().item()), end =" ")
                print("; joint {:.3f}".format(jloss.item()))

            if model_loss < self.best_model["value"]:
                iters_without_change = 0
                self.best_model["value"] = model_loss
                self.best_model["losses"] = losses_all
                self.best_model["encoder"] = copy.deepcopy(self.encoder.state_dict())
                if self.config["write_intermediate"]:
                    write_renders(torch.cat((renders[:,:,:,:3],renders[:,:,:,-1:]),3), self.write_folder, self.config["max_keyframes"]+1)
            else:
                iters_without_change += 1

            if self.config["loss_rgb_weight"] == 0:
                if epoch > 100 or model_loss < 0.1:
                    self.config["loss_rgb_weight"] = 1.0
                    self.best_model["value"] = 100
            else:
                if epoch > 50 and self.best_model["value"] < self.config["stop_value"] and iters_without_change > 10:
                    break
                if iters_without_change > 100:
                    break
            if epoch < self.config["iterations"] - 1:
                jloss = jloss.mean()
                self.optimizer.zero_grad()
                jloss.backward()
                self.optimizer.step()
        self.encoder.load_state_dict(self.best_model["encoder"])

    
    def rgb_apply(self, input_batch, segments, opt_frames = None, bounds = None):
        self.best_model["value"] = 100
        model_state = self.rgb_encoder.state_dict()
        pretrained_dict = self.best_model["encoder"] 
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k != "texture_map"}
        model_state.update(pretrained_dict)
        self.rgb_encoder.load_state_dict(model_state)
        for epoch in range(self.config["rgb_iters"]):
            translation, quaternion, vertices, texture_maps, tdiff, qdiff = self.rgb_encoder(opt_frames)
            renders = self.rendering(translation, quaternion, vertices, self.encoder.face_features, texture_maps)
            losses_all, losses, jloss = self.rgb_loss_function(renders, segments, input_batch, vertices, texture_maps, tdiff, qdiff)
            if self.best_model["value"] < 0.1 and iters_without_change > 10:
                break
            if epoch < self.config["iterations"] - 1:
                jloss = jloss.mean()
                self.rgb_optimizer.zero_grad()
                jloss.backward()
                self.rgb_optimizer.step()

