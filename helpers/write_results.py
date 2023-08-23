import os
import torch
import numpy as np
import cv2
from helpers.torch_helpers import write_renders
from models.kaolin_wrapper import write_obj_mesh
from torchvision.utils import save_image
from utils import write_video, segment2bbox
import torchvision.ops.boxes as bops

class WriteResults():
	def __init__(self, write_folder, imgshape, n_files):
		self.write_folder = write_folder
		self.all_input = cv2.VideoWriter(os.path.join(self.write_folder,'all_input.avi'),cv2.VideoWriter_fourcc(*"MJPG"), 10, (imgshape[4], imgshape[3]),True)
		self.all_segm = cv2.VideoWriter(os.path.join(self.write_folder,'all_segm.avi'),cv2.VideoWriter_fourcc(*"MJPG"), 10, (imgshape[4], imgshape[3]),True)
		self.all_proj = cv2.VideoWriter(os.path.join(self.write_folder,'all_proj.avi'),cv2.VideoWriter_fourcc(*"MJPG"), 10, (imgshape[4], imgshape[3]),True)
		self.all_proj_filtered = cv2.VideoWriter(os.path.join(self.write_folder,'all_proj_filtered.avi'),cv2.VideoWriter_fourcc(*"MJPG"), 10, (imgshape[4], imgshape[3]),True)
		self.baseline_iou = -np.ones((n_files-1,1))
		self.our_iou = -np.ones((n_files-1,1))
		self.our_losses = -np.ones((n_files-1,1))
		self.vsd_errors = -np.ones((n_files-1,10))

	def write_results(self, stepi, track6d, silh_losses, b0, bboxes, segment, depths_dict):
		self.our_losses[stepi-1] = silh_losses[-1]
		if track6d.config["features"] == 'deep':
			track6d.rgb_apply(track6d.images[:,:,:,b0[0]:b0[1],b0[2]:b0[3]], track6d.segments[:,:,:,b0[0]:b0[1],b0[2]:b0[3]], track6d.keyframes, b0)
			tex = torch.nn.Sigmoid()(track6d.rgb_encoder.texture_map)
		with torch.no_grad():
			translation, quaternion, vertices, texture_maps, tdiff, qdiff = track6d.encoder(track6d.keyframes)
			if track6d.config["features"] == 'rgb':
				tex = texture_maps
			feat_renders_crop = track6d.rendering(translation, quaternion, vertices, track6d.encoder.face_features, texture_maps, True)
			depth_map_crop = feat_renders_crop[:,:,:,-1]
			feat_renders_crop = feat_renders_crop[:,:,:,:-1]
			feat_renders_crop = torch.cat((feat_renders_crop[:,:,:,:3],feat_renders_crop[:,:,:,-1:]),3)
			renders_crop = track6d.rendering(translation, quaternion, vertices, track6d.encoder.face_features, tex)
			renders_crop = torch.cat((renders_crop[:,:,:,:3],renders_crop[:,:,:,-1:]),3)
			renders = torch.zeros(renders_crop.shape[:4]+track6d.images_feat.shape[-2:]).to(track6d.device)
			renders[:,:,:,:,b0[0]:b0[1],b0[2]:b0[3]] = renders_crop

			depth_map = torch.zeros(depth_map_crop.shape[:3]+track6d.images_feat.shape[-2:]).to(track6d.device)
			depth_map[:,:,:,b0[0]:b0[1],b0[2]:b0[3]] = depth_map_crop-depth_map_crop.min()

			write_renders(feat_renders_crop, self.write_folder, track6d.config["max_keyframes"]+1, ids=0)
			write_renders(renders_crop, self.write_folder, track6d.config["max_keyframes"]+1, ids=1)
			write_renders(torch.cat((track6d.images[:,:,None,:,b0[0]:b0[1],b0[2]:b0[3]],feat_renders_crop[:,:,:,-1:]),3), self.write_folder, track6d.config["max_keyframes"]+1, ids=2)
			write_obj_mesh(vertices[0].cpu().numpy(), track6d.best_model["faces"], track6d.encoder.face_features[0].cpu().numpy(), os.path.join(self.write_folder,'mesh.obj'))
			save_image(texture_maps[:,:3], os.path.join(self.write_folder,'tex_deep.png'))
			save_image(tex, os.path.join(self.write_folder,'tex.png'))
			write_video(renders[0,:,0,:3].detach().cpu().numpy().transpose(2,3,1,0), os.path.join(self.write_folder,'im_recon.avi'), fps=6)
			write_video(track6d.images[0,:,:3].cpu().numpy().transpose(2,3,1,0), os.path.join(self.write_folder,'input.avi'), fps=6)
			write_video((track6d.images[0,:,:3]*track6d.segments[0,:,1:2]).cpu().numpy().transpose(2,3,1,0), os.path.join(self.write_folder,'segments.avi'), fps=6)
			for tmpi in range(renders.shape[1]):
				img = track6d.images[0,tmpi,:3,b0[0]:b0[1],b0[2]:b0[3]]
				seg = track6d.segments[0,:,1:2][tmpi,:,b0[0]:b0[1],b0[2]:b0[3]].clone()
				save_image(seg, os.path.join(self.write_folder, 'imgs', 's{}.png'.format(tmpi)))
				seg[seg == 0] = 0.35
				save_image(img, os.path.join(self.write_folder, 'imgs', 'i{}.png'.format(tmpi)))
				save_image(track6d.images_feat[0,tmpi,:3,b0[0]:b0[1],b0[2]:b0[3]], os.path.join(self.write_folder, 'imgs', 'if{}.png'.format(tmpi)))
				save_image(torch.cat((img,seg),0), os.path.join(self.write_folder, 'imgs', 'is{}.png'.format(tmpi)))
				save_image(renders_crop[0,tmpi,0,[3,3,3]], os.path.join(self.write_folder, 'imgs', 'm{}.png'.format(tmpi)))
				save_image(renders_crop[0,tmpi,0,:], os.path.join(self.write_folder, 'imgs', 'r{}.png'.format(tmpi)))
				save_image(feat_renders_crop[0,tmpi,0,:], os.path.join(self.write_folder, 'imgs', 'f{}.png'.format(tmpi)))
			AR = 0
			if type(bboxes) is dict or (bboxes[stepi][0] is 'm'):
				gt_segm = None
				if (not type(bboxes) is dict) and bboxes[stepi][0] is 'm':
					m_, offset_ = create_mask_from_string(bboxes[stepi][1:].split(','))
					gt_segm = segment[0,0,-1]*0
					gt_segm[offset_[1]:offset_[1]+m_.shape[0], offset_[0]:offset_[0]+m_.shape[1]] = torch.from_numpy(m_)
				elif stepi in bboxes:
					gt_segm = track6d.tracker.process_segm(bboxes[stepi])[0].to(track6d.device)
				if not gt_segm is None:
					self.baseline_iou[stepi-1] = float((segment[0,0,-1]*gt_segm > 0).sum())/float(((segment[0,0,-1]+gt_segm) > 0).sum()+0.00001)
					self.our_iou[stepi-1] = float((renders[0,-1,0,3]*gt_segm > 0).sum())/float(((renders[0,-1,0,3]+gt_segm) > 0).sum()+0.00001)
					if depths_dict is not None:
						gt_depth = track6d.tracker.process_segm(depths_dict[stepi])[0].to(track6d.device)
						errors = compute_vsd(depth=depth_map[0,-1,0], vis=renders[0,-1,0,3]>0, gt_depth=gt_depth, gt_vis=gt_segm>0, diameter=2.0)
						self.vsd_errors[stepi-1] = errors
						err = self.vsd_errors[:stepi]
						AR = []
						for tau in np.arange(0.05, 0.55, 0.05):
							AR.append((err < tau).sum()/(err.shape[0]*err.shape[1]))
			elif bboxes is not None:   
				bbox = track6d.config["image_downsample"]*torch.tensor([bboxes[stepi]+[0,0,bboxes[stepi][0],bboxes[stepi][1]]])
				self.baseline_iou[stepi-1] = bops.box_iou(bbox, torch.tensor([segment2bbox(segment[0,0,-1])], dtype=torch.float64))
				self.our_iou[stepi-1] = bops.box_iou(bbox, torch.tensor([segment2bbox(renders[0,-1,0,3])], dtype=torch.float64))
			print('Baseline IoU {}, our IoU {}, current AR {}'.format(self.baseline_iou[stepi-1], self.our_iou[stepi-1], np.mean(AR)))
			np.savetxt(os.path.join(self.write_folder,'baseline_iou.txt'), self.baseline_iou, fmt='%.10f', delimiter='\n')
			np.savetxt(os.path.join(self.write_folder,'iou.txt'), self.our_iou, fmt='%.10f', delimiter='\n')
			np.savetxt(os.path.join(self.write_folder,'losses.txt'), self.our_losses, fmt='%.10f', delimiter='\n')
			np.savetxt(os.path.join(self.write_folder,'vsd_errors.txt'), self.vsd_errors, fmt='%.5f', newline='\n')
			self.all_input.write((track6d.images[0,:,:3].clamp(min=0,max=1).cpu().numpy().transpose(2,3,1,0)[:,:,[2,1,0],-1] * 255).astype(np.uint8))
			self.all_segm.write(((track6d.images[0,:,:3]*track6d.segments[0,:,1:2]).clamp(min=0,max=1).cpu().numpy().transpose(2,3,1,0)[:,:,[2,1,0],-1] * 255).astype(np.uint8))
			self.all_proj.write((renders[0,:,0,:3].detach().clamp(min=0,max=1).cpu().numpy().transpose(2,3,1,0)[:,:,[2,1,0],-1] * 255).astype(np.uint8))
			if silh_losses[-1] > 0.3:
				renders[0,-1,0,3] = segment[0,0,-1]
				renders[0,-1,0,:3] = track6d.images[0,-1,:3]*segment[0,0,-1]
			self.all_proj_filtered.write((renders[0,:,0,:3].detach().clamp(min=0,max=1).cpu().numpy().transpose(2,3,1,0)[:,:,[2,1,0],-1] * 255).astype(np.uint8))
		return renders

	def close(self):
		self.all_input.release()
		self.all_segm.release()
		self.all_proj.release()
		self.all_proj_filtered.release()