import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from skimage.measure import label, regionprops
import os
import cv2
import numpy as np
from main_settings import *
import matplotlib.pyplot as plt
from PIL import Image

def	sync_directions_rgba(est_hs):
	for frmi in range(1,est_hs.shape[0]):
		tsr0 = est_hs[frmi-1]
		tsr = est_hs[frmi]
		if frmi == 1:
			forward = np.min([torch.mean((tsr0[-1] - tsr[-1])**2), torch.mean((tsr0[-1] - tsr[0])**2)])
			backward = np.min([torch.mean((tsr0[0] - tsr[-1])**2), torch.mean((tsr0[0] - tsr[0])**2)])
			if backward < forward:
				est_hs[frmi-1] = torch.flip(est_hs[frmi-1],[0])
				tsr0 = est_hs[frmi-1]

		if torch.mean((tsr0[-1] - tsr[-1])**2) < torch.mean((tsr0[-1] - tsr[0])**2):
			## reverse time direction for better alignment
			est_hs[frmi] = torch.flip(est_hs[frmi],[0])
	return est_hs

def write_renders(renders, tmp_folder, nrow=8, ids=None, im_name_base='im_recon'):
	name = im_name_base+'.png'
	if ids is not None:
		name = im_name_base+'{}.png'.format(ids)
	# modelled_renders = (renders[:,:,:,:3]*renders[:,:,:,3:4]).mean(2)
	modelled_renders = renders.mean(2)
	save_image(modelled_renders[0],os.path.join(tmp_folder,name),nrow=nrow)

class SRWriter:
	def __init__(self, imtemp, path, available_gt=True):
		self.available_gt = available_gt
		if self.available_gt:
			fctr = 3
		else:
			fctr = 2
		if imtemp.shape[0] > imtemp.shape[1]:
			self.width = True
			shp = (imtemp.shape[0], imtemp.shape[1]*fctr, 3)
			self.value = imtemp.shape[1]
		else:
			self.width = False
			shp = (imtemp.shape[0]*fctr, imtemp.shape[1], 3)
			self.value = imtemp.shape[0]
		self.video = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*"MJPG"), 12, (shp[1], shp[0]), True)
		self.img = np.zeros(shp)

	def update_ls(self,lsf):
		if self.width:
			self.img[:,:self.value] = lsf
		else:
			self.img[:self.value,:] = lsf

	def write_next(self,hs,est):
		if hs is not None:
			if self.width:
				self.img[:,2*self.value:] = hs
			else:
				self.img[2*self.value:,:] = hs
		if est is not None:
			if self.width:
				self.img[:,self.value:2*self.value] = est
			else:
				self.img[self.value:2*self.value,:] = est
		self.img[self.img>1]=1
		self.img[self.img<0]=0
		self.video.write( (self.img.copy() * 255)[:,:,[2,1,0]].astype(np.uint8) )

	def close(self):
		self.video.release()

def renders2traj(renders,device):
	masks = renders[:,:,-1]
	sumx = torch.sum(masks,-2)
	sumy = torch.sum(masks,-1)
	cenx = torch.sum(sumy*torch.arange(1,sumy.shape[-1]+1)[None,None].float().to(device),-1) / torch.sum(sumy,-1)
	ceny = torch.sum(sumx*torch.arange(1,sumx.shape[-1]+1)[None,None].float().to(device),-1) / torch.sum(sumx,-1)
	est_traj = torch.cat((cenx.unsqueeze(-1),ceny.unsqueeze(-1)),-1)
	return est_traj

def renders2traj_bbox(renders_rgba):
	masks = renders_rgba[:,:,-1]
	est_traj = []
	for ti in range(masks.shape[2]):
		th = np.min([0.1, 0.5*np.max(masks[:,:,ti])])
		dI = (masks[:,:,ti] >= th).astype(float)
		labeled = label(dI)
		regions = regionprops(labeled)
		areas = [reg.area for reg in regions]
		region = regions[np.argmax(areas)]
		bbox = np.array(region.bbox)
		est_traj = np.r_[est_traj, bbox[:2] + (bbox[2:]-bbox[:2])/2]
	est_traj = np.reshape(est_traj, (-1,2)).T
	return est_traj
	
def write_latent(rendering, latent, device, folder, steps,frmi=0,videoname=None):
	if videoname is None:
		videoname = 'output.avi'
	write_video = True
	write_images = False
	eps = 0
	out = None
	translation, quaternion, vertices, texture_maps = latent
	with torch.no_grad():
		times = torch.linspace(0+eps,1-eps,steps).to(device)
		renders = rendering(translation, quaternion, vertices, texture_maps)
		for ki in range(renders.shape[2]):
			ti = times[ki]
			if write_images:
				save_image(renders[0,frmi,ki].clone(), os.path.join(folder, 'latent{:04d}.png'.format(int(ti*100))))
			if write_video:
				if out is None:
					out = cv2.VideoWriter(os.path.join(folder, videoname),cv2.VideoWriter_fourcc(*"MJPG"), 6, (renders.shape[5], renders.shape[4]),True)
				im4 = renders[0,frmi,ki].data.cpu().detach().numpy().transpose(1,2,0)
				im = im4[:,:,[2,1,0]] * im4[:,:,3:] + 1* (1 - im4[:,:,3:])
				out.write( (im * 255).astype(np.uint8) )
	if write_video:
		out.release()

	return renders

def write_gt(gt_paths, folder, bgr_clr = 1, videoname = 'output_gt.avi'):
	write_video = True
	out = None
	renders = []
	n_frms = len(gt_paths)
	if n_frms > 100:
		n_frms = gt_paths.shape[-1]
	for ti in range(n_frms):
		if isinstance(gt_paths, list):
			im4 = np.array(Image.open(gt_paths[ti]))/255
		elif gt_paths.shape[0] < 100:
			im4 = gt_paths[ti].numpy().transpose(1,2,0)
		else:
			im4 = gt_paths[...,ti]

		renders.append(im4[np.newaxis].copy())
		if out is None:
			out = cv2.VideoWriter(os.path.join(folder, videoname),cv2.VideoWriter_fourcc(*"MJPG"), 6, (im4.shape[1], im4.shape[0]),True)
		im = im4[:,:,[2,1,0]] * im4[:,:,3:] + bgr_clr* (1 - im4[:,:,3:])
		out.write( (im.copy() * 255).astype(np.uint8) )
	out.release()
	renders = np.stack(renders,1)
	renders = torch.from_numpy(renders).float().permute(0,1,4,2,3)
	return renders	 

def write_gt_masks(gt_paths, folder, bgr_clr = 1, videoname = 'output_masks_gt.avi'):
	write_video = True
	out = None
	renders = []
	n_frms = len(gt_paths)
	if n_frms > 100:
		n_frms = gt_paths.shape[-1]
	for ti in range(n_frms):
		if isinstance(gt_paths, list):
			im4 = np.array(Image.open(gt_paths[ti]))/255
		elif gt_paths.shape[0] < 100:
			im4 = gt_paths[ti].numpy().transpose(1,2,0)
		else:
			im4 = gt_paths[...,ti]
		renders.append(im4[np.newaxis].copy())
		if out is None:
			out = cv2.VideoWriter(os.path.join(folder, videoname),cv2.VideoWriter_fourcc(*"MJPG"), 6, (im4.shape[1], im4.shape[0]),True)
		im = (im4[:,:,[3,3,3]])
		if bgr_clr == 1:
			im = 1 - im
		out.write( (im.copy() * 255).astype(np.uint8) )
	out.release()
	renders = np.stack(renders,1)
	renders = torch.from_numpy(renders).float().permute(0,1,4,2,3)
	return renders	

def get_images(encoder, rendering, device, val_batch):
	with torch.no_grad():
		translation, quaternion, vertices, texture_maps = encoder(val_batch)
		times = torch.linspace(0,1,2).to(device)
		renders = rendering(translation, quaternion, vertices, texture_maps)

	renders = renders.cpu().numpy()
	renders = renders[:,:,:,3:4]*(renders[:,:,:,:3]-1)+1
	return renders

def normalized_cross_correlation_channels(image1, image2):
	mean1 = image1.mean([2,3,4],keepdims=True)
	mean2 = image2.mean([2,3,4],keepdims=True) 
	std1 = image1.std([2,3,4],unbiased=False,keepdims=True)
	std2 = image2.std([2,3,4],unbiased=False,keepdims=True)
	eps=1e-8
	bs, ts, *sh = image1.shape
	N = sh[0]*sh[1]*sh[2]
	im1b = ((image1-mean1)/(std1*N+eps)).view(bs*ts, sh[0], sh[1], sh[2])
	im2b = ((image2-mean2)/(std2+eps)).reshape(bs*ts, sh[0], sh[1], sh[2])
	padding = tuple(side // 10 for side in sh[:2]) + (0,)
	result = F.conv3d(im1b[None], im2b[:,None], padding=padding, bias=None, groups=bs*ts)
	ncc = result.view(bs*ts, -1).max(1)[0].view(bs, ts)
	return ncc

def normalized_cross_correlation(image1, image2):
	mean1 = image1.mean([2,3],keepdims=True)
	mean2 = image2.mean([2,3],keepdims=True) 
	std1 = image1.std([2,3],unbiased=False,keepdims=True)
	std2 = image2.std([2,3],unbiased=False,keepdims=True)
	eps=1e-8
	bs, ts, *sh = image1.shape
	N = sh[0]*sh[1]
	im1b = ((image1-mean1)/(std1*N+eps)).view(bs*ts, sh[0], sh[1])
	im2b = ((image2-mean2)/(std2+eps)).reshape(bs*ts, sh[0], sh[1])
	padding = tuple(side // 10 for side in sh)
	result = F.conv2d(im1b[None], im2b[:,None], padding=padding, bias=None, groups=bs*ts)
	ncc = result.view(bs*ts, -1).max(1)[0].view(bs, ts)
	return ncc