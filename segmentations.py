import torch
import numpy
from torchvision import transforms
import cv2 as cv
from scipy.ndimage import uniform_filter
from my_utils import *
from skimage.measure import label, regionprops
from kornia.filters import gaussian_blur2d, spatial_gradient
import glob
from skimage.measure import label
from scipy import ndimage

import sys

sys.path.insert(0, './OSTrack')
from lib.test.evaluation import Tracker
from lib.test.tracker.ostrack import OSTrack
from lib.test.parameter.ostrack import parameters



def compute_segments(segment, I, width, height):
    segment = cv2.resize(segment, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    It = transforms.ToTensor()(I/255.0)
    image = It.unsqueeze(0).float()
    segm = transforms.ToTensor()(segment).unsqueeze(0)
    weights = compute_weights(image)
    segments = torch.cat((weights*segm,segm),1)
    return segments

def compute_segments_dist(segment, width, height):
    segment = cv2.resize(segment, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    segm = transforms.ToTensor()(segment).unsqueeze(0)
    dist = ndimage.distance_transform_edt(1-segment)
    distt = transforms.ToTensor()(dist).unsqueeze(0)
    segments = torch.cat((distt,segm),1)
    return segments

class PrecomputedTracker():
    def __init__(self, perc, max_width, baseline_dict, grabcut=False):
        self.perc = perc
        self.max_width = max_width
        self.baseline_dict = baseline_dict
        self.shape = None
        self.grabcut = grabcut
        self.background_mdl = np.zeros((1,65), np.float64)
        self.foreground_mdl = np.zeros((1,65), np.float64)

    def process(self, I, ind):
        self.shape = I.shape
        if self.max_width / I.shape[1] < self.perc:
            self.perc = self.max_width / I.shape[1]
        segment = cv.resize(imread(self.baseline_dict[ind]), self.shape[1::-1]).astype(np.float64)
        if len(segment.shape) > 2:
            segment = segment[:,:,:1]
        segment = (segment > 0.5).astype(segment.dtype)
        width = int(self.shape[1] * self.perc)
        height = int(self.shape[0] * self.perc)
        I = cv2.resize(I, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        I = uniform_filter(I, size=(3, 3, 1))
        if self.grabcut:
            mask = segment.astype('uint8')*cv2.GC_PR_FGD
            mask[mask==0] = cv2.GC_PR_BGD
            cv2.grabCut(I.astype('uint8'), mask, None, self.background_mdl, self.foreground_mdl, 5, cv2.GC_INIT_WITH_MASK)
            segment = np.where((mask==cv2.GC_PR_BGD)|(mask==cv2.GC_BGD),0,1).astype('uint8').astype(np.float64)
        It = transforms.ToTensor()(I/255.0)
        image = It.unsqueeze(0).float()
        segments = compute_segments_dist(segment, width, height)
        return image, segments

    def process_segm(self, segm_path):
        segment = cv.resize(imread(segm_path), self.shape[1::-1]).astype(np.float64)
        width = int(self.shape[1] * self.perc)
        height = int(self.shape[0] * self.perc)
        segment = cv2.resize(segment, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        
        labels = label(segment)
        if labels.max() > 0:
            segment = (labels == np.argmax(np.bincount(labels.flat)[1:])+1)

        segm = transforms.ToTensor()(segment)
        return segm

    def init_bbox(self, file0, bbox0, init_mask=None):
        image, segments = self.next(file0)
        return image, segments, self.perc

    def next(self, file):
        ind = os.path.splitext(os.path.basename(file))[0]
        # ind = (file[-12:-4])
        # if ind[0] == '0':
        #     ind = int(ind)
        I = imread(file)*255
        image, segments = self.process(I, ind)
        return image, segments


class MyTracker():
    def __init__(self, perc, max_width, grabcut):
        sys.path.insert(0, './d3s')
        from pytracking.tracker.segm import Segm
        from pytracking.parameter.segm import default_params as vot_params
        params = vot_params.parameters()
        self.tracker = Segm(params)
        self.perc = perc
        self.shape = None
        self.max_width = max_width
        self.grabcut = grabcut
        self.background_mdl = np.zeros((1,65), np.float64)
        self.foreground_mdl = np.zeros((1,65), np.float64)

    def process(self, I):
        self.shape = I.shape
        if self.max_width / I.shape[1] < self.perc:
            self.perc = self.max_width / I.shape[1]
        segment = cv.resize(self.tracker.mask, I.shape[1::-1]).astype(np.float64)
        width = int(I.shape[1] * self.perc)
        height = int(I.shape[0] * self.perc)
        I = cv2.resize(I, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        I = uniform_filter(I, size=(3, 3, 1))
        if self.grabcut:
            mask = segment.astype('uint8')*cv2.GC_PR_FGD
            cv2.grabCut(I.astype('uint8'), mask, None, self.background_mdl, self.foreground_mdl, 5, cv2.GC_INIT_WITH_MASK)
            segment = np.where((mask==2)|(mask==0),0,1).astype('uint8').astype(np.float64)
        It = transforms.ToTensor()(I/255.0)
        image = It.unsqueeze(0).float()
        segments = compute_segments_dist(segment, width, height)
        return image, segments
    
    def process_segm(self, segm_path):
        segment = cv.resize(imread(segm_path), self.shape[1::-1]).astype(np.float64)
        width = int(self.shape[1] * self.perc)
        height = int(self.shape[0] * self.perc)
        segment = cv2.resize(segment, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        
        segm = transforms.ToTensor()(segment)
        return segm

    def init_bbox(self, file0, bbox0, init_mask=None):
        I = imread(file0)*255
        self.tracker.initialize(I, bbox0, init_mask=init_mask)
        image, segments = self.process(I)
        return image, segments, self.perc

    def next(self, file):
        I = imread(file)*255
        prediction = self.tracker.track(I)
        image, segments = self.process(I)
        return image, segments

class CSRTrack():
    def __init__(self, perc, max_width, grabcut):
        params = vot_params.parameters()
        self.tracker = cv2.TrackerCSRT_create()
        self.perc = perc
        self.max_width = max_width
        self.grabcut = grabcut
        self.background_mdl = np.zeros((1,65), np.float64)
        self.foreground_mdl = np.zeros((1,65), np.float64)
        self.shape = None

    def process(self, I, bbox0):
        self.shape = I.shape
        bbox = (bbox0 + np.array([0,0,bbox0[0],bbox0[1]]))
        if self.max_width / I.shape[1] < self.perc:
            self.perc = self.max_width / I.shape[1]
        segment = np.zeros((I.shape[0],I.shape[1])).astype(np.float64)
        segment[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
        width = int(I.shape[1] * self.perc)
        height = int(I.shape[0] * self.perc)
        I = cv2.resize(I.astype(np.float64), dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        I = uniform_filter(I, size=(3, 3, 1))
        if self.grabcut:
            mask = segment.astype('uint8')
            cv2.grabCut(I.astype('uint8'), mask, bbox0, self.background_mdl, self.foreground_mdl, 5, cv2.GC_INIT_WITH_RECT)
            segment = np.where((mask==2)|(mask==0),0,1).astype('uint8').astype(np.float64)
        It = transforms.ToTensor()(I/255.0)
        image = It.unsqueeze(0).float()
        segments = compute_segments_dist(segment, width, height)
        return image, segments

    def init_bbox(self, file0, bbox0, init_mask=None):
        I = (imread(file0)*255).astype(np.uint8)
        bbox = bbox0.astype(int)
        self.tracker.init(I, bbox)
        image, segments = self.process(I, bbox)
        return image, segments, self.perc

    def next(self, file):
        I = (imread(file)*255).astype(np.uint8)
        ok, bbox0 = self.tracker.update(I)
        image, segments = self.process(I, bbox0)
        return image, segments


def get_ar(img, init_box, ar_path):
    """ set up Alpha-Refine """
    sys.path.insert(0, './AlphaRefine')
    from pytracking.refine_modules.refine_module import RefineModule
    selector_path = 0
    sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default
    RF_module = RefineModule(ar_path, selector_path, search_factor=sr, input_sz=input_sz)
    RF_module.initialize(img, np.array(init_box))
    return RF_module

class OSTracker():
    def __init__(self, perc, max_width, grabcut):
        params = parameters("vitb_384_mae_ce_32x4_ep300")
        params.debug = 0
        params.tracker_name = "ostrack"
        params.param_name = "vitb_384_mae_ce_32x4_ep300"
        self.tracker = OSTrack(params, "video")
        self.perc = perc
        self.max_width = max_width
        self.RF_module = None
        self.shape = None

    def process(self, I, bbox0, segment):
        self.shape = I.shape
        bbox = (bbox0 + np.array([0,0,bbox0[0],bbox0[1]]))
        if self.max_width / I.shape[1] < self.perc:
            self.perc = self.max_width / I.shape[1]
        segment = (segment > 0.5).astype(np.float64)
        width = int(I.shape[1] * self.perc)
        height = int(I.shape[0] * self.perc)
        I = cv2.resize(I.astype(np.float64), dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        I = uniform_filter(I, size=(3, 3, 1))
        It = transforms.ToTensor()(I/255.0)
        image = It.unsqueeze(0).float()
        segments = compute_segments_dist(segment, width, height)
        return image, segments

    def process_segm(self, segm_path):
        segment = cv.resize(imread(segm_path), self.shape[1::-1]).astype(np.float64)
        width = int(self.shape[1] * self.perc)
        height = int(self.shape[0] * self.perc)
        segment = cv2.resize(segment, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        
        segm = transforms.ToTensor()(segment)
        return segm

    def init_bbox(self, file0, bbox0, init_mask=None):
        I = (imread(file0)*255).astype(np.uint8)
        bbox = np.array(bbox0).astype(int)
        self.tracker.initialize(I, {'init_bbox': bbox})
        self.RF_module = get_ar(I, bbox, g_alpha_model)
        segment = self.RF_module.get_mask(I, np.array(bbox0))
        image, segments = self.process(I, bbox, segment)
        return image, segments, self.perc

    def next(self, file):
        I = (imread(file)*255).astype(np.uint8)
        out = self.tracker.track(I)
        bbox0 = [int(s) for s in out['target_bbox']]
        # pred_bbox = self.RF_module.refine(I, np.array(bbox0))
        segment = self.RF_module.get_mask(I, np.array(bbox0))
        image, segments = self.process(I, bbox0, segment)
        return image, segments



def compute_weights(input_batch):
    blurry_input = gaussian_blur2d(input_batch[:,:3], kernel_size=tuple([9,9]), sigma=tuple([5,5]))
    grad_input = spatial_gradient(blurry_input)
    grad_input = (grad_input[:,:,0]**2 + grad_input[:,:,1]**2)**0.5
    grad_input = grad_input.sum(1)
    weights = (grad_input / grad_input.max())[:,None]
    weights = weights + 0.05
    weights = weights / weights.max()
    # inc_step = config["inc_step"]
    # for stepi in range(int(config["input_frames"]/inc_step)):
    #     st = stepi*inc_step
    #     en = (stepi+1)*inc_step

    #     # weights,_ = weights[st:en].max(1)

    #     weights_min,_ = weights[st:en].min(0)
    #     weights[st:en] = (weights[st:en] - weights_min[None,:]) + 0.01
    #     weights[st:en] = (weights[st:en] / weights[st:en].max())

    return weights

def segment_d3s_vot(files, bboxes):
    perc = 0.5
    tracker = None
    params = vot_params.parameters()
    input_batch = torch.Tensor([])
    hs_frames = torch.Tensor([])
    for ind, fl in enumerate(files):
        I = imread(fl)*255
        if tracker is None: 
            tracker = Segm(params)
            tracker.initialize(I, bboxes[ind])
        else:
            prediction = tracker.track(I)
        
        segment = cv.resize(tracker.mask, I.shape[1::-1]).astype(np.float64)

        width = int(I.shape[1] * perc)
        height = int(I.shape[0] * perc)
        I = cv2.resize(I, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        I = uniform_filter(I, size=(3, 3, 1))
        segment = cv2.resize(segment, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

        It = transforms.ToTensor()(I/255.0)
        hs_frames_one = torch.cat((It.clone(), transforms.ToTensor()(segment)), 0).unsqueeze(0).unsqueeze(0)

        input_batch_one = torch.cat((It, 0*It.clone()), 0).unsqueeze(0).float()
        input_batch = torch.cat( (input_batch, input_batch_one), 0)
        hs_frames = torch.cat( (hs_frames, hs_frames_one), 0)
    return input_batch, hs_frames

def segment_given(files, segms, perc = 0.1):
    input_batch = torch.Tensor([])
    segments = torch.Tensor([])
    for ind, fl in enumerate(files):
        I = imread(fl)*255
        segment = (imread(segms[ind])>0.5).astype(np.float64)[:,:,0]

        width = int(I.shape[1] * perc)
        height = int(I.shape[0] * perc)
        I = cv2.resize(I, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        I = uniform_filter(I, size=(3, 3, 1))
        segment = cv2.resize(segment, dsize=(width, height), interpolation=cv2.INTER_LINEAR)> 0.05

        It = transforms.ToTensor()(I/255.0)
        segment_one = torch.cat((It.clone(), transforms.ToTensor()(segment)), 0).unsqueeze(0).unsqueeze(0)

        input_batch_one = torch.cat((It, 0*It.clone()), 0).unsqueeze(0).float()
        input_batch = torch.cat( (input_batch, input_batch_one), 0)
        segments = torch.cat( (segments, segment_one), 0)
    return input_batch, segments

def get_length(cdtb_folder, seqs):
    lens = np.zeros(seqs.shape, dtype=int)
    for ki in range(seqs.shape[0]):
        deformable = np.loadtxt(os.path.join(cdtb_folder, seqs[ki], 'deformable.tag'),delimiter='\n',dtype=int)
        reflective = np.loadtxt(os.path.join(cdtb_folder, seqs[ki], 'reflective-target.tag'),delimiter='\n',dtype=int)
        full_occlusion = np.loadtxt(os.path.join(cdtb_folder, seqs[ki], 'full-occlusion.tag'),delimiter='\n',dtype=int)
        occlusion = np.loadtxt(os.path.join(cdtb_folder, seqs[ki], 'occlusion.tag'),delimiter='\n',dtype=int)
        partial_occlusion = np.loadtxt(os.path.join(cdtb_folder, seqs[ki], 'partial-occlusion.tag'),delimiter='\n',dtype=int)
        out_frame = np.loadtxt(os.path.join(cdtb_folder, seqs[ki], 'out-of-frame.tag'),delimiter='\n',dtype=int)
        non_acceptable = deformable+reflective+out_frame+occlusion+full_occlusion+partial_occlusion
        nonzero = np.nonzero(non_acceptable)[0]
        if nonzero.shape[0] == 0:
            lens[ki] = deformable.shape[0]
        else:
            lens[ki] = nonzero[0]
    return lens

def get_length_st(cdtb_folder, seqs):
    lens = np.zeros(seqs.shape, dtype=int)
    for ki in range(seqs.shape[0]):
        full_occlusion = np.loadtxt(os.path.join(cdtb_folder, seqs[ki], 'full-occlusion.tag'),delimiter='\n',dtype=int)
        out_frame = np.loadtxt(os.path.join(cdtb_folder, seqs[ki], 'out-of-frame.tag'),delimiter='\n',dtype=int)
        non_acceptable = out_frame+full_occlusion
        nonzero = np.nonzero(non_acceptable)[0]-5
        if nonzero.shape[0] == 0:
            lens[ki] = non_acceptable.shape[0]
        else:
            lens[ki] = nonzero[0]
    return lens

def get_length_full(cdtb_folder, seqs):
    lens = np.zeros(seqs.shape, dtype=int)
    for ki in range(seqs.shape[0]):
        lens[ki] = len(glob.glob(os.path.join(cdtb_folder, seqs[ki], 'color', '*')))
    return lens


def rle_to_mask(rle, width, height):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0
    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_+j] = 1
        idx_ += rle[i]

    # reshape vector into 2-D mask
    # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
    return np.array(v, dtype=np.uint8).reshape((height, width))

def create_mask_from_string(mask_encoding):
    """
    mask_encoding: a string in the following format: x0, y0, w, h, RLE
    output: mask, offset
    mask: 2-D binary mask, size defined in the mask encoding
    offset: (x, y) offset of the mask in the image coordinates
    """
    elements = [int(el) for el in mask_encoding]
    tl_x, tl_y, region_w, region_h = elements[:4]
    rle = np.array([el for el in elements[4:]], dtype=np.int32)

    # create mask from RLE within target region
    mask = rle_to_mask(rle, region_w, region_h)

    return mask, (tl_x, tl_y)

def get_bbox(segments):
    all_segments = segments[0,:,1].sum(0) > 0
    nzeros = torch.nonzero(all_segments,as_tuple=True)
    pix_offset = 20
    x0 = max(0, nzeros[0].min().item() - pix_offset)
    x1 = min(all_segments.shape[0]-1, nzeros[0].max().item() + pix_offset)
    y0 = max(0, nzeros[1].min().item() - pix_offset)
    y1 = min(all_segments.shape[1]-1, nzeros[1].max().item() + pix_offset)
    if x1-x0 > y1-y0:
        addall = (x1-x0) - (y1-y0)
        add0 = int(addall/2)
        add1 = addall - add0
        y0 = max(0, y0 - add0)
        y1 = min(all_segments.shape[1]-1, y1 + add1)
    else:
        addall = (y1-y0) - (x1-x0)
        add0 = int(addall/2)
        add1 = addall - add0
        x0 = max(0, x0 - add0)
        x1 = min(all_segments.shape[0]-1, x1 + add1)
    bounds = [x0,x1,y0,y1]
    return bounds

# def segment_grabcut(files):
#     perc = 0.2
#     input_batch = torch.Tensor([])
#     hs_frames = torch.Tensor([])
#     for fl in files:
#         I = imread(fl)

#         # mask = np.zeros(I.shape[:2],np.uint8)
#         bgdModel = np.zeros((1,65),np.float64)
#         fgdModel = np.zeros((1,65),np.float64)
#         # mask, bgdModel, fgdModel = cv.grabCut(I.astype(np.uint8),mask,(100,200,500,750),bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

#         mask = cv.GC_PR_BGD*np.ones(I.shape[:2],np.uint8)
#         mask[500:900,150:400] = cv.GC_FGD
#         mask[0:200] = cv.GC_BGD
#         mask[950:] = cv.GC_BGD
#         mask[:,:70] = cv.GC_BGD
#         mask[:,-200:] = cv.GC_BGD
#         mask, bgdModel, fgdModel = cv.grabCut(I.astype(np.uint8),mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
#         segment = ((mask == cv.GC_PR_FGD) | (mask == cv.GC_FGD)).astype(np.float64)
#         width = int(I.shape[1] * perc)
#         height = int(I.shape[0] * perc)
#         I = cv2.resize(I, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
#         I = uniform_filter(I, size=(3, 3, 1))
#         segment = cv2.resize(segment, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

#         It = transforms.ToTensor()(I/255.0)
#         hs_frames_one = torch.cat((It.clone(), transforms.ToTensor()(segment)), 0).unsqueeze(0).unsqueeze(0)

#         input_batch_one = torch.cat((It, 0*It.clone()), 0).unsqueeze(0).float()
#         input_batch = torch.cat( (input_batch, input_batch_one), 0)
#         hs_frames = torch.cat( (hs_frames, hs_frames_one), 0)
#     return input_batch, hs_frames


# def segment_d3s(files):
#     perc = 1

#     params = vot_params.parameters()
#     tracker = Segm(params)
#     gt_rect = None
#     input_batch = torch.Tensor([])
#     hs_frames = torch.Tensor([])
#     for fl in files:
#         I = imread(fl)
#         if gt_rect is None: 
#             mask = np.zeros(I.shape[:2],np.uint8)
#             bgdModel = np.zeros((1,65),np.float64)
#             fgdModel = np.zeros((1,65),np.float64)
#             mask, bgdModel, fgdModel = cv.grabCut(I.astype(np.uint8),mask,(100,200,500,750),bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
#             segment = ((mask == cv.GC_PR_FGD) | (mask == cv.GC_FGD)).astype(np.float64)
#             regions = regionprops(label(segment))
#             ind = -1
#             maxarea = 0
#             for ki in range(len(regions)):
#                 if regions[ki].area > maxarea:
#                     ind = ki
#                     maxarea = regions[ki].area
#             gt_rect = np.array(regions[ind].bbox)
#             tracker.initialize(I, gt_rect, init_mask=segment)
#         else:
#             prediction = tracker.track(I)
#             segment = cv.resize(tracker.mask, I.shape[1::-1]).astype(np.float64)
            
#         width = int(I.shape[1] * perc)
#         height = int(I.shape[0] * perc)
#         I = cv2.resize(I, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
#         I = uniform_filter(I, size=(3, 3, 1))
#         segment = cv2.resize(segment, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

#         It = transforms.ToTensor()(I/255.0)
#         hs_frames_one = torch.cat((It.clone(), transforms.ToTensor()(segment)), 0).unsqueeze(0).unsqueeze(0)

#         input_batch_one = torch.cat((It, 0*It.clone()), 0).unsqueeze(0).float()
#         input_batch = torch.cat( (input_batch, input_batch_one), 0)
#         hs_frames = torch.cat( (hs_frames, hs_frames_one), 0)
#     return input_batch, hs_frames
