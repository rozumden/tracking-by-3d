import torch
import torch.nn as nn
from main_settings import *
import kaolin
from models.kaolin_wrapper import *
from kornia.geometry.conversions import angle_axis_to_rotation_matrix, quaternion_to_angle_axis, rotation_matrix_to_quaternion
from kornia.geometry.conversions import quaternion_to_rotation_matrix, rotation_matrix_to_angle_axis, QuaternionCoeffOrder
from kornia.morphology import erosion, dilation
from kornia.filters import GaussianBlur2d
from my_utils import *
import math

def deringing(coeffs, window):
    deringed_coeffs = torch.zeros_like(coeffs)
    deringed_coeffs[:, 0] += coeffs[:, 0]
    deringed_coeffs[:, 1:1 + 3] += \
        coeffs[:, 1:1 + 3] * math.pow(math.sin(math.pi * 1.0 / window) / (math.pi * 1.0 / window), 4.0)
    deringed_coeffs[:, 4:4 + 5] += \
        coeffs[:, 4:4 + 5] * math.pow(math.sin(math.pi * 2.0 / window) / (math.pi * 2.0 / window), 4.0)
    return deringed_coeffs

class RenderingKaolin(nn.Module):
    def __init__(self, config, faces, width, height):
        super().__init__()
        self.config = config
        self.height = height
        self.width = width
        camera_proj = kaolin.render.camera.generate_perspective_projection(1.57/2, self.width/self.height ) # 45 degrees
        self.register_buffer('camera_proj', camera_proj)
        self.register_buffer('camera_trans', torch.Tensor([0,0,self.config["camera_distance"]])[None])
        self.register_buffer('obj_center', torch.zeros((1,3)))
        camera_up_direction = torch.Tensor((0,1,0))[None]
        camera_rot,_ = kaolin.render.camera.generate_rotate_translate_matrices(self.camera_trans, self.obj_center, camera_up_direction)
        self.register_buffer('camera_rot', camera_rot)
        self.set_faces(faces)
            
    def set_faces(self, faces):
        self.register_buffer('faces', torch.LongTensor(faces))

    def forward(self, translation, quaternion, unit_vertices, face_features, texture_maps=None, render_depth=False):
        kernel = torch.ones(self.config["erode_renderer_mask"], self.config["erode_renderer_mask"]).to(translation.device)
        
        all_renders = []
        for frmi in range(quaternion.shape[1]):
            translation_vector = translation[:,:,frmi]
            rotation_matrix = quaternion_to_rotation_matrix(quaternion[:,frmi], order=QuaternionCoeffOrder.WXYZ) 
            
            renders = []
            vertices = kaolin.render.camera.rotate_translate_points(unit_vertices, rotation_matrix, self.obj_center) 
            vertices = vertices + translation_vector
            face_vertices_cam, face_vertices_image, face_normals = prepare_vertices(vertices, self.faces, self.camera_rot, self.camera_trans, self.camera_proj)
            face_vertices_z = face_vertices_cam[:,:,:,-1]
            face_normals_z = face_normals[:,:,-1]
            ren_features, ren_mask, red_index = kaolin.render.mesh.dibr_rasterization(self.height, self.width, face_vertices_z, face_vertices_image, face_features, face_normals_z, sigmainv=self.config["sigmainv"], boxlen=0.02, knum=30, multiplier=1000)
            if not texture_maps is None:
                ren_features = kaolin.render.mesh.texture_mapping(ren_features, texture_maps, mode='bilinear')
            result = ren_features.permute(0,3,1,2)
            if self.config["erode_renderer_mask"] > 0:
                ren_mask = erosion(ren_mask[:,None], kernel)[:,0]
            if render_depth:
                depth_map = face_vertices_z[0, red_index, :].mean(3)[:,None]
                result_rgba = torch.cat((result,ren_mask[:,None],depth_map),1)
            else:
                result_rgba = torch.cat((result,ren_mask[:,None]),1)
            renders.append(result_rgba)
            renders = torch.stack(renders,1).contiguous()
            all_renders.append(renders)
        all_renders = torch.stack(all_renders,1).contiguous()
        return all_renders

    def get_rgb_texture(self, translation, quaternion, unit_vertices, face_features, input_batch):
        kernel = torch.ones(self.config["erode_renderer_mask"], self.config["erode_renderer_mask"]).to(translation.device)
        tex = torch.zeros(1,3,self.config["texture_size"],self.config["texture_size"])
        cnt = torch.zeros(self.config["texture_size"],self.config["texture_size"])
        for frmi in range(quaternion.shape[1]):
            translation_vector = translation[:,:,frmi]
            rotation_matrix = quaternion_to_rotation_matrix(quaternion[:,frmi], order=QuaternionCoeffOrder.WXYZ) 
            vertices = kaolin.render.camera.rotate_translate_points(unit_vertices, rotation_matrix, self.obj_center) 
            vertices = vertices + translation_vector
            face_vertices_cam, face_vertices_image, face_normals = prepare_vertices(vertices, self.faces, self.camera_rot, self.camera_trans, self.camera_proj)
            face_vertices_z = face_vertices_cam[:,:,:,-1]
            face_normals_z = face_normals[:,:,-1]
            ren_features, ren_mask, red_index = kaolin.render.mesh.dibr_rasterization(self.height, self.width, face_vertices_z, face_vertices_image, face_features, face_normals_z, sigmainv=self.config["sigmainv"], boxlen=0.02, knum=30, multiplier=1000)
            coord = torch.round((1-ren_features)*self.config["texture_size"]).to(int)
            coord[coord >= self.config["texture_size"]] = self.config["texture_size"] - 1
            coord[coord < 0] = 0
            xc = coord[0,:,:,1].reshape([coord.shape[1]*coord.shape[2]])
            yc = (self.config["texture_size"]-1-coord[0,:,:,0]).reshape([coord.shape[1]*coord.shape[2]])
            cr = input_batch[0,frmi,0].reshape([coord.shape[1]*coord.shape[2]])
            cg = input_batch[0,frmi,1].reshape([coord.shape[1]*coord.shape[2]])
            cb = input_batch[0,frmi,2].reshape([coord.shape[1]*coord.shape[2]])
            for ki in range(xc.shape[0]):
                cnt[xc[ki],yc[ki]] = cnt[xc[ki],yc[ki]] + 1
                tex[0,0,xc[ki],yc[ki]] = tex[0,0,xc[ki],yc[ki]] + cr[ki]
                tex[0,1,xc[ki],yc[ki]] = tex[0,1,xc[ki],yc[ki]] + cg[ki]
                tex[0,2,xc[ki],yc[ki]] = tex[0,2,xc[ki],yc[ki]] + cb[ki]
        tex_final = tex / cnt[None,None]
        return tex_final

def generate_rotation(rotation_current, my_rot, steps=3):
    step = angle_axis_to_rotation_matrix(torch.Tensor([my_rot])).to(rotation_current.device)
    step_back = angle_axis_to_rotation_matrix(torch.Tensor([-np.array(my_rot)])).to(rotation_current.device)
    for ki in range(steps):
        rotation_current = torch.matmul(rotation_current, step_back)
    rotation_matrix_join = torch.cat((step[None],rotation_current[None]),1)[None]
    return rotation_matrix_join

def generate_all_views(best_model, static_translation, rotation_matrix, rendering, small_step, extreme_step=None, num_small_steps=1):
    rendering.config["fmo_steps"] = 2
    if not extreme_step is None:
        ext_renders = rendering(static_translation, generate_rotation(rotation_matrix,extreme_step,0), best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
        ext_renders_neg = rendering(static_translation, generate_rotation(rotation_matrix,-np.array(extreme_step),0), best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    rendering.config["fmo_steps"] = num_small_steps+1
    renders = rendering(static_translation, generate_rotation(rotation_matrix,small_step,0), best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    renders_neg = rendering(static_translation, generate_rotation(rotation_matrix,-np.array(small_step),0), best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    if not extreme_step is None:
        all_renders = torch.cat((ext_renders_neg[:,:,-1:], torch.flip(renders_neg[:,:,1:],[2]), renders, ext_renders[:,:,-1:]),2)
    else:
        all_renders = torch.cat((torch.flip(renders_neg[:,:,1:],[2]), renders),2)
    return all_renders.detach().cpu().numpy()[0,0].transpose(2,3,1,0)
  

def generate_novel_views(best_model, config):
    width = best_model["renders"].shape[-1]
    height = best_model["renders"].shape[-2]
    config["erode_renderer_mask"] = 7
    config["fmo_steps"] = best_model["renders"].shape[-4]
    rendering = RenderingKaolin(config, best_model["faces"], width, height).to(best_model["translation"].device)
    static_translation = best_model["translation"].clone()
    static_translation[:,:,:,1] = static_translation[:,:,:,1] + 0.5*static_translation[:,:,:,0]
    static_translation[:,:,:,0] = 0

    quaternion = best_model["quaternion"][:,:1].clone()
    rotation_matrix = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(quaternion[:,0,1]))    
    rotation_matrix_step = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(quaternion[:,0,0])/config["fmo_steps"]/2)
    for ki in range(int(config["fmo_steps"]/2)): rotation_matrix = torch.matmul(rotation_matrix, rotation_matrix_step)
    
    vertical =   generate_all_views(best_model, static_translation, rotation_matrix, rendering, [math.pi/2/9,0,0], [math.pi/3,0,0], 3)
    horizontal = generate_all_views(best_model, static_translation, rotation_matrix, rendering, [0,math.pi/2/9,math.pi/2/9], [0,math.pi/3,math.pi/3], 3)
    joint = generate_all_views(best_model, static_translation, rotation_matrix, rendering, [math.pi/2/9,math.pi/2/9,math.pi/2/9], [math.pi/3,math.pi/3,math.pi/3], 3)

    # steps = 1
    # config["fmo_steps"] = 2*steps+1
    # rot_joint = generate_rotation(rotation_matrix,[math.pi/3,0,0],steps)
    # hor_ext_renders = rendering(static_translation, rot_joint, best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    # rot_joint = generate_rotation(rotation_matrix,[0,math.pi/3,math.pi/3],steps)
    # ver_ext_renders = rendering(static_translation, rot_joint, best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    
    # steps = 3
    # config["fmo_steps"] = 2*steps+1
    # rot_joint = generate_rotation(rotation_matrix,[math.pi/2/9,0,0],steps)
    # hor_renders = rendering(static_translation, rot_joint, best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    # rot_joint = generate_rotation(rotation_matrix,[0,math.pi/2/9,math.pi/2/9],steps)
    # ver_renders = rendering(static_translation, rot_joint, best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    
    # vertical = torch.cat((ver_ext_renders[:,:,:1],ver_renders,ver_ext_renders[:,:,-1:]),2)
    # horizontal = torch.cat((hor_ext_renders[:,:,:1],hor_renders,hor_ext_renders[:,:,-1:]),2)

    # renders = rendering(best_model["translation"], best_model["quaternion"], best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    # save_image(renders[0,0], 'orig.png')
    # save_image(horizontal[0,0], 'hor.png')
    # save_image(vertical[0,0], 'ver.png')
    # save_image(depth[0,0], 'depth.png')

    return horizontal, vertical, joint


def generate_video_views(best_model, config):
    width = best_model["renders"].shape[-1]
    height = best_model["renders"].shape[-2]
    config["erode_renderer_mask"] = 7
    config["fmo_steps"] = best_model["renders"].shape[-4]
    rendering = RenderingKaolin(config, best_model["faces"], width, height).to(best_model["translation"].device)
    static_translation = best_model["translation"].clone()
    static_translation[:,:,:,1] = static_translation[:,:,:,1] + 0.5*static_translation[:,:,:,0]
    static_translation[:,:,:,0] = 0

    quaternion = best_model["quaternion"][:,:1].clone()
    rotation_matrix = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(quaternion[:,0,1]))    
    rotation_matrix_step = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(quaternion[:,0,0])/config["fmo_steps"]/2)
    for ki in range(int(config["fmo_steps"]/2)): rotation_matrix = torch.matmul(rotation_matrix, rotation_matrix_step)
    
    views = generate_all_views(best_model, static_translation, rotation_matrix, rendering, [math.pi/2/9/10,0,0], None, 45)
    return views

def generate_tsr_video(best_model, config, steps = 8):
    width = best_model["renders"].shape[-1]
    height = best_model["renders"].shape[-2]
    config["erode_renderer_mask"] = 7
    config["fmo_steps"] = steps
    rendering = RenderingKaolin(config, best_model["faces"], width, height).to(best_model["translation"].device)
    renders = rendering(best_model["translation"], best_model["quaternion"], best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    tsr = renders.detach().cpu().numpy()[0,0].transpose(2,3,1,0)
    return tsr