import open3d as o3d
import torch.utils.data
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from easy_utils import sample_points

def print_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        # print('Learning rate: ', param_group['lr'])
        return param_group['lr']
    
class PoseEstimator(torch.nn.Module):
    def __init__(self, num_parts, init_r, init_t, device):
        super(PoseEstimator, self).__init__()
        self.num_parts = num_parts
        self.device = device
      
        
        self.rot_quat_s = []
        self.tra_s = []
        for idx in range(self.num_parts):
            x, y, z, w = R.from_matrix(init_r[idx].cpu().numpy()).as_quat()
            rot_quat = torch.nn.Parameter(torch.tensor(
                [w, x, y, z], device=device, dtype=torch.float), requires_grad=True)  # q=a+bi+ci+di
            tra = torch.nn.Parameter(init_t[idx].reshape(3,1).clone().detach().to(device), requires_grad=True)
            self.rot_quat_s.append(rot_quat)
            self.tra_s.append(tra)

        self.rot_quat_s = nn.ParameterList([torch.nn.Parameter(torch.tensor( \
            [w, x, y, z], device=device, dtype=torch.float), requires_grad=True) for idx in range(self.num_parts)])
        self.tra_s = nn.ParameterList([torch.nn.Parameter(init_t[idx].reshape(3,1).clone().detach().to(device), requires_grad=True) for idx in range(self.num_parts)])
       
        
   
    def chamfer_distance(self, x, y):
        try:
            x = x.to(torch.float64)
            y = y.to(torch.float64)
            dist_matrix = torch.cdist(x, y)  # 计算x和y的二范数

            min_dist_x_to_y, _ = torch.min(dist_matrix, dim=1)
            Dxy = torch.mean(min_dist_x_to_y, dim=0)

            min_dist_y_to_x, _ = torch.min(dist_matrix, dim=0)
            Dyy = torch.mean(min_dist_y_to_x, dim=0)

            chamfer = torch.mean(Dxy + Dyy)

            return chamfer

        except:
            print(x.shape, y.shape)
            tensor = torch.tensor(1.0, dtype=torch.float64, device='cuda:0', requires_grad=True)
            return tensor
  

    def forward(self, camera_pts, cad_pts, part_weight):
        all_objective = 0.
        e_geo = 0.
        transforms = []
        
      
        scad_pts = cad_pts
        scamera_pts = camera_pts


        scad_pts = [torch.cat([pts.to(self.device), torch.ones(pts.shape[0], 1, device=self.device)], dim=-1) for pts in scad_pts]
        scamera_pts = [torch.cat([pts.to(self.device), torch.ones(pts.shape[0], 1, device=self.device)], dim=-1) for pts in scamera_pts]
        for idx in range(self.num_parts):

            base_r_quat = self.rot_quat_s[idx] / torch.norm(self.rot_quat_s[idx])
            a, b, c, d = base_r_quat[0], base_r_quat[1], base_r_quat[2], base_r_quat[3]  # q=a+bi+ci+di
            base_rot_matrix = torch.stack([1 - 2 * c * c - 2 * d * d, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                                        2 * b * c + 2 * a * d, 1 - 2 * b * b - 2 * d * d, 2 * c * d - 2 * a * b,
                                        2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d,
                                        1 - 2 * b * b - 2 * c * c]).reshape(3, 3)
            base_transform = torch.cat([torch.cat([base_rot_matrix, self.tra_s[idx]], dim=1),
                                        torch.tensor([[0., 0., 0., 1.]], device=self.device)], dim=0).float()
            transforms.append(base_transform)
            cad_base = base_transform.matmul(scad_pts[idx].T).T
            camera_base = scamera_pts[idx]
            # base_objective = self.E_geo(base_transform, scamera_pts[idx], scad_pts[idx])
            base_objective = self.chamfer_distance(cad_base, camera_base)
            # if not choosen_opt[idx]:
            #     base_objective = 0.

            all_objective += part_weight[idx] * base_objective
        
        transforms = torch.stack(transforms, axis=0)
        return all_objective, transforms.detach()


def optimize_pose(estimator, camera_pts, cad_pts, part_weight, k=False):
    """
    使用观测点云和预测点云计算chamfer_distance, 将变换矩阵参数化, 进行优化更新
    pose_estimator = pose_optimizer.PoseEstimator(num_parts=n_parts, init_r=init_base_r, init_t=init_base_t,
                                             device=device, joint_type='revolute')

    loss, optim_transforms = pose_optimizer.optimize_pose(pose_estimator, xyz_camera, cad, part_weight)
    
    """
    estimator.rot_quat_s.requires_grad_(True)
    estimator.tra_s.requires_grad_(True)
   

    lr = 0.1
    MAX_EPOCH = 50 # 100
    et_lr = 0.01
    
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=et_lr)
    best_loss = float('inf')
    current_lr = lr
    transforms = None
    
    for iter in range(MAX_EPOCH):
        loss, transform = estimator(camera_pts, cad_pts, part_weight)
        
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            transforms = transform
        if best_loss <= 0.3:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # change lr
        current_lr = print_learning_rate(optimizer)
        scheduler.step()
 
    return loss, transforms