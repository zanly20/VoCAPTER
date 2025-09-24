import os

import sys

import cupy as cp
import math
import time
import open3d as o3d
import torch
import numpy as np
import copy
import os.path as osp
import argparse
import mmcv
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
import tqdm
import omegaconf
from utils.dataset_inference import ArtImage_infer
from utils.util import backproject, fibonacci_sphere, convert_layers, estimate_normals
from models.model import VoCAPTER, PointEncoder
from models.voting import rot_voting_kernel, backvote_kernel, ppf_kernel
from utils import pose_optimizer
import sympy as sp

CLASSES = ['laptop', 'eyeglasses', 'dishwasher', 'drawer', 'scissors']

def RotateAnyAxis_np(step):  
    
    v1 = np.array([0.59978752, -0.45188404, -0.0555921])
    axis = np.array([-1.,0.,0.])

    a, b, c = v1[0], v1[1], v1[2]    
    u, v, w = axis[0], axis[1], axis[2]  

    cos = np.cos(-step)  
    sin = np.sin(-step)

    
    rot = np.concatenate([np.stack([u*u+(v*v+w*w)*cos, u*v*(1-cos)-w*sin, u*w*(1-cos)+v*sin,
                                                   (a*(v*v+w*w)-u*(b*v+c*w))*(1-cos)+(b*w-c*v)*sin,
                                                   u*v*(1-cos)+w*sin, v*v+(u*u+w*w)*cos, v*w*(1-cos)-u*sin,
                                                   (b*(u*u+w*w)-v*(a*u+c*w))*(1-cos)+(c*u-a*w)*sin,
                                                   u*w*(1-cos)-v*sin, v*w*(1-cos)+u*sin, w*w+(u*u+v*v)*cos,
                                                   (c*(u*u+v*v)-w*(a*u+b*v))*(1-cos)+(a*v-b*u)*sin]).reshape(3, 4),
                                                   np.array([[0., 0., 0., 1.]])], axis=0)

    return rot

def calErr(pred_r, pred_t, gt_r, gt_t, sort_part):
    num_parts = gt_r.shape[0]
    r_errs = []
    t_errs = []
    for i in range(num_parts):
        if i in [0, sort_part]:
            r_err = rot_diff_degree(gt_r[i], pred_r[i])
            if r_err > 90:
                r_err = 180 - r_err
            if r_err > 45:
                r_err = 90 - r_err
            t_err = np.linalg.norm(gt_t[i] - pred_t[i])   # 平方距离（L2范数）
            r_errs.append(r_err)
            t_errs.append(t_err)
    return r_errs, t_errs


#####################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--cp_device', type=int, default=0, help='GPU device number for custom voting algorithms')
    parser.add_argument('--ckpt_path', default='/yourpath/checkpoint/dishwasher', help='Model checkpoint path')
    parser.add_argument('--num_parts', type=int, default=2, help='Angle precision in orientation voting')
    parser.add_argument('--angle_prec', type=float, default=1.5, help='Angle precision in orientation voting')
    parser.add_argument('--num_rots', type=int, default=72, help='Number of candidate center votes generated for a given point pair')
    parser.add_argument('--n_threads', type=int, default=512, help='Number of cupy threads')
    parser.add_argument('--optim', default=True,action='store_true', help='Whether to use bbox mask instead of instance segmentations')
    parser.add_argument('--bbox_mask', action='store_true', help='Whether to use bbox mask instead of instance segmentations')
    parser.add_argument('--adaptive_voting',default=True, action='store_true', help='Whether to use adaptive center voting')
    
    
    args = parser.parse_args()
    
    cp_device = args.cp_device
    device = torch.device("cuda")
    num_parts = args.num_parts
    
    nepoch = 'best'
    # cfg文件不影响 里面就centric不一样，在推理的时候用不上
    cfg = omegaconf.OmegaConf.load(f'{args.ckpt_path}/.hydra/config.yaml')
    cls_name = cfg.category
    angle_tol = args.angle_prec
    num_samples = int(4 * np.pi / (angle_tol / 180 * np.pi))
    sphere_pts = np.array(fibonacci_sphere(num_samples))

    num_rots = args.num_rots
    n_threads = args.n_threads
    bcelogits = torch.nn.BCEWithLogitsLoss()
    compute_iou_3d = compute_3d_iou_size()

   
       
    point_encoder = PointEncoder(k=cfg.knn, spfcs=[32, 64, 32, 32], num_layers=1, out_dim=32).cuda().eval()
    ppf_encoder = VoCAPTER(ppffcs=[84, 32, 32, 16], out_dim=2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins + 2 + 3).cuda().eval()

    point_encoder.load_state_dict(torch.load(f'{args.ckpt_path}/point_encoder_epoch{nepoch}.pth'))
    ppf_encoder.load_state_dict(torch.load(f'{args.ckpt_path}/ppf_encoder_epoch{nepoch}.pth'))

      
    
    # para_ppf = {}
    # para_ppf['cfg'] = cfg  
    # para_ppf['cp_device'] = cp_device
    # para_ppf['num_rots'] = num_rots
    # para_ppf['n_threads'] = n_threads
    # para_ppf['adaptive_voting'] = args.adaptive_voting
    # para_ppf['sphere_pts'] = sphere_pts
    # para_ppf['angle_tol'] = angle_tol
    dataset = ArtImage_infer(cfg=cfg,
                        mode='val',
                        data_root='/home/yourpath/codes/dataset1',
                        num_pts=cfg.num_points,
                        num_cates=5,
                        num_parts=args.num_parts,
                        device='cpu',
                        )
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    k_fix = []
    test_count = 0

    sample_id = 0
  

    ini_base_r_error_all = 0.
    ini_sort_child_r_error_all = 0.
    ini_base_t_error_all = 0.
    ini_sort_child_t_error_all = 0.
    ini_j_state_error_all = 0.


    optim_base_r_error_all = 0.
    optim_sort_child_r_error_all = 0.
    optim_base_t_error_all = 0.
    optim_sort_child_t_error_all = 0.
    optim_j_state_error_all = 0.


    rt_key = [_ for _ in range(num_parts)]
    r_key = [_ for _ in range(num_parts)]
    t_key = [_ for _ in range(num_parts)]
    key_j_state_gt = torch.tensor(0.)
    turns = 0
    video_num = 0
    key_dis = 0
    gt_rt_list = []
   

    time1 = time.time()
    for i, data in enumerate(test_dataloader):  # batch_size=1， 每一个data仅仅包含一条数据，因此关键帧可以通过i来计算
        turns += 1
        # cloud 相机空间下
        cloud = data['cloud'][0].numpy()  # [num_parts, num_pts, 3]
        camera_cloud, cano_cloud = data['camera_cloud'], data['cano_cloud'][0].numpy() 
        camera_part, canos_part = data['camera_part'][0].numpy(), data['canos_part'][0].numpy()
        gt_part_r, gt_part_t, gt_part_rt = \
            data['part_target_r'],data['part_target_t'],data['part_target_rt']
        gt_joint_state = data['joint_state'][0].numpy()
        color_path = data['color_path'][0]
        sort_part = data['sort_part'][0]
        joint_loc = data['norm_joint_loc'][0].numpy()  # [3,]
        joint_axis = data['norm_joint_axis'][0].numpy()
        gt_size, mean_shape_part = data['fsnet_scale'].numpy(), data['mean_shape_part'].numpy()
         
        ori_gt_state = gt_joint_state 

        ori_gt_part_r = copy.deepcopy(gt_part_r[0].numpy())  # [2,3,3]
        ori_gt_part_t = copy.deepcopy(gt_part_t[0].numpy().reshape(2,3)) # reshape将[2,3,1]转换为[2,3]
        ori_gt_part_rt = [compose_rt(ori_gt_part_r[p_idx], ori_gt_part_t[p_idx]) for p_idx in range(num_parts)]

        
        # angles = np.array([np.random.uniform(-5., 5.),
        #                 np.random.uniform(-5., 5.),
        #                 np.random.uniform(-5., 5.)])
        # base_fix_t = np.array([np.random.uniform(-0.1, 0.1),
        #                             np.random.uniform(-0.1, 0.1),
        #                             np.random.uniform(-0.1, 0.1)])
        angles = np.array([0,0,0])
        base_fix_t = np.array([0,0,0])
        """
        Rotation.from_euler()函数
        利用欧拉角生成旋转矩阵
        如果 angles 是 [30, 45, 60]，那么这段代码将生成一个旋转矩阵，
        该矩阵表示首先绕 y 轴旋转 30 度，然后绕 x 轴旋转 45 度，最后绕 z 轴旋转 60 度的组合旋转。
        """
        base_fix_r = Rotation.from_euler('yxz', angles, degrees=True).as_matrix() # as_matrix用于将旋转对象转换为旋转矩阵
        base_fix_rt = compose_rt(base_fix_r, base_fix_t)
        flag = 0
        inner_index = i % 29
        
        print(f'i: {i}')
        print(f'inner_index: {inner_index}')
        pred_rt = None
        if inner_index == 0:  
            k_fix = base_fix_rt
            flag = 1
            pre_errdis = np.inf
            update_num=0
            r_key = copy.deepcopy(gt_part_r[0].numpy())  # [2,3,3]  绝对量
            t_key = copy.deepcopy(gt_part_t[0].numpy())  # [2,1,3]
            # print('gt_joint_state', gt_joint_state)
            key_j_state_gt = ori_gt_state   
            key_j_state = ori_gt_state   

            for part_idx in range(num_parts):
                rt_key[part_idx] = compose_rt(r_key[part_idx], t_key[part_idx])
                rt_key[part_idx] = rt_key[part_idx] @ k_fix
                

        key_inner_id = ((inner_index - 1) // 5) * 5 if flag == 0 else 0  
        key_dis = inner_index - key_inner_id
        # print('key_dis: ',key_dis)
        
        clouds = [_ for _ in range(num_parts)]
        gt_part_r = gt_part_r[0].numpy()
        gt_part_t = gt_part_t[0].numpy()
        gt_part_rt = [_ for _ in range(num_parts)]

        gt_joint_state -= key_j_state
        for part_idx in range(num_parts):
            gt_part_rt[part_idx] = compose_rt(gt_part_r[part_idx], gt_part_t[part_idx])
          
            gt_part_rt[part_idx] = np.linalg.inv(rt_key[part_idx]) @ gt_part_rt[part_idx]  
            gt_part_rt[part_idx] = base_fix_rt @ gt_part_rt[part_idx]  # base_fix_rt是随机生成的噪声矩阵
            gt_part_r[part_idx] = gt_part_rt[part_idx][:3, :3]  
            gt_part_t[part_idx] = gt_part_rt[part_idx][:3, 3]

           
            
            if part_idx not in [0, sort_part]:
                rt_key[part_idx] = rt_key[0]
            cloud_pcd = o3d.geometry.PointCloud()
            cloud_pcd.points = o3d.utility.Vector3dVector(camera_part[part_idx])
            cloud_pcd.transform(np.linalg.inv(rt_key[part_idx]))   
            cloud_pcd.transform(base_fix_rt) # 
            clouds[part_idx] = np.asarray(cloud_pcd.points, dtype=np.float32)  
        clouds = np.array(clouds)    
        cloud_pseu = np.concatenate(clouds, axis=0)  #

       
        pred_rt_list = [_ for _ in range(num_parts)]
        optim_pred_rt_list = [_ for _ in range(num_parts)]
       
        bcelogits = torch.nn.BCEWithLogitsLoss()  
        RTs = np.eye(4, dtype=np.float32)
        scales = np.ones((3,), dtype=np.float32)
        camera_cloud = (camera_cloud).cuda()

        pc_normal = estimate_normals(cloud_pseu, cfg.knn).astype(np.float32)
        pc_normals = torch.from_numpy(pc_normal[None]).cuda()
        pcs = torch.from_numpy(cloud_pseu[None]).cuda()

        point_idxs = np.random.randint(0, cloud_pseu.shape[0], (10000, 2))
            
        with torch.no_grad():
            dist = torch.cdist(pcs, pcs)
            sprin_feat_out = point_encoder(pcs, pc_normals, dist, camera_cloud)
            sprin_feat = sprin_feat_out[0]  # [1, N, 32]
            scales = sprin_feat_out[1].cpu().numpy()  # [1, 3]
            state = sprin_feat_out[2].squeeze().cpu().numpy()  # [1, 1]
            preds = ppf_encoder(pcs, pc_normals, sprin_feat, idxs=point_idxs)
            preds_tr = preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins)
    
        preds_tr = torch.softmax(preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins), -1)
        preds_tr = torch.cat([torch.multinomial(preds_tr[:, 0], 1), torch.multinomial(preds_tr[:, 1], 1)], -1).float()[None]
        preds_tr[0, :, 0] = preds_tr[0, :, 0] / (cfg.tr_num_bins - 1) * 2 * cfg.vote_range[0] - cfg.vote_range[0]
        preds_tr[0, :, 1] = preds_tr[0, :, 1] / (cfg.tr_num_bins - 1) * cfg.vote_range[1]
            
        # vote for center
        with cp.cuda.Device(cp_device):
            block_size = (cloud_pseu.shape[0] ** 2 + 512 - 1) // 512
            # extended_c = np.array([0,0,0.5])
            corners = np.stack([np.min(cloud_pseu, 0), np.max(cloud_pseu, 0)])
            grid_res = ((corners[1] - corners[0]) / cfg.res).astype(np.int32) + 1   # 投票的范围被限定在了grid里面，也即bbox中
            grid_res = np.clip(grid_res, a_min= 0, a_max=50)
            grid_obj = cp.asarray(np.zeros(grid_res, dtype=np.float32))
            ppf_kernel(
                (block_size, 1, 1),
                (512, 1, 1),
                (
                    cp.asarray(cloud_pseu).astype(cp.float32), cp.asarray(preds_tr[0].cpu().numpy()).astype(cp.float32), cp.asarray(np.ones((cloud_pseu.shape[0],))).astype(cp.float32),
                    cp.asarray(point_idxs).astype(cp.int32), grid_obj, cp.asarray(corners[0]), cp.float32(cfg.res), 
                    point_idxs.shape[0], num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2], True if args.adaptive_voting else False
                )
            )
            
            grid_obj = grid_obj.get()
            cand = np.array(np.unravel_index([np.argmax(grid_obj, axis=None)], grid_obj.shape)).T[::-1]
            cand_world = corners[0] + cand * cfg.res
            
        T_est = cand_world[-1]
        # print('平移量 T_est 为：[{:.3f},{:.3f},{:.3f}]'.format(T_est[0],T_est[1],T_est[2]))
        # np.savetxt('pc_out.txt',cloud_pseu)

        corners = np.stack([np.min(cloud_pseu, 0), np.max(cloud_pseu, 0)])
        RTs[:3, -1] = T_est
        
        # back vote filtering
        block_size = (point_idxs.shape[0] + n_threads - 1) // n_threads

        pred_center = T_est
        with cp.cuda.Device(cp_device):
            output_ocs = cp.zeros((point_idxs.shape[0], 3), cp.float32)
            backvote_kernel(
                (block_size, 1, 1),
                (n_threads, 1, 1),
                (
                    cp.asarray(cloud_pseu), cp.asarray(preds_tr[0].cpu().numpy()), output_ocs, cp.asarray(point_idxs).astype(cp.int32), cp.asarray(corners[0]), cp.float32(cfg.res), 
                    point_idxs.shape[0], num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2], cp.asarray(pred_center).astype(cp.float32), cp.float32(3 * cfg.res)
                )
            )
        oc = output_ocs.get()
        mask = np.any(oc != 0, -1)
        point_idxs = point_idxs[mask]  # 候选数量太少了
            
        with torch.cuda.device(0):
            with torch.no_grad():
                # sprin_feat = point_encoder.forward_nbrs(cloud_pseu[None], pc_normal[None], torch.from_numpy(knn_idxs).cuda()[None])[0]
                preds = ppf_encoder(pcs, pc_normals, sprin_feat, idxs=point_idxs)
                
                preds_tr = preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins)
                preds_up = preds[..., 2 * cfg.tr_num_bins:2 * cfg.tr_num_bins + cfg.rot_num_bins]
                preds_right = preds[..., 2 * cfg.tr_num_bins + cfg.rot_num_bins:2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins]
                preds_up_aux = preds[..., -5]
                preds_right_aux = preds[..., -4]
            
                
                preds_tr = torch.softmax(preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins), -1)
                preds_tr = torch.cat([torch.multinomial(preds_tr[:, 0], 1), torch.multinomial(preds_tr[:, 1], 1)], -1).float()[None]
                preds_tr[0, :, 0] = preds_tr[0, :, 0] / (cfg.tr_num_bins - 1) * 2 * cfg.vote_range[0] - cfg.vote_range[0]
                preds_tr[0, :, 1] = preds_tr[0, :, 1] / (cfg.tr_num_bins - 1) * cfg.vote_range[1]
                
                preds_up = torch.softmax(preds_up[0], -1)
                preds_up = torch.multinomial(preds_up, 1).float()[None]
                preds_up[0] = preds_up[0] / (cfg.rot_num_bins - 1) * np.pi
                
                preds_right = torch.softmax(preds_right[0], -1)
                preds_right = torch.multinomial(preds_right, 1).float()[None]
                preds_right[0] = preds_right[0] / (cfg.rot_num_bins - 1) * np.pi

        final_directions = []
        for j, (direction, aux) in enumerate(zip([preds_up, preds_right], [preds_up_aux, preds_right_aux])):
            if j == 1 and not cfg.regress_right:
                continue
                
            # vote for orientation
            with cp.cuda.Device(cp_device):
                candidates = cp.zeros((point_idxs.shape[0], num_rots, 3), cp.float32)

                block_size = (point_idxs.shape[0] + 512 - 1) // 512
                rot_voting_kernel(
                    (block_size, 1, 1),
                    (512, 1, 1),
                    (
                        cp.asarray(cloud_pseu), cp.asarray(preds_tr[0].cpu().numpy()), cp.asarray(direction[0].cpu().numpy()), candidates, cp.asarray(point_idxs).astype(cp.int32), cp.asarray(corners[0]).astype(cp.float32), cp.float32(cfg.res), 
                        point_idxs.shape[0], num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2]
                    )
                )
                sph_cp = torch.tensor(sphere_pts.T, dtype=torch.float32).cuda()
                start = np.arange(0, point_idxs.shape[0] * num_rots, num_rots)
                np.random.shuffle(start)
                sub_sample_idx = (start[:10000, None] + np.arange(num_rots)[None]).reshape(-1)
                candidates = torch.as_tensor(candidates, device='cuda').reshape(-1, 3)
                candidates = candidates[torch.LongTensor(sub_sample_idx).cuda()]
                cos = candidates.mm(sph_cp)
                counts = torch.sum(cos > np.cos(angle_tol / 180 * np.pi), 0).cpu().numpy()
            best_dir = np.array(sphere_pts[np.argmax(counts)])
            
            # filter up
            ab = cloud_pseu[point_idxs[:, 0]] - cloud_pseu[point_idxs[:, 1]]
            distsq = np.sum(ab ** 2, -1)
            ab_normed = ab / (np.sqrt(distsq) + 1e-7)[..., None]
            
            pairwise_normals = pc_normal[point_idxs[:, 0]]
            pairwise_normals[np.sum(pairwise_normals * ab_normed, -1) < 0] *= -1
                
            with torch.no_grad():
                target = torch.from_numpy((np.sum(pairwise_normals * best_dir, -1) > 0).astype(np.float32)).cuda()
                up_loss = bcelogits(aux[0], target).item()
                down_loss = bcelogits(aux[0], 1. - target).item()
                
            if down_loss < up_loss:
                final_dir = -best_dir
            else:
                final_dir = best_dir
            final_directions.append(final_dir)
            
        up = final_directions[0]
        if up[1] < 0:
            up = -up
            
        if cfg.regress_forward:
            forward = final_directions[1]
            if forward[2] < 0:
                forward = -forward
            forward -= np.dot(up, forward) * up
            forward /= (np.linalg.norm(forward) + 1e-9)
        else:
            forward = np.array([-up[1], up[0], 0.])
            forward /= (np.linalg.norm(forward) + 1e-9)
            
        if np.linalg.norm(forward) < 1e-7: # right is zero
            # input('right real?')
            forward = np.random.randn(3)
            forward -= forward.dot(up) * up
            forward /= np.linalg.norm(forward)

        R_est = np.stack([np.cross(up, forward), up, forward], -1)
    
        RTs[:3, :3] = R_est 
        
        pred_rt_base, Pred_s, pred_state = RTs, scales, state


        # pred_rt_base, Pred_s, pred_state = tracking_func(cloud_pseu, point_encoder, ppf_encoder,para_ppf, 20000, camera_cloud)
        d_pred_state = pred_state - key_j_state          
        gt_size = gt_size + mean_shape_part
        Pred_s = Pred_s + mean_shape_part
        iou_3d = compute_iou_3d(gt_size,Pred_s,num_parts) #
        

        relative = RotateAnyAxis_np(np.deg2rad(pred_state*10))

        pred_r_list = np.array([pred_rt_base[:3,:3], relative[:3,:3]])
        pred_t_list = np.array([pred_rt_base[:3,3],  relative[:3,3]])

        part_weight = [4, 6]
        
                        
        init_base_r = torch.from_numpy(pred_r_list)
        init_base_t = torch.from_numpy(pred_t_list)
        n_parts = len(pred_r_list)
        pose_estimator = pose_optimizer.PoseEstimator(num_parts=n_parts, init_r=init_base_r, init_t=init_base_t,
                                            device=device)
        
        xyz_camera = torch.from_numpy(copy.deepcopy(clouds)).cuda()
        cad = torch.from_numpy(copy.deepcopy(canos_part)).cuda()   # canonical_pts
    
        
        loss, optim_transforms = pose_optimizer.optimize_pose(pose_estimator, xyz_camera, cad, part_weight)
        # print("loss:{}".format(loss.item()))
        optim_transforms = optim_transforms.cpu().numpy()
        optim_pred_r_list = [optim_transforms[0][:3,:3], optim_transforms[1][:3,:3]]
        optim_pred_t_list = [optim_transforms[0][:3,3], optim_transforms[1][:3,3]]
        optim_pred_rt_list = optim_transforms         
        optim_errs = calErr(optim_pred_r_list, optim_pred_t_list, gt_part_r, gt_part_t, sort_part)   # gt_part_t [2,1,3]  gt_part_r [2,3,3]
            
        optim_base_r_err = optim_errs[0][0]
        optim_child_r_err = optim_errs[0][1]
        optim_base_t_err = optim_errs[1][0]   
        optim_child_t_err = optim_errs[1][1]

                
        optim_base_r_error_all += optim_base_r_err
        optim_sort_child_r_error_all += optim_child_r_err
        optim_base_t_error_all += optim_base_t_err
        optim_sort_child_t_error_all += optim_child_t_err
        
        print(f'optim base r_err: {optim_base_r_err}')
        print(f'optim child r_err: {optim_child_r_err}')
        

        print(f'optim base t_err: {optim_base_t_err}')
        print(f'optim child t_err: {optim_child_t_err}')
     
        print(f'part0 3D IOU: {iou_3d[0]} ;  part1 3D IOU: {iou_3d[1]}')
        print()


#######################  设计优化算法 END  ############################################################
        # 关键帧很重要，最好用些优化算法优化关键帧，不然投票法波动太大很容易导致关键帧有问题进而引发连锁反应
        update_num += 1  
        if update_num == 5:
            rt_key[part_idx] =  rt_key[part_idx] @ optim_pred_rt_list[part_idx]  # 这里的rt_key是伪标准空间下的变换矩阵
            r_key[part_idx] = rt_key[part_idx][:3, :3]
            t_key[part_idx] = rt_key[part_idx][:3, 3] 
            update_num = 0
            key_j_state = ori_gt_state


    print(f'turn:{turns}')

    print(f"optim base r error mean:{optim_base_r_error_all/turns}")
    print(f"optim child r error mean:{optim_sort_child_r_error_all/turns}")
    print(f"optim base t error mean:{optim_base_t_error_all/turns}")
    print(f"optim child t error mean:{optim_sort_child_t_error_all/turns}")
    print()
    time2 = time.time()
    print((time2-time1)/len(test_dataloader))