import numpy as np
from easydict import EasyDict
import yaml
import torch
import argparse
import os

from scipy.spatial.transform import Rotation
from models.VecUnico import VecUnico


# import models.builder as builder
from utils.config import cfg_from_yaml_file
from datasets.io import IO
from datasets.data_transforms import Compose

DEFAULT_CFG = "./cfgs/BuildingNL_models/VecUnico_multi.yaml"
DEFAULT_CKPT = "./ckpt/checkpoint.pth"
DEFAULT_DATA_ROOT = "./evaluation/pc"
DEFAULT_OUTPUT_DIR = "./evaluation/vg"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DATA_MODE = "npy"        # ply | npy
DEFAULT_OUTPUT_FORMAT = "vg"    # seg | vg | xyz
DEFAULT_PARAM_MODE = "points"    # points | analytic
DEFAULT_THRESHOLD = 0.5


# Quadric normal helpers
def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.clip(n, eps, None)
    return v / n

def quadric_normals_at_points(points_xyz, ids, coeffs_10):
    N = points_xyz.shape[0]
    normals = np.zeros((N, 3), dtype=np.float64)
    if coeffs_10 is None or len(coeffs_10) == 0:
        return normals.astype(np.float32)

    ids = ids.astype(int)
    for gid in np.unique(ids):
        if gid < 0 or gid >= coeffs_10.shape[0]:
            continue
        idx = np.where(ids == gid)[0]
        if idx.size == 0:
            continue
        x  = points_xyz[idx]
        th = coeffs_10[gid]
        normals[idx] = _normalize(th[7:10])  # Use plane normal approximation
    return normals.astype(np.float32)


def load_config_with_fallback():
    try:
        from utils import parser
        from utils.config import get_config
        args = parser.get_args()
        config = get_config(args, logger=None)
        return config
    except Exception:
        return cfg_from_yaml_file(DEFAULT_CFG)
    
def build_model_from_config(config, ckpt_path=DEFAULT_CKPT, device=DEFAULT_DEVICE):
    checkpoint = torch.load(ckpt_path)
    base_model = VecUnico(config.model)
    base_model.load_state_dict(checkpoint["base_model"])
    base_model.to(device)
    base_model.eval()
    return base_model


def inference_single(model, pc_file,data_root=DEFAULT_DATA_ROOT,out_dir=DEFAULT_OUTPUT_DIR,device=DEFAULT_DEVICE,  output_format=DEFAULT_OUTPUT_FORMAT, data_mode=DEFAULT_DATA_MODE, aug_mode = "no_agu"):

    os.makedirs(out_dir, exist_ok=True)
    
    if data_mode not in ("npy", "ply"):
        raise ValueError(f"Unsupported data_mode: {data_mode}")

    transform = Compose([
        {'callback': 'RandomSamplePoints', 'parameters': {'n_points': 2048}, 'objects': ['input']}
    ])
    
    # import pdb; pdb.set_trace()

    # set random seed for reproducibility
    pc =  IO.get(pc_file).astype(np.float32)
    pc_out = pc_file

    np.random.seed(42)
    if aug_mode == "rotation":
        pc_out = pc_file.replace('/pc', '/pc_rotation')
        os.makedirs(os.path.dirname(pc_out), exist_ok=True)
        axis = np.array([0, 0, 1])     # z 轴
        theta = np.random.uniform(0, 2*np.pi)

        Rz = Rotation.from_rotvec(axis * theta).as_matrix()
        pc = np.matmul(pc, Rz)
        np.save(pc_out, pc)
    elif aug_mode == "translation":
        pc_out = pc_file.replace('/pc', '/pc_translation')
        os.makedirs(os.path.dirname(pc_out), exist_ok=True)
        pc += np.random.uniform(-0.5, 0.5, size=(1, 3))
        np.save(pc_out, pc)
    elif aug_mode == "scale":
        pc_out = pc_file.replace('/pc', '/pc_scale')
        os.makedirs(os.path.dirname(pc_out), exist_ok=True)    
        scale = np.random.uniform(0.5, 2.0)*0.5
        pc *= scale 
        np.save(pc_out, pc)
    elif aug_mode == "sim3":
        pc_out = pc_file.replace('/pc', '/pc_sim3')
        os.makedirs(os.path.dirname(pc_out), exist_ok=True)
        scale = np.random.uniform(0.5, 2.0)
        translation = np.random.uniform(-0.5, 0.5, size=(1, 3))
        axis = np.array([0, 0, 1])     # z 轴
        theta = np.random.uniform(0, 2*np.pi)
        Rz = Rotation.from_rotvec(axis * theta).as_matrix()
        pc = np.matmul(pc, Rz) * scale + translation
        np.save(pc_out, pc)
         

    ret = model(torch.from_numpy(pc).float().unsqueeze(0).to(device), epoch=600)
    rebuild_points, pred_masks, _raw_cls = ret['rebuild_points'], ret['pred_masks'].sigmoid(), ret['class_prob'].softmax(dim=-1)  

    class_prob = (1.0 - _raw_cls[..., -1]).unsqueeze(-1) # [B, Q, 1]
    m = pred_masks > 0.5  # b, m, 1
    c_m = (pred_masks * m).sum(-1) / torch.clamp(m.sum(-1), min=1)  # b, m, 1
    keep = torch.bitwise_and(m.sum(-1) > 1, (c_m * class_prob.squeeze(-1)) > DEFAULT_THRESHOLD)  
    

    heatmap = class_prob * keep.float().unsqueeze(-1) * pred_masks
    # heatmap: [B, Q, N]
    assigned_query = heatmap.argmax(dim=1)       # [B, N]
    assigned_score = heatmap.max(dim=1).values   # [B, N]
    assigned_query[assigned_score < 0.5] = -1
    
    B, N = assigned_query.shape
    for b in range(B):
        query_ids = assigned_query[b].unique()
        query_ids = query_ids[query_ids >= 0]  # only consider assigned queries
        probs = ret['class_prob'][b].softmax(dim=-1)  # (Q,5)
        type_ids = probs.argmax(dim=-1)               # (Q,)

        batch_instances = []
        accepted_qids = []  # original query indices that produced a segment
        for qid in query_ids:
            qid_int = int(qid.item())
            tid = int(type_ids[qid_int].item())
            if tid == 1:
                # skip background class
                continue

            point_mask = (assigned_query[b] == qid)  # [N]
            if point_mask.sum() >= 1:
                current_points = (
                    rebuild_points[b]
                    .reshape(512, -1, 3)[point_mask, :, :]
                    .reshape(-1, 3)
                    .detach()
                    .cpu()
                    .numpy()
                )
                batch_instances.append(
                    np.concatenate(
                        (current_points, np.ones((current_points.shape[0], 1)) * len(accepted_qids)),
                        axis=1
                    )
                )
                accepted_qids.append(qid_int)

        batch_instances = (
            np.concatenate(batch_instances, axis=0)
            if len(batch_instances) > 0 else np.zeros((0, 4), dtype=np.float32)
        )

        if output_format == "vg":
            if batch_instances.shape[0] == 0:
                print(f"[WARN] {id}: no valid primitives after filtering; skipping export.")
                continue

            xyz_lab = batch_instances  # (K,4) with xyz + local gid
            pts_xyz = xyz_lab[:, :3].astype(np.float32)
            local_ids = xyz_lab[:, 3].astype(int)

            quadric_params = ret.get('quadrics', None)
            group_info = []
            coeffs_list = []
            if quadric_params is not None and len(accepted_qids) > 0:
                new_gid = 0
                for orig_qid in accepted_qids:
                    if orig_qid >= type_ids.shape[0]:
                        continue
                    tid = int(type_ids[orig_qid].item())

                    params10 = quadric_params[b, orig_qid].detach().cpu().numpy()[:3]
                    params10[:3] = params10[:3]/2
                    params10 = np.concatenate([np.zeros(7, dtype=np.float32), params10], axis=0)
                    params10 = params10 / np.linalg.norm(params10)  # normalize
                    group_info.append({
                        "id": new_gid,
                        "type": tid,
                        "parameters": params10
                    })
                    coeffs_list.append(params10)
                    new_gid += 1

            coeffs_10 = np.asarray(coeffs_list, dtype=np.float32) if coeffs_list else None
            normals = quadric_normals_at_points(pts_xyz, local_ids, coeffs_10)

            
            from utils.save_vg import save_vg
            out_path = pc_out.replace('/pc', '/vg').replace('.npy', '.vg').replace('.ply', '.vg')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_vg(
                xyz_lab,
                out_path,
                group_info=group_info if group_info else None,
                normals=normals
            )
        else:
            out_path = os.path.join(out_dir, f"{os.path.basename(pc_out).split('.')[0]}.xyz")
            np.savetxt(out_path, batch_instances)
            
def main():
    data_root = os.getenv("NEURALPRIMITIVE_DATA_ROOT", DEFAULT_DATA_ROOT)
    out_dir = os.getenv("NEURALPRIMITIVE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    device = DEFAULT_DEVICE
    data_mode = os.getenv("NEURALPRIMITIVE_DATA_MODE", DEFAULT_DATA_MODE)
    output_format = os.getenv("NEURALPRIMITIVE_OUTPUT_FORMAT", DEFAULT_OUTPUT_FORMAT).lower()
    config = load_config_with_fallback()
    base_model = build_model_from_config(config, ckpt_path=DEFAULT_CKPT, device=device)
    aug_mode = os.getenv("NEURALPRIMITIVE_AUG_MODE", "no_agu")
    test_id = os.getenv("NEURALPRIMITIVE_TEST_ID", "building_01")
    pc_file = os.path.join(data_root, f"{test_id}." + data_mode)

    inference_single(base_model, 
                     pc_file, 
                     data_root=data_root,
                     out_dir=out_dir,
                     device=device,  
                     output_format=output_format, 
                     data_mode=data_mode, 
                     aug_mode = aug_mode)
    
if __name__ == '__main__':
    main()