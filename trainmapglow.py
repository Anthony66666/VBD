#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
from tqdm import tqdm
import numpy as np
import math

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from pathlib import Path
from typing import Optional
from collections import defaultdict

# ==== trajdata 相关 ====
from trajdata import UnifiedDataset
from trajdata import SceneBatch
from trajdata import MapAPI
from trajdata.caching.scene_cache import SceneCache
from trajdata.maps.vec_map_elements import MapElementType
from trajdata.data_structures import state as traj_state

# 修复 multiprocessing pickle 报错: AttributeError: Can't get attribute 'StateArrayXYXdYdXddYddH'
# 显式注册该动态类，确保子进程能找到定义
traj_state.NP_STATE_TYPES["x,y,xd,yd,xdd,ydd,h"]
traj_state.TORCH_STATE_TYPES["x,y,xd,yd,xdd,ydd,h"]
traj_state.NP_STATE_TYPES["x,y,xd,yd,xdd,ydd,s,c"]
traj_state.TORCH_STATE_TYPES["x,y,xd,yd,xdd,ydd,s,c"]

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # 降低日志等级

# === 保留你原有的路径设置/Model 引用方式 ===
# sys.path.append(rf"C:\SceneGlow\MapGlow")
from MapGLow import Glow


# ----------------------------
# 你原来的 CustomDataset（略微整理）
# ----------------------------
class CustomDataset(Dataset):
    def __init__(self, history_data, target_data, labels, map_data, map_mask,
                 agent_types, scene_stats, map_name,
                 target_vehicle_mask, history_vehicle_mask):
        self.history_data = history_data
        self.target_data = target_data
        self.labels = labels
        self.map_data = map_data
        self.map_mask = map_mask
        self.agent_types = agent_types
        self.scene_stats = scene_stats
        self.map_name = map_name
        self.target_vehicle_mask = target_vehicle_mask
        self.history_vehicle_mask = history_vehicle_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.history_data[idx],
                self.target_data[idx],
                self.labels[idx],
                self.map_data[idx],
                self.map_mask[idx],
                self.agent_types[idx],
                self.scene_stats[idx],
                self.map_name[idx],
                self.target_vehicle_mask[idx],
                self.history_vehicle_mask[idx])


# ----------------------------
# 辅助函数：计算 z shapes（保留你的实现）
# ----------------------------
def calc_z_shapes(n_channel, input_size_h, input_size_w, n_block):
    """
    计算每个 block 输出的 z 形状。
    squeeze factor = 2: 每个 block 将时间维度减半，通道翻倍。
    """
    z_shapes = []
    for i in range(n_block - 1):
        input_size_h //= 2  # squeeze factor = 2
        # n_channel *= 2
        z_shapes.append((n_channel, input_size_h, input_size_w))

    input_size_h //= 2  # squeeze factor = 2
    z_shapes.append((n_channel * 2, input_size_h, input_size_w))  # no split, full channels = 2x
    return z_shapes


# ============================
# 统一的 DataLoader 构建函数
# ============================
def build_dataloader(args, is_distributed, local_rank, ngpus_per_node):
    """
    根据 args.use_trajdata 参数选择不同的数据加载方式。
    
    Returns:
        dataloader: DataLoader 对象
        map_api: MapAPI 对象（仅 trajdata 模式），否则为 None
        sampler: DistributedSampler（分布式训练时），否则为 None
    """
    if args.use_trajdata:
        # trajdata + Waymo 流式加载
        dataloader, map_api = build_trajdata_dataloader(args, is_distributed, local_rank)
        sampler = None  # trajdata 内部处理 shuffle
    else:
        # 传统 npz 数据集加载
        if args.newdataset:
            dataset = build_dataset_npz(args.path, args.map_path)
        else:
            dataset = build_dataset_npz_new(args.path, args.map_path)

        if is_distributed:
            print("Use DistributedSampler")
            sampler = DistributedSampler(dataset, num_replicas=ngpus_per_node,
                                         rank=local_rank, shuffle=True, drop_last=True)
            dataloader = DataLoader(dataset, batch_size=args.batch, sampler=sampler,
                                    num_workers=args.num_workers, pin_memory=True)
        else:
            sampler = None
            dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True)
        map_api = None

    return dataloader, map_api, sampler


def process_batch(batch_raw, args, map_api, device):
    """
    统一处理来自不同数据源的 batch，返回标准化的张量。
    
    Returns:
        dict: 包含所有需要的数据张量
    """
    if args.use_trajdata:
        # 确定数据集名称
        if args.td_split == "train":
            ds_name = "waymo_train"
        elif args.td_split == "val":
            ds_name = "waymo_val"
        elif args.td_split == "test":
            ds_name = "waymo_test"
        else:
            ds_name = None

        (history_data, target_data, labels,
         map_data, map_mask, agent_types, scene_stats,
         map_name, target_vehicle_mask, history_vehicle_mask, 
         timestep_mask, map_type, extra_data) = \
            scene_batch_to_mapglow_batch(
                batch_raw,
                map_api,
                num_lanes=args.num_lanes,  # trajdata 默认 256
                num_points=30,
                T_hist=11,
                T_fut=80,
                max_agents=args.img_size_w,
                dataset_name=ds_name,
                map_radius=args.map_radius
            )
        
        # 移动到 device
        history_data = history_data.to(device, non_blocking=True)
        target_data = target_data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        map_data = map_data.to(device, non_blocking=True)
        map_mask = map_mask.to(device, non_blocking=True)
        agent_types = agent_types.to(device, non_blocking=True)
        scene_stats = scene_stats.to(device, non_blocking=True)
        target_vehicle_mask = target_vehicle_mask.to(device, non_blocking=True)
        history_vehicle_mask = history_vehicle_mask.to(device, non_blocking=True)
        timestep_mask = timestep_mask.to(device, non_blocking=True)
        map_type = map_type.to(device, non_blocking=True)
        
    else:
        # npz 数据集
        history_data = batch_raw[0].to(device, non_blocking=True)
        target_data = batch_raw[1].to(device, non_blocking=True)
        labels = batch_raw[2].to(device, non_blocking=True)
        map_data_raw = batch_raw[3].to(device, non_blocking=True)
        map_mask = batch_raw[4].to(device, non_blocking=True)
        agent_types = batch_raw[5].to(device, non_blocking=True)
        scene_stats = batch_raw[6].to(device, non_blocking=True)
        map_name = batch_raw[7]
        target_vehicle_mask = batch_raw[8].to(device, non_blocking=True)
        history_vehicle_mask = batch_raw[9].to(device, non_blocking=True)
        
        # npz 的 map_data 只有 (x, y)，需要补充 yaw 和 type 以匹配模型输入
        # 模型期望 map_data: [B, L, P, 6] = [x, y, yaw, type0, type1, type2]
        # 其中 yaw 需要从相邻点计算，type 使用 one-hot (默认为 centerline = type 0)
        B_map, L, P, C_map = map_data_raw.shape
        
        if C_map == 2:
            # 只有 (x, y)，需要补充 yaw 和 type
            # 注意：npz 中的 map_data 已经归一化过了，不需要再除以 100
            xy = map_data_raw  # [B, L, P, 2]，已归一化
            
            # 计算 yaw：从相邻点的方向（注意 xy 已归一化，但方向计算不受影响）
            yaw = torch.zeros(B_map, L, P, 1, device=device, dtype=xy.dtype)
            if P > 1:
                diff = xy[:, :, 1:, :] - xy[:, :, :-1, :]  # [B, L, P-1, 2]
                yaw_vals = torch.atan2(diff[..., 1], diff[..., 0])  # [B, L, P-1]
                yaw[:, :, :-1, 0] = yaw_vals
                yaw[:, :, -1, 0] = yaw[:, :, -2, 0]  # 最后一点复制倒数第二点的 yaw
            
            # yaw 归一化到 [-1, 1]（除以 pi）
            yaw = yaw / torch.pi
            
            # 默认类型为 centerline (type=0)，one-hot: [1, 0, 0]
            type_onehot = torch.zeros(B_map, L, P, 3, device=device, dtype=xy.dtype)
            type_onehot[..., 0] = 1.0  # centerline
            
            # 对于 padding 位置（mask=False），将 type 置零
            type_onehot = type_onehot * map_mask.unsqueeze(-1).float()
            
            # 拼接：[x, y, yaw, type0, type1, type2]，xy 已归一化无需再处理
            map_data = torch.cat([xy, yaw, type_onehot], dim=-1)  # [B, L, P, 6]
        elif C_map == 4:
            # 已有 (x, y, yaw, type)，转换 type 为 one-hot
            xy = map_data_raw[..., :2] / 100.0
            yaw = map_data_raw[..., 2:3] / torch.pi
            map_types = map_data_raw[..., 3].long().clamp(min=0, max=2)
            type_onehot = torch.nn.functional.one_hot(map_types, num_classes=3).to(dtype=xy.dtype)
            map_data = torch.cat([xy, yaw, type_onehot], dim=-1)
        else:
            # 假设已是正确格式 (6维)
            map_data = map_data_raw
        
        # 非 trajdata 路径可能没有 map_type 和 timestep_mask
        map_type = torch.zeros(map_mask.shape[0], map_mask.shape[1], dtype=torch.long, device=device)
        timestep_mask = None  # npz 模式暂不支持精细时间步 mask
        extra_data = None
        
        # 修正 agent_types
        agent_types = torch.where(labels == 4, torch.tensor(2, device=agent_types.device), agent_types)
    
    return {
        'history_data': history_data,
        'target_data': target_data,
        'labels': labels,
        'map_data': map_data,
        'map_mask': map_mask,
        'agent_types': agent_types,
        'scene_stats': scene_stats,
        'map_name': map_name,
        'target_vehicle_mask': target_vehicle_mask,
        'history_vehicle_mask': history_vehicle_mask,
        'timestep_mask': timestep_mask,
        'map_type': map_type,
        'extra_data': extra_data if args.use_trajdata else None
    }


# ============================
# 你原来的 npz 构建 Dataset（原样保留）
# ============================
def build_dataset_npz(path, map_path):
    """传统 npz 数据集加载（旧格式）"""
    full_dataset = np.load(path, allow_pickle=True)
    full_map_data = np.load(map_path, allow_pickle=True)

    map_data = full_map_data['map_data']
    map_mask = full_map_data['map_mask']

    labels = full_dataset['labels']
    data = full_dataset["trajectories"]  # shape [N, C, T, V]

    # pad_pos: bool [N, T, V] True 表示该时间步为 padding
    pad_pos = np.isclose(data, 0.0, atol=0.0).all(axis=1)
    data = np.where(pad_pos[:, None, :, :], -1.0, data)

    history_data = data[:, :, :10, :]    # [N, C, T_hist, V]
    target_data = data                   # 保留整个时间轴

    agent_types = full_dataset['agent_types']
    map_name = full_dataset['map_names']
    scene_stats = full_dataset['scene_stats']

    combined_features = []
    for s in scene_stats:
        feature_vector = np.concatenate([s['mean'], np.array([s['scale']])])
        combined_features.append(feature_vector)
    features_array = np.array(combined_features, dtype=np.float32)

    target_vehicle_padded = pad_pos.all(axis=1)          # [N, V]
    history_vehicle_padded = pad_pos[:, :10, :].all(axis=1)
    target_vehicle_mask = ~target_vehicle_padded         # bool [N, V]
    history_vehicle_mask = ~history_vehicle_padded

    print("数据加载完成：")
    print(f"- 总样本数: {target_data.shape[0]}")
    print(f"- 平均每个场景的真实车辆数 (目标): {target_vehicle_mask.sum(axis=1).mean():.2f}")
    print(f"- 平均每个场景的真实车辆数 (历史): {history_vehicle_mask.sum(axis=1).mean():.2f}")

    tensor_scene_stats = torch.tensor(features_array)
    tensor_history_data = torch.tensor(history_data, dtype=torch.float32)
    tensor_target_data = torch.tensor(target_data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    tensor_map_data = torch.tensor(map_data, dtype=torch.float32)
    tensor_map_mask = torch.tensor(map_mask, dtype=torch.bool)
    tensor_agent_types = torch.tensor(agent_types, dtype=torch.long)
    tensor_target_vehicle_mask = torch.tensor(target_vehicle_mask, dtype=torch.bool)
    tensor_history_vehicle_mask = torch.tensor(history_vehicle_mask, dtype=torch.bool)

    dataset = CustomDataset(
        tensor_history_data, tensor_target_data, tensor_labels,
        tensor_map_data, tensor_map_mask, tensor_agent_types,
        tensor_scene_stats, map_name,
        tensor_target_vehicle_mask, tensor_history_vehicle_mask
    )
    return dataset


def build_dataset_npz_new(path, map_path):
    """传统 npz 数据集加载（新格式）"""
    full_dataset = np.load(path, allow_pickle=True)
    full_map_data = np.load(map_path, allow_pickle=True)  # 加载地图数据

    map_data = full_map_data['map_data']
    map_mask = full_map_data['map_mask']

    labels = full_dataset['trajectory_labels']
    data = full_dataset["data"]  # 原始数据
    data = data[:, 0:5, :, :]     # 只取前 5 个通道作为轨迹相关特征

    pad_pos = np.isclose(data, 0.0, atol=0.0).all(axis=1)  # [N, T, V]
    data = np.where(pad_pos[:, None, :, :], -1.0, data)

    history_data = data[:, :, :10, :]
    target_data = data

    agent_types = full_dataset['data'][:, 7, 0, :]

    map_name = full_dataset['map_names']
    scene_stats = full_dataset['scene_stats']

    combined_features = []
    for s in scene_stats:
        feature_vector = np.concatenate([s['mean'], np.array([s['scale']])])
        combined_features.append(feature_vector)
    features_array = np.array(combined_features, dtype=np.float32)

    target_vehicle_padded = pad_pos.all(axis=1)            # [N,V]
    history_vehicle_padded = pad_pos[:, :10, :].all(axis=1)
    target_vehicle_mask = ~target_vehicle_padded
    history_vehicle_mask = ~history_vehicle_padded

    tensor_scene_stats = torch.tensor(features_array)
    tensor_history_data = torch.tensor(history_data, dtype=torch.float32)
    tensor_target_data = torch.tensor(target_data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    tensor_map_data = torch.tensor(map_data, dtype=torch.float32)
    tensor_map_mask = torch.tensor(map_mask, dtype=torch.bool)
    tensor_agent_types = torch.tensor(agent_types, dtype=torch.long)

    tensor_target_vehicle_mask = torch.tensor(target_vehicle_mask, dtype=torch.bool)
    tensor_history_vehicle_mask = torch.tensor(history_vehicle_mask, dtype=torch.bool)

    dataset = CustomDataset(
        tensor_history_data, tensor_target_data, tensor_labels,
        tensor_map_data, tensor_map_mask, tensor_agent_types,
        tensor_scene_stats, map_name,
        tensor_target_vehicle_mask, tensor_history_vehicle_mask
    )
    return dataset


# ----------------------------
# trajdata: 矢量地图编码为 [64,20,2] + mask
# ----------------------------
def encode_vec_map_to_grid(gt_map,
                           ego_pos,
                           ego_heading,
                           num_lanes=64,
                           num_points=20):
    """
    gt_map: dict, keys = ['centerlines', 'left_edges', 'right_edges', 'crosswalks']
            每个 value 是 list of (Ni, 3) numpy array [x, y, yaw]
    ego_pos: (2,) 世界坐标
    ego_heading: 标量，弧度
    返回:
      map_data: [num_lanes, num_points, 4] -> [x, y, yaw, type]
      map_mask: [num_lanes, num_points] bool
      map_type: [num_lanes] int (0: centerline, 1: boundary, 2: crosswalk)
    """
    polylines = []
    types = []  # 0: centerline, 1: boundary, 2: crosswalk

    # 1. Centerlines
    for line in gt_map.get("centerlines", []):
        if line is None: continue
        line = np.asarray(line)
        if line.shape[0] < 2: continue
        # line should be [N, 3] (x, y, yaw)
        polylines.append(line[:, :3].astype(np.float32))
        types.append(0)

    # 2. Boundaries
    for key in ["left_edges", "right_edges"]:
        for line in gt_map.get(key, []):
            if line is None: continue
            line = np.asarray(line)
            if line.shape[0] < 2: continue
            polylines.append(line[:, :3].astype(np.float32))
            types.append(1)

    # 3. Crosswalks
    for line in gt_map.get("crosswalks", []):
        if line is None: continue
        line = np.asarray(line)
        if line.shape[0] < 2: continue
        polylines.append(line[:, :3].astype(np.float32))
        types.append(2)

    if len(polylines) == 0:
        return (np.zeros((num_lanes, num_points, 4), dtype=np.float32),
                np.zeros((num_lanes, num_points), dtype=bool),
                np.zeros((num_lanes,), dtype=np.int64))

    # 转到 ego 坐标（以 ego 为原点，heading 为 x 轴）
    x0, y0 = ego_pos[0], ego_pos[1]
    c, s = np.cos(-ego_heading), np.sin(-ego_heading)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    new_polys = []
    for pts in polylines:
        # pts: [N, 3] -> x, y, yaw
        xy = pts[:, :2]
        yaw = pts[:, 2]
        
        # Transform position
        xy_ego = (xy - np.array([x0, y0], dtype=np.float32)) @ R.T
        
        # Transform yaw
        yaw_ego = yaw - ego_heading
        # Wrap to [-pi, pi]
        yaw_ego = np.mod(yaw_ego + np.pi, 2 * np.pi) - np.pi
        
        new_pts = np.column_stack([xy_ego, yaw_ego])
        new_polys.append(new_pts)
        
    polylines = new_polys

    # 按距离排序（离 ego 最近的优先）
    def line_min_dist(pts):
        return np.linalg.norm(pts[:, :2], axis=1).min()

    combined = list(zip(polylines, types))
    combined.sort(key=lambda x: line_min_dist(x[0]))
    
    polylines = [x[0] for x in combined]
    types = [x[1] for x in combined]

    map_data = np.zeros((num_lanes, num_points, 4), dtype=np.float32)
    map_mask = np.zeros((num_lanes, num_points), dtype=bool)
    map_type = np.zeros((num_lanes,), dtype=np.int64)

    for lane_idx in range(min(num_lanes, len(polylines))):
        pts = polylines[lane_idx] # [N, 3] (x, y, yaw)
        t = types[lane_idx]
        
        n = min(pts.shape[0], num_points)
        
        # [x, y, yaw, type]
        map_data[lane_idx, :n, :3] = pts[:n]
        map_data[lane_idx, :n, 3] = float(t)
        
        map_mask[lane_idx, :n] = True
        map_type[lane_idx] = t

    return map_data, map_mask, map_type


def return_inf():
    return np.inf

# ----------------------------
# trajdata: Waymo DataLoader（流式）
# ----------------------------
def build_trajdata_dataloader(args, is_distributed, local_rank):
    """
    构建基于 trajdata + Waymo 的 DataLoader（流式），并初始化 MapAPI。
    """
    # 1) 选择 Waymo 的 dataset id
    if args.td_split == "train":
        waymo_id = "waymo_train"
    elif args.td_split == "val":
        waymo_id = "waymo_val"
    elif args.td_split == "test":
        waymo_id = "waymo_test"
    else:
        raise ValueError(f"Unknown td_split: {args.td_split}")

    dataset = UnifiedDataset(
        desired_data=[waymo_id],
        centric="scene",
        desired_dt=0.1,          # Waymo 10Hz -> 0.1s
        incl_robot_future=True,
        incl_raster_map=False,   # 我们只要 vec map
        incl_vector_map=True,
        standardize_data=False,
        num_workers=args.num_workers,
        history_sec=(1.0, 1.0),                # 1s 历史（10帧）
        future_sec=(8.0, 8.0),  
        verbose=(local_rank == 0),
        cache_location="~/.unified_data_cache",
        agent_interaction_distances=defaultdict(return_inf),
        data_dirs={
            waymo_id: args.td_dataroot,
        },
    )

    if is_distributed:
        shuffle = True
    else:
        shuffle = True

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=dataset.get_collate_fn(),  # 返回 SceneBatch
        pin_memory=True,
    )

    cache_path = Path("~/.unified_data_cache").expanduser()
    map_api = MapAPI(cache_path)

    return dataloader, map_api


# ----------------------------
# trajdata: SceneBatch -> Glow Batch
# ----------------------------
def scene_batch_to_mapglow_batch(batch: SceneBatch,
                                 map_api: MapAPI,
                                 num_lanes=64,
                                 num_points=20,
                                 T_hist=11,
                                 T_fut=80,
                                 max_agents=32,
                                 dataset_name=None,
                                 map_radius=150.0):
    """
    把 trajdata 的 SceneBatch 转成 Glow 训练用的元组。
    
    返回的 mask 设计：
    - target_vehicle_mask: [B, V] bool, True = 该 agent 有有效数据（至少有一个有效时间步）
    - history_vehicle_mask: [B, V] bool, True = 该 agent 在历史阶段有有效数据
    - timestep_mask: [B, T, V] bool, True = 该时间步有有效数据
    
    Padding 策略：
    - 无效位置的数据值设为 -1.0，便于检查和调试
    - 通过 mask 来区分有效/无效
    """
    B = batch.agent_hist.shape[0]
    device = batch.agent_hist.device

    history_data_list = []
    target_data_list = []
    labels_list = []
    map_data_list = []
    map_mask_list = []
    map_type_list = []
    agent_types_list = []
    scene_stats_list = []
    map_name_list = []
    target_vehicle_mask_list = []
    history_vehicle_mask_list = []
    timestep_mask_list = []  # 新增：精细的时间步 mask
    extra_data_list = []

    T_total = T_hist + T_fut

    for b in range(B):
        # -------- 1) 基本信息 --------
        agent_hist = batch.agent_hist[b]  # [Na, Th_full, D]
        agent_fut = batch.agent_fut[b]    # [Na, Tf_full, 2]
        agent_type = batch.agent_type[b]  # [Na]
        ego_state = batch.centered_agent_state
        ego_pos = ego_state.position[b]       # [1,2]
        ego_heading = ego_state.heading[b]    # [1]

        # 1. 直接使用 scene_id (Waymo 等数据集地图与场景绑定)
        if hasattr(batch, 'scene_ids'):
            map_name = batch.scene_ids[b]
            if dataset_name and "waymo" in dataset_name and map_name.startswith("scene_"):
                map_name = map_name.replace("scene_", f"{dataset_name}_")
        else:
            map_name = "unknown_map"

        # 2. 修正格式为 "env:map"
        if map_name != "unknown_map" and ":" not in map_name:
            if dataset_name is not None:
                 map_name = f"{dataset_name}:{map_name}"
            elif hasattr(batch, "data_source"):
                 map_name = f"{batch.data_source[b]}:{map_name}"

        Th_full = agent_hist.shape[1]
        Tf_full = agent_fut.shape[1]
        
        # 允许数据长度不足，进行切片
        Th_actual = min(Th_full, T_hist)
        Tf_actual = min(Tf_full, T_fut)

        hist_feat = agent_hist[:, -Th_actual:, :]    # [Na, Th_actual, D]
        fut_pos = agent_fut[:, :Tf_actual, :2]       # [Na, Tf_actual, 2]

        full_pos = torch.cat([hist_feat[:, :, 0:2], fut_pos], dim=1)  # [Na, Th_actual + Tf_actual, 2]

        # --- Coordinate Transformation: Global -> Ego ---
        full_pos = full_pos - ego_pos.view(1, 1, 2)
        c = torch.cos(-ego_heading)
        s = torch.sin(-ego_heading)
        x = full_pos[..., 0]
        y = full_pos[..., 1]
        new_x = c * x - s * y
        new_y = s * x + c * y
        full_pos = torch.stack([new_x, new_y], dim=-1)

        dt = batch.dt[b].item() if hasattr(batch, "dt") else 0.1
        if dt < 1e-4: dt = 0.1
        vel = torch.zeros_like(full_pos)
        vel[:, 1:, :] = (full_pos[:, 1:, :] - full_pos[:, :-1, :]) / dt
        vel[:, 0, :] = vel[:, 1, :]

        vx = vel[:, :, 0:1]
        vy = vel[:, :, 1:2]

        # -------- Extra Features: Heading, Length, Width --------
        hist_h = batch.agent_hist.heading[b]
        fut_h = batch.agent_fut.heading[b]
        hist_ext = batch.agent_hist_extent[b]
        fut_ext = batch.agent_fut_extent[b]

        hist_h = hist_h[:, -Th_actual:]
        fut_h = fut_h[:, :Tf_actual]
        full_h = torch.cat([hist_h, fut_h], dim=1)
        
        # Transform heading to ego frame
        full_h = full_h - ego_heading
        full_h = torch.remainder(full_h + np.pi, 2 * np.pi) - np.pi
        
        hist_ext = hist_ext[:, -Th_actual:, :2]
        fut_ext = fut_ext[:, :Tf_actual, :2]
        full_ext = torch.cat([hist_ext, fut_ext], dim=1)
        
        if full_h.dim() == 2:
            full_h = full_h.unsqueeze(-1)

        # C=7: x, y, vx, vy, heading, length, width
        data_full = torch.cat([full_pos, vx, vy, full_h, full_ext], dim=-1)  # [Na, T_actual, 7]

        # -------- 创建原始数据的有效性 mask --------
        # 检测 NaN 来确定哪些位置是有效的
        # valid_mask_raw: [Na, T_actual] bool, True = 该位置有效
        valid_mask_raw = ~torch.isnan(data_full).any(dim=-1)  # [Na, T_actual]

        # Normalize BEFORE padding
        data_full[..., [0, 1, 2, 3, 5, 6]] /= 100.0
        data_full[..., 4] /= torch.pi

        # 将 NaN 替换为 -1.0（padding 值）
        data_full = torch.nan_to_num(data_full, nan=-1.0)

        # -------- 2) agent 维度补齐/截断到 max_agents (V) --------
        Na = data_full.shape[0]
        V = max_agents
        C = 7
        T_actual = data_full.shape[1]

        # 初始化为 -1.0（padding）
        data_img = torch.full((C, T_total, V), -1.0, device=device)
        extra_img = torch.zeros((3, T_total, V), device=device)
        agent_types_img = torch.zeros((V,), dtype=torch.long, device=device)
        traj_labels = torch.zeros((V,), dtype=torch.long, device=device)
        
        # 精细的时间步 mask: [T_total, V]，True = 有效
        timestep_mask_img = torch.zeros((T_total, V), dtype=torch.bool, device=device)

        num_used_agents = min(Na, V)
        
        # 计算在 data_img 中的放置位置
        dst_start = T_hist - Th_actual
        dst_end = dst_start + T_actual

        for k in range(num_used_agents):
            data_img[:, dst_start:dst_end, k] = data_full[k].transpose(0, 1)
            agent_types_img[k] = agent_type[k]
            traj_labels[k] = agent_type[k]
            # 设置时间步 mask
            timestep_mask_img[dst_start:dst_end, k] = valid_mask_raw[k]

        # 从 timestep_mask 派生 vehicle mask
        # target_vehicle_mask: [V] 该 agent 在 target 阶段（T_hist 之后）有至少一个有效时间步
        # history_vehicle_mask: [V] 该 agent 在 history 阶段（前 T_hist 步）有至少一个有效时间步
        target_vehicle_mask = timestep_mask_img[T_hist:, :].any(dim=0)   # [V]
        history_vehicle_mask = timestep_mask_img[:T_hist, :].any(dim=0)  # [V]

        # -------- 3) 地图：vec map -> grid --------
        ego_pos_np = ego_pos.cpu().numpy().reshape(-1)
        heading = ego_heading.item()
        gt_map = {}

        if map_name and ":" in map_name:
            try:
                scene_cache: Optional[SceneCache] = None
                vec_map = map_api.get_map(
                    map_name,
                    scene_cache=scene_cache,
                    incl_road_lanes=True,
                    incl_road_areas=True,
                    incl_ped_crosswalks=True,
                    incl_ped_walkways=True,
                )

                if map_radius > 0:
                    lanes = vec_map.get_lanes_within(
                        np.array([ego_pos_np[0], ego_pos_np[1], 0.0], dtype=np.float32),
                        map_radius
                    )
                else:
                    if isinstance(vec_map.lanes, dict):
                        lanes = list(vec_map.lanes.values())
                    else:
                        lanes = vec_map.lanes
                
                def get_points_with_yaw(pts_in):
                    if pts_in.shape[0] < 2:
                        return np.zeros((0, 3))
                    if pts_in.shape[1] >= 4:
                        return pts_in[:, [0, 1, 3]]
                    xy = pts_in[:, :2]
                    diff = xy[1:] - xy[:-1]
                    yaw = np.arctan2(diff[:, 1], diff[:, 0])
                    yaw = np.append(yaw, yaw[-1])
                    return np.column_stack([xy, yaw])

                centerlines = []
                for lane in lanes:
                    pts = lane.center.interpolate(num_pts=num_points).points
                    centerlines.append(get_points_with_yaw(pts))
                
                def process_boundary_edge(edge, out_list):
                    if edge is None: return
                    pts = edge.points
                    if pts.shape[0] < 2: return
                    dists = np.linalg.norm(pts[1:, :2] - pts[:-1, :2], axis=1)
                    split_inds = np.where(dists > 5.0)[0] + 1
                    if len(split_inds) > 0:
                        segs = np.split(pts, split_inds)
                    else:
                        segs = [pts]
                    for seg in segs:
                        if seg.shape[0] < 2: continue
                        if seg.shape[0] > num_points:
                            indices = np.linspace(0, seg.shape[0] - 1, num_points).astype(int)
                            seg = seg[indices]
                        out_list.append(get_points_with_yaw(seg))

                left_boundary = []
                for lane in lanes:
                    process_boundary_edge(lane.left_edge, left_boundary)

                right_boundary = []
                for lane in lanes:
                    process_boundary_edge(lane.right_edge, right_boundary)

                crosswalks = []
                try:
                    if map_radius > 0:
                        cw_elements = vec_map.get_areas_within(
                            np.array([ego_pos_np[0], ego_pos_np[1], 0.0], dtype=np.float32),
                            MapElementType.PED_CROSSWALK,
                            map_radius
                        )
                    else:
                        cw_elements = list(vec_map.elements[MapElementType.PED_CROSSWALK].values())
                    
                    for cw in cw_elements:
                        pts = cw.polygon.points
                        if pts.shape[0] > 2 and not np.allclose(pts[0], pts[-1]):
                            pts = np.vstack([pts, pts[0]])
                        if pts.shape[0] > num_points:
                             indices = np.linspace(0, pts.shape[0] - 1, num_points).astype(int)
                             pts = pts[indices]
                        crosswalks.append(get_points_with_yaw(pts))
                except Exception:
                    pass

                gt_map = {
                    "centerlines": centerlines,
                    "left_edges": left_boundary,
                    "right_edges": right_boundary,
                    "crosswalks": crosswalks,
                }
            except Exception as e:
                print(f"Warning: Failed to load map {map_name}: {e}")
                gt_map = {}

        map_data_np, map_mask_np, map_type_np = encode_vec_map_to_grid(
            gt_map,
            ego_pos=ego_pos_np,
            ego_heading=heading,
            num_lanes=num_lanes,
            num_points=num_points,
        )

        map_data_t = torch.from_numpy(map_data_np).to(device)
        map_mask_t = torch.from_numpy(map_mask_np).to(device)
        map_type_t = torch.from_numpy(map_type_np).to(device)

        # -------- 4) scene_stats --------
        mean_xy = torch.zeros(2, device=device)
        std_xy = torch.tensor(100.0, device=device)
        scene_stats = torch.cat([mean_xy, std_xy[None]], dim=0)

        # -------- 5) 收集到 list --------
        history_data_list.append(data_img[:, :T_hist, :].unsqueeze(0))
        target_data_list.append(data_img[:, T_hist:, :].unsqueeze(0))
        labels_list.append(traj_labels.unsqueeze(0))
        map_data_list.append(map_data_t.unsqueeze(0))
        map_mask_list.append(map_mask_t.unsqueeze(0))
        map_type_list.append(map_type_t.unsqueeze(0))
        agent_types_list.append(agent_types_img.unsqueeze(0))
        scene_stats_list.append(scene_stats.unsqueeze(0))
        map_name_list.append(map_name)
        target_vehicle_mask_list.append(target_vehicle_mask.unsqueeze(0))
        history_vehicle_mask_list.append(history_vehicle_mask.unsqueeze(0))
        timestep_mask_list.append(timestep_mask_img[T_hist:, :].unsqueeze(0))  # 只保存 target 阶段的 mask
        extra_data_list.append(extra_img.unsqueeze(0))

    history_data = torch.cat(history_data_list, dim=0)
    target_data = torch.cat(target_data_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    map_data = torch.cat(map_data_list, dim=0)
    map_mask = torch.cat(map_mask_list, dim=0)
    map_type = torch.cat(map_type_list, dim=0)
    agent_types = torch.cat(agent_types_list, dim=0)
    scene_stats = torch.cat(scene_stats_list, dim=0)
    target_vehicle_mask = torch.cat(target_vehicle_mask_list, dim=0)
    history_vehicle_mask = torch.cat(history_vehicle_mask_list, dim=0)
    timestep_mask = torch.cat(timestep_mask_list, dim=0)  # [B, T_fut, V]
    extra_data = torch.cat(extra_data_list, dim=0)

    # Normalize
    history_data, target_data, map_data = normalize_batch(history_data, target_data, map_data)

    return (history_data, target_data, labels, map_data, map_mask,
            agent_types, scene_stats, map_name_list,
            target_vehicle_mask, history_vehicle_mask, timestep_mask, map_type, extra_data)


def normalize_batch(history_data, target_data, map_data):
    """
    Normalize data and encode map types.
    history_data: [B, 7, T, V]
    target_data: [B, 7, T, V]
    map_data: [B, L, P, 4]
    """
    # 1. Normalize History & Target
    # Channels: 0:x, 1:y, 2:vx, 3:vy, 4:h, 5:l, 6:w
    # /100: 0,1,2,3,5,6
    # /pi: 4
    
    # Clone to avoid side effects if needed, but here we modify in place or return new tensors
    # Note: history_data is [B, 7, T, V]
    
    # Position, Velocity, Dimensions -> / 100
    # history_data[:, [0, 1, 2, 3, 5, 6]] /= 100.0
    # target_data[:, [0, 1, 2, 3, 5, 6]] /= 100.0
    
    # # Heading -> / pi
    # history_data[:, 4] /= torch.pi
    # target_data[:, 4] /= torch.pi
    
    # 2. Normalize Map
    # map_data: [B, L, P, 4] (x, y, yaw, type)
    # x, y -> / 100
    map_data[..., :2] /= 100.0
    # yaw -> / pi
    map_data[..., 2] /= torch.pi
    
    # 3. Map Type One-Hot
    # type is at index 3.
    # Note: padding 位置的 type 是 0，one-hot 后变成 [1,0,0]
    # 但这些位置会被 map_mask 过滤掉，所以不影响
    map_types = map_data[..., 3].long().clamp(min=0, max=2)  # 确保在有效范围内
    # One-hot encode (assuming 3 types: 0, 1, 2)
    map_types_onehot = torch.nn.functional.one_hot(map_types, num_classes=3).to(dtype=map_data.dtype)
    
    # Concatenate: x, y, yaw (3 channels) + types_onehot (3 channels) = 6 channels
    map_data_new = torch.cat([map_data[..., :3], map_types_onehot], dim=-1)
    
    return history_data, target_data, map_data_new


# ----------------------------
# 安全加载 checkpoint（如果文件不存在则忽略）
# ----------------------------
def safe_load_state(path, map_location='cpu'):
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"[safe_load_state] 文件不存在：{path}，跳过加载。")
        return None
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        print(f"[safe_load_state] 加载 {path} 时异常：{e}，跳过。")
        return None


# ----------------------------
# 训练主函数（每个 spawned 进程都会运行）
# ----------------------------
def train(gpu, ngpus_per_node, args):
    is_distributed = (ngpus_per_node > 1)
    local_rank = gpu
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    if is_distributed:
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=ngpus_per_node,
            rank=local_rank
        )

    is_main = (not is_distributed) or (local_rank == 0)

    # ----------------------------
    # 构建模型 + 优化器
    # ----------------------------
    model_single = Glow(in_channel=5, condition_dim=32, n_flow=args.n_flow,
                        n_block=args.n_block, affine=args.affine, conv_lu=not args.no_lu)
    model_single.to(device)

    optimizer = torch.optim.AdamW(model_single.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = StepLR(optimizer, step_size=10000, gamma=0.96)
    
    # AMP GradScaler
    use_amp = args.amp and torch.cuda.is_available()
    scaler = GradScaler('cuda', enabled=use_amp)
    if is_main and use_amp:
        print(f"[rank {local_rank}] AMP (Mixed Precision) training enabled")
    
    # Optional torch.compile (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        if is_main:
            print(f"[rank {local_rank}] Compiling model with torch.compile...")
        model_single = torch.compile(model_single, mode='reduce-overhead')

    if args.loadckpt:
        print("Loading checkpoint...")
        if args.load_model_path is not None:
            ck = safe_load_state(args.load_model_path, map_location='cpu')
            if ck is not None:
                try:
                    model_single.load_state_dict(ck, strict=False)
                    print(f"[rank {local_rank}] 成功加载模型权重: {args.load_model_path}")
                except Exception as e:
                    print(f"[rank {local_rank}] 加载模型权重时异常: {e}")

        if args.load_optim_path is not None:
            ok = safe_load_state(args.load_optim_path, map_location='cpu')
            if ok is not None:
                try:
                    optimizer.load_state_dict(ok)
                    print(f"[rank {local_rank}] 成功加载优化器状态: {args.load_optim_path}")
                except Exception as e:
                    print(f"[rank {local_rank}] 加载优化器状态时异常: {e}")

    # if args.lr is not None:
    #     for idx, g in enumerate(optimizer.param_groups):
    #         old_lr = g.get("lr", None)
    #         g["lr"] = args.lr
    #         if idx == 0:
    #             print(f"[rank {local_rank}] 重设学习率: {old_lr} -> {args.lr}")

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model_single, device_ids=[local_rank], find_unused_parameters=True
        )
    else:
        model = model_single

    # ----------------------------
    # 构建 DataLoader：使用统一的 build_dataloader 函数
    # ----------------------------
    dataloader, map_api, sampler = build_dataloader(args, is_distributed, local_rank, ngpus_per_node)

    # z_shapes 预置
    z_shapes = calc_z_shapes(5, args.img_size_h, args.img_size_w, args.n_block)
    z_sample = [torch.randn(args.batch, *z).to(device) * args.temp for z in z_shapes]

    writer = None
    if is_main:
        run_id = time.strftime('%Y%m%d_%H%M%S')
        logdir = os.path.join(args.log_dir, f"run_{run_id}")
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=logdir)
        print(f"[rank {local_rank}] TensorBoard logdir: {logdir}")

    total_steps = args.iter
    start_iter = args.start_iter
    epoch = 0
    dataloader_iter = iter(dataloader)
    if is_main:
        pbar = tqdm(range(start_iter, total_steps), ncols=120)
    else:
        pbar = range(start_iter, total_steps)

    for i in pbar:
        if sampler is not None and (i == start_iter or (i - start_iter) % len(dataloader) == 0):
            sampler.set_epoch(epoch)
            epoch += 1
            dataloader_iter = iter(dataloader)

        try:
            batch_raw = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch_raw = next(dataloader_iter)

        # 使用统一的 process_batch 函数处理数据
        batch = process_batch(batch_raw, args, map_api, device)
        
        history_data = batch['history_data']
        target_data = batch['target_data']
        labels = batch['labels']
        map_data = batch['map_data']
        map_mask = batch['map_mask']
        agent_types = batch['agent_types']
        scene_stats = batch['scene_stats']
        map_name = batch['map_name']
        target_vehicle_mask = batch['target_vehicle_mask']
        history_vehicle_mask = batch['history_vehicle_mask']
        timestep_mask = batch['timestep_mask']
        map_type = batch['map_type']

        if torch.isnan(map_data).any() or torch.isnan(map_mask).any():
            if is_main:
                print(f"[Iter {i}] 检测到 NaN 在 map_data 或 map_mask，跳过此 batch")
            continue

        # Fix negative indices for Embedding
        agent_types = agent_types + 1
        labels = labels + 1

        use_conditional = False
        condition_input = labels if use_conditional else None

        # 对于非 trajdata 路径，需要从数据生成 timestep_mask
        # target_data: [B, C, T, V]，padding 位置的值是 -1.0
        if timestep_mask is None:
            timestep_mask = ~torch.isclose(target_data, torch.tensor(-1.0, device=device), atol=0.05).all(dim=1)  # [B, T, V]

        if i == start_iter and is_main:
            with torch.no_grad():
                with autocast('cuda', enabled=use_amp):
                    _ = model_single(target_data, condition_input, map_data, map_mask,
                                     agent_types, history_data, target_vehicle_mask, history_vehicle_mask,
                                     timestep_mask=timestep_mask)
            continue

        with autocast('cuda', enabled=use_amp):
            log_p, logdet, z = model(target_data, condition_input, map_data, map_mask,
                                     agent_types, history_data, target_vehicle_mask, history_vehicle_mask,
                                     timestep_mask=timestep_mask)

            logdet_mean = logdet.mean()
            loss_value = -(logdet_mean + log_p).mean()

        # 检测 NaN，跳过此 batch 而不是终止训练
        if torch.isnan(loss_value) or torch.isinf(loss_value) or torch.isnan(log_p).any() or torch.isnan(logdet_mean):
            if is_main:
                print(f"[Iter {i}] 检测到 NaN/Inf，跳过此 batch. Loss: {loss_value.item()}, log_p: {log_p.mean().item()}, logdet: {logdet_mean.item()}")
            # 清零梯度，跳过此迭代
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss_value).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # 记录 scaler 的 scale 值，用于判断是否跳过了优化器更新
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        
        # 只有当优化器真正执行了更新时才调用 scheduler.step()
        # 如果 scale 变小了，说明发生了 inf/nan，优化器被跳过
        if old_scale <= new_scale:
            scheduler.step()

        if is_main:
            loss_value = loss_value.mean()
            log_p_mean = log_p.mean()
            pbar.set_description(
                f"Loss: {loss_value.item():.5f}; logP: {log_p_mean.item():.5f}; lr: {optimizer.param_groups[0]['lr']:.7f}"
            )
            writer.add_scalar('train/Loss', loss_value.item(), i)
            writer.add_scalar('train/log_p', log_p_mean.item(), i)
            writer.add_scalar('train/logdet', logdet_mean.item(), i)

        # 采样与保存（和你原来的逻辑一致）
        if is_main and (i % args.sample_interval == 0 or i == start_iter + 20):
            with torch.no_grad():
                sample_labels = labels[:args.batch]
                sample_map_data = map_data[:args.batch]
                sample_map_mask = map_mask[:args.batch]
                sample_agent_types = agent_types[:args.batch]
                sample_scene_stats = scene_stats[:args.batch]
                sample_history_data = history_data[:args.batch]
                sample_map_name = map_name[:args.batch] if hasattr(map_name, '__len__') else map_name
                sample_gt = target_data[:args.batch]
                sample_target_vehicle_mask = target_vehicle_mask[:args.batch]
                sample_history_vehicle_mask = history_vehicle_mask[:args.batch]
                sample_map_type = map_type[:args.batch]
                
                valid_agents_mask = sample_history_vehicle_mask | sample_target_vehicle_mask
                sample_timestep_mask = valid_agents_mask.unsqueeze(1).repeat(1, timestep_mask.shape[1], 1)

                n_modes = args.n_modes
                con_samples_multimodal = []
                uncon_samples_multimodal = []
                uncon_samples_multimodal_no_condition = []

                temp_values = [args.temp] * n_modes

                for mode_idx in range(n_modes):
                    current_temp = temp_values[mode_idx]
                    z_sample_mode = [torch.randn(args.batch, *z).to(device) * current_temp for z in z_shapes]
                    con_sample_mode = model_single.reverse(
                        z_sample_mode, sample_labels,
                        map_data=sample_map_data, map_mask=sample_map_mask,
                        agent_types=sample_agent_types, history_data=sample_history_data,
                        target_vehicle_mask=sample_target_vehicle_mask,
                        history_vehicle_mask=sample_history_vehicle_mask,
                        timestep_mask=sample_timestep_mask
                    ).cpu().data
                    con_samples_multimodal.append(np.array(con_sample_mode))

                    z_sample_mode_uncon = [torch.randn(args.batch, *z).to(device) * current_temp for z in z_shapes]
                    uncon_sample_mode = model_single.reverse(
                        z_sample_mode_uncon,
                        map_data=sample_map_data, map_mask=sample_map_mask,
                        agent_types=sample_agent_types, history_data=sample_history_data,
                        target_vehicle_mask=sample_target_vehicle_mask,
                        history_vehicle_mask=sample_history_vehicle_mask,
                        timestep_mask=sample_timestep_mask
                    ).cpu().data
                    uncon_samples_multimodal.append(np.array(uncon_sample_mode))

                    z_sample_mode_uncon_no_condition = [torch.randn(args.batch, *z).to(device) * current_temp for z in z_shapes]
                    uncon_sample_mode_no_condition = model_single.reverse(
                        z_sample_mode_uncon_no_condition,
                        map_data=sample_map_data, map_mask=sample_map_mask,
                        agent_types=sample_agent_types,
                        target_vehicle_mask=sample_target_vehicle_mask,
                        timestep_mask=sample_timestep_mask
                    ).cpu().data
                    uncon_samples_multimodal_no_condition.append(np.array(uncon_sample_mode_no_condition))

                con_samples_multimodal = np.stack(con_samples_multimodal, axis=0)
                uncon_samples_multimodal = np.stack(uncon_samples_multimodal, axis=0)
                uncon_samples_multimodal_no_condition = np.stack(uncon_samples_multimodal_no_condition, axis=0)

                out_dir = args.sample_out_dir
                os.makedirs(out_dir, exist_ok=True)
                out_name = os.path.join(out_dir, f"{str(i + 1).zfill(6)}_11_27_npz_stage1.npz")
                np.savez(
                    out_name,
                    con_sample_multimodal=con_samples_multimodal,
                    uncon_sample_multimodal=uncon_samples_multimodal,
                    uncon_sample_multimodal_no_condition=uncon_samples_multimodal_no_condition,
                    gt=np.array(sample_gt.cpu().data),
                    labels=np.array(sample_labels.cpu().data),
                    maps=np.array(sample_map_data.cpu().data),
                    map_mask=np.array(sample_map_mask.cpu().data),
                    agent_types=np.array(sample_agent_types.cpu().data),
                    scene_stats=np.array(sample_scene_stats.cpu().data),
                    history_data=np.array(sample_history_data.cpu().data),
                    map_name=np.array(sample_map_name),
                    map_type=np.array(sample_map_type.cpu().data),
                    target_vehicle_mask=np.array(sample_target_vehicle_mask.cpu().data),
                    history_vehicle_mask=np.array(sample_history_vehicle_mask.cpu().data),
                    timestep_mask=np.array(sample_timestep_mask.cpu().data),
                    n_modes=n_modes
                )
                print(f"[rank {local_rank}] Saved samples to {out_name}")

        if is_main and (i % args.save_interval == 0):
            ckpt_dir = args.ckpt_dir
            os.makedirs(ckpt_dir, exist_ok=True)
            model_path = os.path.join(ckpt_dir, f"model_3B16F_11_27_stage1_npz.pt")
            optim_path = os.path.join(ckpt_dir, f"optim_3B16F_11_27_stage1_npz.pt")
            torch.save(model_single.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optim_path)
            print(f"[rank {local_rank}] Saved checkpoint: {model_path}, {optim_path}")

    if writer is not None:
        writer.close()
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


# ----------------------------
# main 启动逻辑
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description='Conditional Glow trainer (DDP friendly)')
    parser.add_argument('--batch', default=1, type=int, help='batch size')
    parser.add_argument('--iter', default=400000, type=int, help='maximum iterations')
    parser.add_argument('--start_iter', default=200000, type=int, help='start iteration (for resumed runs)')
    parser.add_argument('--n_flow', default=16, type=int, help='number of flows in one block')
    parser.add_argument('--n_block', default=3, type=int, help='number of blocks')
    parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
    parser.add_argument('--affine', action='store_true', help='use affine coupling instead of additive')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--img_size_h', default=40, type=int, help='image size height (T_total)')
    parser.add_argument('--img_size_w', default=32, type=int, help='image size width (max agents V)')
    parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
    parser.add_argument('--n_sample', default=20, type=int, help='number of samples')
    parser.add_argument('--n_modes', default=6, type=int, help='number of multimodal samples to save')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers')
    parser.add_argument('--sample_interval', default=2000, type=int, help='how many iters between sample saves')
    parser.add_argument('--save_interval', default=2000, type=int, help='how many iters between checkpoint saves')
    parser.add_argument('--log_dir', default='./runs', type=str, help='tensorboard log dir')
    parser.add_argument('--ckpt_dir', default=f'/home2/wangyj/SG/MapFlow/mapGlow/results/checkpoint/', type=str,
                        help='checkpoint dir')
    parser.add_argument('--sample_out_dir', default=f'/home2/wangyj/SG/MapFlow/mapGlow/results/samples/11_12/',
                        type=str, help='samples output dir')
    parser.add_argument('--path', default='/home2/wangyj/Dataset/flow_data/train_dataset_9_2_normalized_resorted.npz',
                        type=str, help='Path to dataset npz (旧 npz 流程用)')
    parser.add_argument('--map_path', default=f'/home2/wangyj/Dataset/flow_data/train_dataset_9_2_map.npz',
                        type=str, help='Path to map data npz (旧 npz 流程用)')
    parser.add_argument('--load_model_path',
                        default=f'/home2/wangyj/SG/MapFlow/mapGlow/results/checkpoint/model_1B32F_11_27_stage1_npz.pt',
                        type=str, help='pretrained model path (optional)')
    parser.add_argument('--load_optim_path',
                        default=f'/home2/wangyj/SG/MapFlow/mapGlow/results/checkpoint/optim_1B32F_11_27_stage1_npz.pt',
                        type=str, help='pretrained optimizer path (optional)')
    parser.add_argument('--newdataset', action='store_true', help='use old/new npz dataset format')
    parser.add_argument('--loadckpt', action='store_true', help='load checkpoint')

    # 新增：trajdata + Waymo 流式开关
    parser.add_argument('--use_trajdata', action='store_true',
                        help='使用 trajdata+Waymo 流式加载，而不是 npz')
    parser.add_argument('--td_dataroot', type=str, default='',
                        help='trajdata 能识别的 Waymo 数据根目录')
    parser.add_argument('--td_split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Waymo split (trajdata: waymo_train/val/test)')
    parser.add_argument('--map_radius', type=float, default=-1,
                        help='Map query radius. -1 means use all lanes.')
    parser.add_argument('--num_lanes', type=int, default=64,
                        help='Max number of lanes for map encoding. Default: 64 for npz, 256 for trajdata.')
    parser.add_argument('--amp', action='store_true',
                        help='Enable automatic mixed precision (AMP) training for faster speed')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for model optimization (PyTorch 2.0+)')

    args = parser.parse_args()

    # 根据数据源设置默认 num_lanes
    if args.num_lanes is None:
        args.num_lanes = 256 if args.use_trajdata else 64

    print("训练参数：")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '12350')

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node == 0:
        print("未检测到GPU，使用单进程 CPU 模式运行（仅用于调试，训练速度会很慢）")
        train(0, 1, args)
    else:
        print(f"检测到 GPU 数量: {ngpus_per_node}，使用 mp.spawn 启动 DDP")
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args,))


if __name__ == "__main__":
    main()
