import os
import sys
import random
import json

import numpy as np
import torch.utils.data as data
import h5py

# Set up the base directory and add it to the system path for local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import data_transforms
from .build import DATASETS


@DATASETS.register_module()
class ABCMulti(data.Dataset):
    """
    Real ABC-multi dataset (from H5s produced by your quadric pipeline).

    Expects:
      data_path/
        h5/                 00000022.h5, 00000023.h5, ...
        train.txt | test.txt   -> IDs like '00000022' (with or without .h5)

    H5 layout (per your script):
      /points                : (N,4) [x,y,z,primitive_id]
      /quadrics/coeffs       : (K,10) float64
      /quadrics/prim_ids     : (K,)   int32
      /quadrics/types        : (K,)   utf-8 strings (optional)
      /normalization/center  : (3,)
      /normalization/scale   : ()

    Returns (same 5-tuple/signature you used before):
      mid, (
        points_gt   : [N, 3] float32
        labels_gt   : [N]    int64  (primitive_id, can be -1)
        planes_gt   : [K,10] float32  (uniform quadric-10)
        planes_index: [K,1]  int64    (prim_ids reshaped)
        points_pc   : [P,3]  float32  (P=N_POINTS; subsampled from GT if no PC source)
      )
    """

    def __init__(self, config, logger=None):
        """
        Args (typical):
          config.data_path (str): dataset root
          config.subset (str): 'train' or 'test'
          config.N_POINTS (int): desired input size for points_pc (e.g., 2048)
          config.NUM_PLANES (int): (only used by transforms like UpSamplePlanes, if you use them)
          config.augment (bool): whether to apply data augmentation (default: False)
          Optional:
            config.h5_path (str) or config.h5_dir (str): subdir for H5s (default: 'h5')
            config.list_file (str): override list file path
            config.keep_types (bool): also load 'types' (not returned by default)
            config.points_pc_source (str|None): Optional path to precomputed partial PCs (ignored by default)
            config.h5_suffix (str): default '.h5'
        """
        self.root = os.path.abspath(config.data_path)
        self.subset = config.subset
        self.num_points = int(config.N_POINTS)
        self.num_planes = int(getattr(config, "NUM_PLANES", 0))  # used only by transforms
        self.augment = getattr(config, "augment", False)

        # Where the H5s live
        self.h5_subdir = getattr(config, "h5_path", None) or getattr(config, "h5_dir", "h5")
        self.h5_dir = os.path.join(self.root, self.h5_subdir)

        # Optional external partial PCs (unused by default; we will subsample from GT)
        self.pc_source = getattr(config, "points_pc_source", None)

        # Filelist
        list_file = getattr(config, "list_file", os.path.join(self.root, f"{self.subset}.txt"))
        if not os.path.isfile(list_file):
            raise FileNotFoundError(f"List file not found: {list_file}")
        with open(list_file, "r") as f:
            ids = [ln.strip() for ln in f if ln.strip()]

        self.h5_suffix = getattr(config, "h5_suffix", ".h5")
        self.items = []
        for raw in ids:
            # Accept '00000022' or '00000022.h5'
            stem = raw
            if stem.endswith(self.h5_suffix):
                stem = stem[: -len(self.h5_suffix)]
            h5_path = os.path.join(self.h5_dir, stem + self.h5_suffix)
            if not os.path.isfile(h5_path):
                raise FileNotFoundError(f"H5 not found for id '{stem}': {h5_path}")
            self.items.append({"model_id": stem, "h5_path": h5_path})
        self.items = self.items[:60]  # DEBUG: limit dataset size
        
        self.transforms = data_transforms.Compose([
            {
                "callback": "UpSamplePlanes",
                "parameters": {"n_planes": self.num_planes},
                "objects": ["prim_coeffs", "prim_types", "prim_ids"],
            },
            {
                "callback": "ToTensor",
                "objects": ["pts", "pts_ids", "prim_coeffs", "prim_ids"],
            },
        ])

    def __len__(self):
        return len(self.items)
    
    @staticmethod
    def quadric_to_mats(coeffs10):
        A,B,C,D,E,F,G,H,I,J = map(float, coeffs10)
        Q = np.array([[A, D, E],
                    [D, B, F],
                    [E, F, C]], dtype=np.float32)
        b = np.array([G, H, I], dtype=np.float32)
        return Q, b, float(J)
    
    @staticmethod
    def mats_to_quadric(Q, b, J):
        return np.array([Q[0,0], Q[1,1], Q[2,2], Q[0,1], Q[0,2], Q[1,2],
                        b[0],    b[1],    b[2],    J], dtype=np.float32)


    def __getitem__(self, idx):
        rec = self.items[idx]
        mid = rec["model_id"]
        h5_path = rec["h5_path"]

        with h5py.File(h5_path, "r") as h5:
            # ---- Points & labels ----
            pts_all = np.asarray(h5["points"][:])  # (N,4) float + int
            pts = pts_all[:, :3].astype(np.float32)       # (N,3)
            pts_ids = pts_all[:, 3].astype(np.int64)      # (N,)

            # ---- Quadrics ----
            qgrp = h5["quadrics"]
            prim_coeffs = np.asarray(qgrp["coeffs"][:], dtype=np.float64).astype(np.float32)  # (K,10)

            # stored as variable-length utf-8 strings; keep as object array of Python str
            prim_types = np.asarray(qgrp["types"][:], dtype=object)
            prim_types = np.array([
                data_transforms.PRIMITIVE_TYPE_MAP[str(x.decode('utf-8') if isinstance(x, bytes) else x).lower()]
                for x in prim_types
            ], dtype=np.int64).reshape(-1, 1)


        data = {
            "pts": pts,                       # (N,3) float32
            "pts_ids": pts_ids,               # (N,)  int64
            "prim_coeffs": prim_coeffs,       # (K,10) float32
            "prim_types": prim_types,         # (K,1) int64
        }

        if self.transforms is not None:
            data = self.transforms(data)

        return mid, (
            data["pts"],          # [N,3]
            data["pts_ids"],      # [N]
            data["prim_coeffs"],  # [maxK,10]
            data["prim_types"],   # [maxK,1]
        )
