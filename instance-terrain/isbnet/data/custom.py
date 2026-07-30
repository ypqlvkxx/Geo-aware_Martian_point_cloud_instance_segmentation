import numpy as np
import scipy.interpolate
import scipy.ndimage
import torch
from torch.utils.data import Dataset

import math
import os.path as osp
from glob import glob
from ..ops import voxelization_idx
from tqdm import tqdm
from os.path import join
import os
import sys



class CustomDataset(Dataset):

    CLASSES = None

    def __init__(self, numclass, data_root, prefix, suffix, num_layers, sub_grid_size, sub_sampling_ratio, k_n, voxel_cfg=None, training=True, repeat=1, logger=None):
        self.numclass = numclass
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.repeat = repeat
        self.logger = logger
        self.num_layers = num_layers
        self.sub_grid_size = sub_grid_size
        self.sub_sampling_ratio = sub_sampling_ratio
        self.k_n = k_n
        self.mode = "train" if training else "test"
        self.filenames = self.get_filenames()
        self.num_per_class = [0 for _ in range(self.numclass)]

        self.logger.info(f"Load {self.mode} dataset: {len(self.filenames)} scans")
        self.weights = self.get_class_weight()
   
    def get_class_weights(self):
        # pre-calculate the number of points in each category
        num_per_class = [0 for _ in range(self.numclass)]
        for file_path in tqdm(os.listdir(join(self.data_root,'preprocess'))):
            if self.mode in file_path:
                file_path = join(self.data_root,'preprocess',file_path)
                label = torch.load(file_path)[2].astype(int).reshape(-1)
                inds, counts = np.unique(label, return_counts=True)
                for i, c in zip(inds, counts):
                    if i == -1:      # 0 : unlabeled
                        continue
                    else:
                        num_per_class[i] += c

        num_per_class = np.array(num_per_class)
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        indices_to_zero = np.where(ce_label_weight == 50)[0]  
        ce_label_weight[indices_to_zero] = 0 
        num = len(ce_label_weight) - len(indices_to_zero)
        ce_label_weight = ce_label_weight * num / float(sum(ce_label_weight))
        return list(ce_label_weight)

    def get_class_weight(self):
        return self.get_class_weights()

    def get_filenames(self):
        if self.prefix == "trainval":
            filenames_train = glob(osp.join(self.data_root, "train", "*" + self.suffix))
            filenames_val = glob(osp.join(self.data_root, "val", "*" + self.suffix))

            filenames = filenames_train + filenames_val
        else:
            filenames_all = []
            for p in self.prefix:
                filenames = glob(osp.join(self.data_root, "preprocess", p + "*" + self.suffix))
                assert len(filenames) > 0, f"Empty {p}"
                filenames_all.extend(filenames)
        filenames_all = sorted(filenames_all * self.repeat)
        return filenames_all

    def load(self, filename):
        return torch.load(filename)

    def __len__(self):
        return len(self.filenames)
    

    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype("float32") / 3
        blur1 = np.ones((1, 3, 1)).astype("float32") / 3
        blur2 = np.ones((1, 1, 3)).astype("float32") / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype("float32") for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    def dataAugment(self, xyz, jitter=False, flip=False, rot=False, prob=1.0):
        m = np.eye(3)
        if jitter and np.random.rand() < prob:
            m += np.random.randn(3, 3) * 0.001

        if rot and np.random.rand() < prob:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]]
            )
        else:
            # Empirically, slightly rotate the scene can match the results from checkpoint
            theta = 0.35 * math.pi
            m = np.matmul(
                m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]]
            )

        rotated_xyz = np.matmul(xyz, m)

        # NOTE flip
        if flip:
            for i in (0, 1):
                if np.random.rand() < 0.5:
                    rotated_xyz[:, i] = -rotated_xyz[:, i]

        return rotated_xyz

    def crop(self, xyz, step=32):
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]
        spatial_shape = np.array([self.voxel_cfg.spatial_shape[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while valid_idxs.sum() > self.voxel_cfg.max_npoint:
            step_temp = step
            if valid_idxs.sum() > 1e6:
                step_temp = step * 2
            offset = np.clip(spatial_shape - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < spatial_shape).sum(1) == 3)
            spatial_shape[:2] -= step_temp
        return xyz_offset, valid_idxs

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def transform_train(self, xyz, rgb, semantic_label, instance_label, spp, aug_prob=1.0):
        xyz_middle = self.dataAugment(xyz, True, True, True, aug_prob)
        return xyz, xyz_middle, rgb, semantic_label, instance_label, spp

    def transform_test(self, xyz, rgb, semantic_label, instance_label, spp):
        xyz_middle = self.dataAugment(xyz, False, False, False)
        return xyz, xyz_middle, rgb, semantic_label, instance_label, spp

    def __getitem__(self, index):
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, "")

        # if scan_id in ['scene0636_00', 'scene0154_00']:
        #     return self.__getitem__(0)
        xyz, rgb, semantic_label, instance_label, spp = self.load(filename)

        data = (
            self.transform_train(xyz, rgb, semantic_label, instance_label, spp)
            if self.training
            else self.transform_test(xyz, rgb, semantic_label, instance_label, spp)
        )
        if data is None:
            return None

        xyz, xyz_middle, rgb, semantic_label, instance_label, spp = data

        if np.min(rgb)<-2 or np.max(rgb)>2:
            print("1")

        #inst_num = int(instance_label.max()) + 1
        inst_num = int(instance_label.max()) 

        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb).float()
        if self.training:
            feat += torch.randn(1) * 0.1

        semantic_label = torch.from_numpy(semantic_label)
        instance_label = torch.from_numpy(instance_label)

        if isinstance(spp, np.ndarray):
            spp = torch.from_numpy(spp)
            
        spp = torch.unique(spp, return_inverse=True)[1]

        return (
            scan_id,
            coord,
            coord_float,
            feat,
            semantic_label,
            instance_label,
            spp,
            inst_num,
        )


    def collate_fn(self, batch):
        scan_ids = []
        coords = []
        coords_float = []
        feats = []
        semantic_labels = []
        instance_labels = []

        spps = []

        instance_batch_offsets = [0]

        total_inst_num = 0
        batch_id = 0

        spp_bias = 0

        for data in batch:
            if data is None:
                continue
            (
                scan_id,
                coord,
                coord_float,
                feat,
                semantic_label,
                instance_label,
                spp,
                inst_num,
            ) = data

            spp += spp_bias
            spp_bias = spp.max().item() + 1

            instance_label[instance_label != -100] += total_inst_num
            total_inst_num += inst_num
            scan_ids.append(scan_id)
            coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            semantic_labels.append(semantic_label)
            instance_labels.append(instance_label)
            spps.append(spp)

            instance_batch_offsets.append(total_inst_num)

            batch_id += 1
        assert batch_id > 0, "empty batch"
        if batch_id < len(batch):
            self.logger.info(f"batch is truncated from size {len(batch)} to {batch_id}")

        # merge all the scenes in the batch
        coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
        batch_idxs = coords[:, 0].int()
        coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        semantic_labels = torch.cat(semantic_labels, 0).long()  # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)
        spps = torch.cat(spps, 0).long()

        instance_batch_offsets = torch.tensor(instance_batch_offsets, dtype=torch.long)

        return {
            "scan_ids": scan_ids,
            "batch_idxs": batch_idxs,
            "coords_float": coords_float,
            "feats": feats,
            "semantic_labels": semantic_labels,
            "instance_labels": instance_labels,
            "instance_batch_offsets": instance_batch_offsets,
            "batch_size": batch_id,
        }
