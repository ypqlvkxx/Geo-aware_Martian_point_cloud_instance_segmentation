import imp
import numpy as np
import torch

from .custom_large import CustomLargeDataset
from tqdm import tqdm
from os.path import join
from isbnet.util import data_process as DP

class TerrainLargeDataset(CustomLargeDataset):

    CLASSES = (
        'background', 'crater', 'terrace','island','mesa','canyon'
    )

    BENCHMARK_SEMANTIC_IDXS = [i for i in range(len(CLASSES))]  # NOTE DUMMY values just for save results
    
    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def load(self, filename):
        if self.prefix == "large":
            xyz, rgb, ij = torch.load(filename)
        else:
            xyz, rgb , ij = torch.load(filename)

        # NOTE currently stpls3d does not have spps, we will add later
        # spp = np.zeros(xyz.shape[0], dtype=np.long)
        xyz = self.pc_normalize(xyz)
        rgb = self.pc_normalize(rgb)
        xyz = xyz * 10
        spp = np.arange(xyz.shape[0], dtype=np.long)
        #rgb = rgb[:,np.newaxis]

        return xyz, rgb, ij, spp

    def get_neighbor(self, batch_pc):
        #features = batch_pc
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []
        for i in range(self.num_layers):
            neighbour_idx = DP.DataProcessing.knn_search(batch_pc, batch_pc, self.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // self.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // self.sub_sampling_ratio[i], :]
            up_i = DP.DataProcessing.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points
        input_list = input_points + input_neighbors + input_pools + input_up_samples
        return input_list

    def collate_fn(self, batch):
        scan_ids = []
        coords = []
        coords_float = []
        feats = []
        i_js = []
        
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
                i_j,
                spp,
            ) = data

            scan_ids.append(scan_id)
            coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            i_js.append(i_j)

            instance_batch_offsets.append(total_inst_num)

            batch_id += 1
        assert batch_id > 0, "empty batch"
        if batch_id < len(batch):
            self.logger.info(f"batch is truncated from size {len(batch)} to {batch_id}")

        flat_inputs = self.get_neighbor(torch.stack(coords_float, 0))

        xyz, neigh_idx, sub_idx, interp_idx = [], [], [], []
        for tmp in flat_inputs[:self.num_layers]:
            xyz.append(tmp)
        for tmp in flat_inputs[self.num_layers: 2 * self.num_layers]:
            neigh_idx.append(torch.from_numpy(tmp).long())
        for tmp in flat_inputs[2 * self.num_layers:3 * self.num_layers]:
            sub_idx.append(torch.from_numpy(tmp).long())
        for tmp in flat_inputs[3 * self.num_layers:4 * self.num_layers]:
            interp_idx.append(torch.from_numpy(tmp).long())
        
        # merge all the scenes in the batch
        coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
        batch_idxs = coords[:, 0].int()
        coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        i_js = torch.cat(i_js, 0).to(torch.int32)  # float (N, 2)

        instance_batch_offsets = torch.tensor(instance_batch_offsets, dtype=torch.long)

        

        return {
            "scan_ids": scan_ids,
            "batch_idxs": batch_idxs,
            "coords_float": coords_float,
            "xyz": xyz,
            "neigh_idx": neigh_idx,
            "sub_idx": sub_idx,
            "interp_idx": interp_idx,
            "feats": feats,
            "ij": i_js,
            "instance_batch_offsets": instance_batch_offsets,
            "batch_size": batch_id,
        }