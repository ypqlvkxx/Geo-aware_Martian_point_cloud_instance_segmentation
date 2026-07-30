import numpy as np
from collections import defaultdict
from osgeo import gdal
import os

class Instance:
    def __init__(self, points, semantic_label):
        self.points = points          # 实例所有点的全局坐标 (N, 2)
        self.semantic_label = semantic_label
        self.bbox = self._compute_bbox()
    
    def _compute_bbox(self):
        """计算全局坐标系下的边界框 (xmin, ymin, xmax, ymax)"""
        x_coords = self.points[:, 0]
        y_coords = self.points[:, 1]
        return (
            np.min(x_coords), np.min(y_coords),
            np.max(x_coords), np.max(y_coords)
        )

def extract_instances(coordinates, semantic_labels, instance_labels):
    """从单个切片中提取实例信息（按实例标签 ID 分组）"""
    instances = []
    labels_flat = instance_labels.ravel()
    sem_flat = semantic_labels.ravel()
    for label in np.unique(labels_flat):
        if label <= 0:
            continue
        mask = labels_flat == label
        points = coordinates[mask]
        semantic_label = int(np.argmax(np.bincount(sem_flat[mask])))
        instances.append(Instance(points, semantic_label))
    return instances

class InstanceMerger:
    def __init__(self):
        self.semantic_groups = defaultdict(list)  # {semantic_label: [Instance]}
        self.union_find = {}                     # {instance_id: root}
        self.instance_id_map = {}                # {hash(Instance): instance_id}
        self.next_id = 0
    
    def add_instances(self, instances):
        """添加实例到合并器"""
        for inst in instances:
            # 为每个实例分配唯一标识
            inst_id = self.next_id
            self.next_id += 1
            self.instance_id_map[id(inst)] = inst_id
            self.union_find[inst_id] = inst_id
            self.semantic_groups[inst.semantic_label].append(inst)
    
    def merge(self, merge_distance=5, cell_size=100, iou_threshold=0.3):
        """
        合并空间上相邻或重叠的同类实例。

        相邻实例的 bbox 交集面积为 0，IoU 也为 0，因此不能仅靠 IoU。
        使用 bbox 间隙距离 + 可选 IoU 联合判断。
        """
        for instances in self.semantic_groups.values():
            grid = defaultdict(list)
            for inst in instances:
                xmin, ymin, xmax, ymax = inst.bbox
                min_gx = int(xmin // cell_size)
                max_gx = int(xmax // cell_size)
                min_gy = int(ymin // cell_size)
                max_gy = int(ymax // cell_size)
                for gx in range(min_gx, max_gx + 1):
                    for gy in range(min_gy, max_gy + 1):
                        grid[(gx, gy)].append(inst)

            margin_cells = int(np.ceil(merge_distance / cell_size)) + 1
            for inst in instances:
                xmin, ymin, xmax, ymax = inst.bbox
                min_gx = int((xmin - merge_distance) // cell_size)
                max_gx = int((xmax + merge_distance) // cell_size)
                min_gy = int((ymin - merge_distance) // cell_size)
                max_gy = int((ymax + merge_distance) // cell_size)

                for gx in range(min_gx - margin_cells, max_gx + margin_cells + 1):
                    for gy in range(min_gy - margin_cells, max_gy + margin_cells + 1):
                        for other_inst in grid.get((gx, gy), []):
                            if inst is other_inst:
                                continue
                            if self._should_merge(
                                inst.bbox, other_inst.bbox, merge_distance, iou_threshold
                            ):
                                self._union_instances(inst, other_inst)

    def _bbox_gap(self, bbox1, bbox2):
        """两个轴对齐 bbox 之间的最小间隙（像素），相邻或重叠时为 0"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        dx = max(0, max(x1_min - x2_max, x2_min - x1_max))
        dy = max(0, max(y1_min - y2_max, y2_min - y1_max))
        return max(dx, dy)

    def _calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的 IoU"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        return inter_area / (area1 + area2 - inter_area + 1e-6)

    def _should_merge(self, bbox1, bbox2, merge_distance, iou_threshold):
        gap = self._bbox_gap(bbox1, bbox2)
        if gap <= merge_distance:
            return True
        if iou_threshold > 0 and self._calculate_iou(bbox1, bbox2) >= iou_threshold:
            return True
        return False
    
    def _union_instances(self, inst1, inst2):
        """合并两个实例"""
        id1 = self.instance_id_map[id(inst1)]
        id2 = self.instance_id_map[id(inst2)]
        root1 = self._find(id1)
        root2 = self._find(id2)
        if root1 != root2:
            self.union_find[root2] = root1
    
    def _find(self, inst_id):
        """并查集查找根节点"""
        if self.union_find[inst_id] != inst_id:
            self.union_find[inst_id] = self._find(self.union_find[inst_id])
        return self.union_find[inst_id]
    
def build_global_mask(merger, global_shape):
    """生成全局实例标签矩阵"""
    global_mask = np.zeros(global_shape, dtype=int)
    # 分配全局ID
    root_to_global = {}
    next_global_id = 1
    for inst_id in merger.union_find:
        root = merger._find(inst_id)
        if root not in root_to_global:
            root_to_global[root] = next_global_id
            next_global_id += 1
    # 填充全局掩膜
    for sem_label, instances in merger.semantic_groups.items():
        for inst in instances:
            root = merger._find(merger.instance_id_map[id(inst)])
            global_id = root_to_global[root]
            # 直接将实例点写入全局掩膜（忽略冲突）
            for (x, y) in inst.points:
                if 0 <= y < global_shape[0] and 0 <= x < global_shape[1]:
                    if global_mask[y, x] == 0:
                        global_mask[y, x] = global_id
    return global_mask

def read_dem_geotrans(dempath):
    dataset = gdal.Open(dempath) # 打开DEM数据
    if (dataset == None):
        print(dempath+"没有fusion")
    else:
        dem_XSize = dataset.RasterXSize  # 列数
        dem_YSize = dataset.RasterYSize  # 行数
        dem_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        return dem_geotrans, dem_XSize, dem_YSize


dir = r''
tif_path = r''
default_label = 0
dem_geotrans, dem_XSize, dem_YSize = read_dem_geotrans(tif_path)

all_points = []
for file in os.listdir(dir):
    file_path = os.path.join(dir, file)
    data = np.loadtxt(file_path)
    all_points.append((data[:,0:2],data[:,2:3],data[:,3:4]))

global_shape = (dem_YSize, dem_XSize)

# 1. 解析所有实例
merger = InstanceMerger()

# 1. 提取所有实例
for point in all_points:
    coords, sem, inst = point
    coords = coords.astype(np.int32); sem = sem.astype(np.int32) ; inst = inst.astype(np.int32)
    instances = extract_instances(coords, sem, inst)
    merger.add_instances(instances)

# 2. 合并实例
merger.merge(merge_distance=5, iou_threshold=0.3)

global_mask = build_global_mask(merger, global_shape)
ij_coords = np.arange(1, dem_YSize*dem_XSize+1).reshape(dem_YSize, dem_XSize)
ij_coords = np.empty((dem_YSize, dem_XSize), dtype=object)

ij_coords = np.zeros((dem_YSize*dem_XSize,3), dtype=int)
count = 0
for i in range(dem_YSize):
    for j in range(dem_XSize):
        ij_coords[count,:] = np.array([i,j,global_mask[i,j]])# 如果行列从1开始计数
        count += 1

out_path = r''
np.savetxt(out_path, ij_coords)
