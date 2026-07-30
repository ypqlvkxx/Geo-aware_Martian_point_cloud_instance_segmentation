import numpy as np
from collections import defaultdict
from osgeo import gdal
import os

def complete_labels(datapath, dem_geotrans, window_radius=2, default_label=None):
    # 提取i和j的范围
    data = np.loadtxt(datapath)
    i_j = data[:,0:2].astype(np.int32)
    sem_label = data[:,2:3].astype(np.int32)
    ins_label = data[:,3:4].astype(np.int32)
    i_coords = [point[0] for point in i_j]
    j_coords = [point[1] for point in i_j]
    i_min, i_max = min(i_coords), max(i_coords)
    j_min, j_max = min(j_coords), max(j_coords)
    
    sem_data = np.concatenate((i_j,sem_label),axis=1)
    ins_data = np.concatenate((i_j,ins_label),axis=1)

    # 创建坐标到标签的快速查询字典
    coord_sem_label = {(i, j): label for (i, j, label) in sem_data}
    coord_ins_label = {(i, j): ins for (i, j, ins) in ins_data}
    
    # 生成需要补全的所有坐标点
    existing_points = set(coord_sem_label.keys())
    points_to_fill = [
        (i, j) 
        for i in range(i_min, i_max + 1) 
        for j in range(j_min, j_max + 1) 
        if (i, j) not in existing_points
    ]
    
    # 补全每个点的标签
    completed = []
    for (i, j) in points_to_fill:
        # 统计窗口内的标签
        sem_label_counts = defaultdict(int)
        ins_label_counts = defaultdict(int)
        for di in range(-window_radius, window_radius + 1):
            for dj in range(-window_radius, window_radius + 1):
                ni, nj = i + di, j + dj
                if (ni, nj) in coord_sem_label:
                    sem_label = coord_sem_label[(ni, nj)]
                    ins_label = coord_ins_label[(ni, nj)]
                    sem_label_counts[sem_label] += 1
                    ins_label_counts[ins_label] += 1
        
        # 确定最多数的标签
        if sem_label_counts:
            max_sem_label = max(sem_label_counts.items(), key=lambda x: x[1])[0]
            max_ins_label = max(ins_label_counts.items(), key=lambda x: x[1])[0]
            x_g = np.round(dem_geotrans[0] + i * dem_geotrans[1] + j * dem_geotrans[2], 1)
            y_g = np.round(dem_geotrans[3] + i * dem_geotrans[4] + j * dem_geotrans[5], 1)
            completed.append((i,j, max_sem_label, max_ins_label))
        else:
            # 窗口内无数据时的处理
            if default_label is not None:
                x_g = np.round(dem_geotrans[0] + i * dem_geotrans[1] + j * dem_geotrans[2], 1)
                y_g = np.round(dem_geotrans[3] + i * dem_geotrans[4] + j * dem_geotrans[5], 1)
                completed.append((i,j, default_label, default_label))
            else:
                # 可以选择跳过或者抛出警告
                pass
    
    raw_data = np.concatenate((data[:,0:2],data[:,2:3],data[:,3:4]),axis=1)
    completed = np.array(completed)
    # 合并原始数据和补全数据
    return np.concatenate((raw_data , completed),axis =0)

def read_dem_geotrans(dempath):
    dataset = gdal.Open(dempath) # 打开DEM数据
    if (dataset == None):
        print(dempath+"没有fusion")
    else:
        dem_XSize = dataset.RasterXSize  # 列数
        dem_YSize = dataset.RasterYSize  # 行数
        dem_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        return dem_geotrans


path = r'' #点云补全之前的路劲（即刚刚预测后的点云路径）
tif_path = r'' #原始DEM的路径
default_label = 0
dem_geotrans = read_dem_geotrans(tif_path)

outpath = r'' #输出路径
for txt_file in os.listdir(path):
    datapath = os.path.join(path, txt_file)
    completed_data = complete_labels(datapath, dem_geotrans, window_radius=2, default_label = default_label)
    outdempath = os.path.join(outpath,txt_file)
    print("saving:"+str(txt_file))
    fmt = ' %d %d %.1f %.1f' #(x y label ins_label i j)
    np.savetxt(outdempath, completed_data, fmt = fmt)
    

    