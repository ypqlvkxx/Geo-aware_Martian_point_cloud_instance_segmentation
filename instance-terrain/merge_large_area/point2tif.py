import math
import numpy as np
from osgeo import gdal, osr


def read_dem_geotrans(dempath):
    dataset = gdal.Open(dempath) # 打开DEM数据
    if (dataset == None):
        print(dempath+"没有fusion")
    else:
        dem_XSize = dataset.RasterXSize  # 列数
        dem_YSize = dataset.RasterYSize  # 行数
        dem_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        projection = dataset.GetProjection()  
        return dem_geotrans, projection

def array_to_geotiff(data, dem_geotrans, projection, output_filename, epsg=4326, nodata=-9999):
    """
    将包含(x, y, label)的数组转换为GeoTIFF文件。
    
    参数:
        data (list): 列表，每个元素为(x, y, label)。
        resolution (float): 输出栅格的分辨率（单位与坐标相同）。
        output_filename (str): 输出TIFF文件路径。
        epsg (int): 坐标系EPSG代码，默认为4326（WGS84）。
        nodata (int): NoData值，默认为-9999。
    """
    # 提取坐标和标签
    # x_coords = data[:,0]
    # y_coords = data[:,1]
    j_coords = data[:,0]
    i_coords = data[:,1]
    labels = data[:,2]
    # i_coords = data[:,3]
    # j_coords = data[:,4]
    
    
    min_i, max_i = min(i_coords), max(i_coords)
    min_j, max_j = min(j_coords), max(j_coords)

    # 计算栅格的行数和列数
    cols = int(max_i - min_i) 
    rows = int(max_j - min_j) 
    
    # 创建数据数组并填充NoData
    raster_array = np.full((rows, cols), nodata, dtype=np.int32)
    
    # 将每个点的标签填充到对应位置
    for i, j, label in data:
        row = int(i)
        col = int(j)
        if 0 <= row < rows and 0 <= col < cols:
            raster_array[row, col] = int(label)
        else:
            print(f"警告：点({i}, {j})超出栅格范围，已忽略。")
    
    # 创建GeoTIFF文件
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_filename, cols, rows, 1, gdal.GDT_Int32)
    
    # 设置地理变换参数（左上角x, 东西分辨率, 旋转, 左上角y, 旋转, 南北分辨率）
    geotransform = dem_geotrans
    out_ds.SetGeoTransform(geotransform)
    
    # 设置投影
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    out_ds.SetProjection(projection)
    
    # 写入数据并设置NoData值
    band = out_ds.GetRasterBand(1)
    band.WriteArray(raster_array)
    band.SetNoDataValue(nodata)
    
    # 关闭数据集以确保写入磁盘
    out_ds = None

# 示例使用
if __name__ == "__main__":
    tif_path = r'' #原始DEM的路径
    default_label = 0
    dem_geotrans, projection = read_dem_geotrans(tif_path)
    path = r'' #合并后的点云路径
    out_file = r'' #输出路径
    data = np.loadtxt(path)
    array_to_geotiff(data, dem_geotrans,projection, out_file)