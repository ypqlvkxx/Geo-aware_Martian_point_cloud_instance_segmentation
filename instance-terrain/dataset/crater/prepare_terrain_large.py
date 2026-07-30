import numpy as np
import torch
from scipy.spatial import KDTree

import configargparse
import glob
import natsort
import os
import sys


sys.path.append(".")

parser = configargparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, default="", help="Path to the original data"
)

Crater_SEMANTICS_COLORS = np.array(
    [
        (255, 255, 255),  # background
        (0, 0, 0),  # crater
    ]  # clutter
)

INS_COLORS = np.array(
    [[np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)] for _ in range(1000)]
)


def read_scene_txt(name, data_dir):
    area = name.split(".")[0]
    name = name.split(".")[1]

    pts = np.loadtxt(os.path.join(data_dir + "/" + area, name + ".txt"))
    return pts


def preprocess_crater(data_dir):
    scene_list = []

    area = data_dir + "/*"
    tmp = glob.glob(area + "/*")
    for scene_name in tmp:
        scene_name = scene_name.split("/")[-2] + "." + scene_name.split("/")[-1]
        scene_list.append(scene_name)


    scene_list = natsort.natsorted(scene_list)

    for scene_name in scene_list:
        try:
            area = scene_name.split(".")[0]
            name = scene_name.split(".")[1]
            save_dir = ""
            scene_pth = os.path.join(save_dir, f"{area}_{name}_inst_nostuff.pth")

            os.makedirs(save_dir, exist_ok=True)

            scene_data = read_scene_txt(scene_name, data_dir)

            torch.save(
                (
                    scene_data[:, :3].astype(np.float32),
                    scene_data[:, 3:4].astype(np.float32),
                    scene_data[:, 4:7].astype(np.int32),
                ),
                scene_pth,
            )

        except:
            print("scene_name:"+str(scene_name))

cfg = parser.parse_args()
preprocess_crater(cfg.data_dir)
