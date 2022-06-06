import trimesh
import shapex
import utils
import numpy as np
from copy import deepcopy
import potpourri3d as pp3d
import matplotlib.pyplot as plt
from tqdm import tqdm
# scaling of shape: eigenvector, eigenvalue

if __name__ == "__main__":
    mesh = trimesh.load("data/homer.obj")
    print(mesh.is_watertight)

    desc = shapex.GDM(mesh, mode="eig")
    print(desc.shape)
        
    