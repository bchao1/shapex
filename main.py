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
    _, ev = shapex.ShapeDNA(mesh, 100)
    mesh_list = []
    for i in range(10):
        colored_mesh = utils.colorize_mesh(deepcopy(mesh), ev[:, i])
        colored_mesh.apply_translation([0, 0, -0.5*i])
        mesh_list.append(colored_mesh)
    trimesh.Scene(mesh_list).show()
        
    