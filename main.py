import trimesh
import shapectra
import utils
import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt

# scaling of shape: eigenvector, eigenvalue

if __name__ == "__main__":
    mesh = trimesh.load("data/homer.obj")
    print(mesh.is_watertight)
    #wks = shapectra.WKS(mesh)
    #eig_val, eig_vec = shapectra.ShapeDNA(mesh, 300)
    #colored_mesh = utils.colorize_mesh(mesh, eig_vec[:, 1])
    #trimesh.Scene(colored_mesh).show()
    #hks = shapectra.HKS(mesh, np.linspace(np.log(4*np.log(10)/eig_val[300]), np.log(4*np.log(10)/eig_val[2]), 100), 301)
    #for i in range(wks.shape[0]):
    #    plt.plot(wks[i])
    #plt.show()
    #exit()
    t = np.power(2, np.linspace(0, 10, 100))
    hks = shapectra.SI_HKS(mesh)

    scaled_mesh = deepcopy(mesh)
    scaled_mesh.apply_scale(11)

    scaled_hks = shapectra.SI_HKS(scaled_mesh)
    #colored_mesh = utils.colorize_mesh(deepcopy(mesh), wks[:, 50])
    #trimesh.Scene(colored_mesh).show()
    plt.plot(hks[0], scaled_hks[0])
    plt.show()
    exit()
    mesh_list = []
    for i in range(10):
        colored_mesh = utils.colorize_mesh(deepcopy(mesh), hks[:, int(10*i)])
        colored_mesh.apply_translation([0, 0, -0.5*i])
        colored_scaled_mesh = utils.colorize_mesh(deepcopy(scaled_mesh), scaled_hks[:, int(10*i)])
        colored_scaled_mesh.apply_translation([0, 2, -0.5*i])
        mesh_list.append(colored_mesh)
        mesh_list.append(colored_scaled_mesh)
    trimesh.Scene(mesh_list).show()