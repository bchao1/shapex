import trimesh as tm
import lapy as lp
from lapy import ShapeDNA as lp_ShapeDNA
import potpourri3d as pp3d
import numpy as np 
from tqdm import tqdm

# first k or all eigenvalues?
def ShapeDNA(mesh, n_eigs, lump=False, aniso=False, aniso_smooth=10, ignore_first=False, return_dict=False):
    # scale shape by a, eigen value scale by 1/a2)
    lp_mesh = lp.TriaMesh(mesh.vertices, mesh.faces)
    if ignore_first:
        n_eigs += 1
    res = lp_ShapeDNA.compute_shapedna(lp_mesh, n_eigs, lump, aniso, aniso_smooth)
    eig_val = res["Eigenvalues"]
    eig_vec = res["Eigenvectors"]
    if ignore_first:
        eig_val = eig_val[1:]
        eig_vec = eig_vec[:, 1:]
    # normalize?
    eig_vec_norm = np.linalg.norm(eig_vec, axis=0)
    eig_vec /= eig_vec_norm # normalize eigenvectors to unit norm?
    if return_dict:
        return eig_val, eig_vec, res
    else:
        return eig_val, eig_vec

def HKS(mesh, n_eigs=300, M=100, t=None, normalize=False):
    eig_val, eig_vec = ShapeDNA(mesh, n_eigs)
    if t is None:
        # signature depends on eigenvalue! that's why HKS is scale-invariant by this sampling scheme
        # if t is not sampled this way, HKS is not scale invariant
        tmin = 4 * np.log(10) / eig_val[-1]
        tmax = 4 * np.log(10) / eig_val[1]
        t = np.geomspace(tmin, tmax, M)
    hks = (eig_vec**2) @ np.exp(-eig_val[:, None] @ t[None, :])
    # heat trace normalization
    if normalize:
        heat_trace = hks.sum(axis=-1, keepdims=True)
        hks /= heat_trace
    return hks

def SI_HKS(mesh, n_eigs=300, M=100, t=None, normalize=False):
    if t is not None:
        t = np.power(2, np.linspace(1, 25, 24 * 16)) # specified in the paper
    hks = HKS(mesh, n_eigs, M + 1, t, normalize)
    hks_diff = np.log(hks[:, 1:]) - np.log(hks[:, :-1])
    hks_diff_spectrum = np.fft.fft(hks_diff, axis=-1)
    si_hks = np.abs(hks_diff_spectrum)
    return si_hks

def GPS(mesh, n_eigs):
    eig_val, eig_vec = ShapeDNA(mesh, n_eigs, ignore_first=True)
    gps = eig_vec / np.sqrt(eig_val)[None, ]
    return gps

def WKS(mesh, n_eigs=300, M=100, energies=None, normalize=True):
    eig_val, eig_vec = ShapeDNA(mesh, n_eigs)

    delta = np.log(eig_val[-1] / eig_val[1]) / (M + 27) # this is derived from the paper
    sigma = 7 * delta 

    if energies is None:
        emin = np.log(eig_val[1]) + 2 * sigma
        emax = np.log(eig_val[-1]) - 2 * sigma
        energies = np.linspace(emin, emax, M)

    log_eig_val = np.log(np.maximum(np.abs(eig_val), 1e-8)) # numerical stability for 0-eigenvalue
    phi2 = eig_vec**2
    exp = -(energies[:, None] - log_eig_val[None, :])**2 / (2 * (sigma**2))
    wks = phi2 @ exp.T
    if normalize:
        energy_trace = wks.sum(axis=-1, keepdims=True)
        wks /= energy_trace
    return wks

def IWKS(mesh, t):
    pass

def SGWS(mesh, t):
    pass

# Geodesics-based descriptors
def GDM(mesh, mode="svd", dim=50):
    solver = pp3d.MeshHeatMethodDistanceSolver(mesh.vertices, mesh.faces)
    num_vertices = mesh.vertices.shape[0]
    gdm_matrix = np.zeros((num_vertices, num_vertices))
    print("Computing geodesic distance matrix ...")
    for i, v in enumerate(tqdm(range(num_vertices))):
        dist = solver.compute_distance(v)
        gdm_matrix[v] = dist
    # approximations in in geodesic computation leads to non-symmetric geodesic distance matrices.
    # take the average of upper and lower triangle
    gdm_matrix = (gdm_matrix + gdm_matrix.T) * 0.5 
    if mode == "svd":
        print("Computing singular values ...")
        _, desc, _ = np.linalg.svd(gdm_matrix, hermitian=True)
    elif mode == "eig":
        print("Computing Eigenvalues ...")
        # eigenvalue decomposition for symmetric matrix
        desc, _ = np.linalg.eigh(gdm_matrix)
    return desc[:dim]

def R_BiHDM(mesh, M=100, L=30, load_K=None, save_K=None):
    assert M > max(2 * L, 60)

    if load_K is None:
        A = mesh.area
        eig_val, eig_vec, ev_dict = ShapeDNA(mesh, M + 1, return_dict=True)
        D = ev_dict["mass"] # mass matrix

        K = np.zeros((M + 1, M + 1)) # projection matrix
        K[0, 0] = (2 / eig_val[1:]**2).sum()
        K[np.arange(1, M+1), np.arange(1, M+1)] = -2 / (eig_val[1:]**2)

        eigs = ((eig_vec[:, 1:] / eig_val[None, 1:])**2).sum(axis=-1)
        a_list = []
        for i in tqdm(range(1, M + 1)):
            a = ((eig_vec[:, i][:, None] @ eigs[None, :]) * D).sum()
            a_list.append(a)
        a_list = np.sqrt(A) * np.array(a_list)
        K[0, 1:] = a_list
        K[1:, 0] = a_list
        if save_K is not None:
            np.save(save_K, K)
    else:
        K = np.load(load_K)
    k_eig, _ = np.linalg.eigh(K)
    k_eig = -(k_eig / k_eig[-1])[:L]
    return k_eig
# Mesh filtering
