import trimesh as tm
import lapy as lp
from lapy import ShapeDNA as lp_ShapeDNA
import numpy as np 

def ShapeDNA(mesh, n_eigs, lump=False, aniso=False, aniso_smooth=10, ignore_first=False):
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
def GDM(mesh):
    pass

def R_BiHDM(mesh):
    pass

def BiHDM(mesh):
    pass

# Mesh filtering
