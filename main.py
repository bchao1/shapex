import trimesh 
import shapex

if __name__ == "__main__":
    mesh = trimesh.load("data/homer.obj")
    K_path = "data/R-BiHDM_K.npy"
    shapex.R_BiHDM(mesh)# load_K=K_path)