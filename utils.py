import trimesh

def colorize_mesh(mesh, colors, cmap="coolwarm"):
    mesh.visual.vertex_colors = trimesh.visual.interpolate(colors, color_map=cmap)
    return mesh