import os
import igl
import scipy as sp
import plotly.offline as offline
import numpy as np
import torch
import plotly.graph_objs as go
import trimesh
import sys
sys.path.append("../")
from utils import set_seed_everywhere, get_scene_dict

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f'{abs_path}/data/figs/mesh', exist_ok=True)


def plot_histogram_on_surface(obj, mesh, samples, colorscale=None):
    verts = mesh.vertices
    I, J, K = mesh.faces.transpose()

    closest_points, _, closest_faces = trimesh.proximity.closest_point(mesh, samples)
    unique_faces, counts = np.unique(closest_faces, return_counts=True)
    probs = np.zeros(len(mesh.faces))
    probs[unique_faces] = counts / len(samples)
    densities = probs / mesh.area_faces
    densities[np.isnan(densities)] = 0

    cmin, cmax = -0.1, np.percentile(densities, 95) if colorscale is None else colorscale
    traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                        i=I, j=J, k=K, name='Samples_hist',
                        opacity=1.0, intensity=densities, intensitymode="cell", colorscale="Viridis",
                        cmin=cmin, cmax=cmax)]
    layout = go.Layout(title=f'Histgram of {samples.shape[0]} scatters', scene=get_scene_dict(obj), width=1400, height=1400, showlegend=True)
    fig = go.Figure(data=traces, layout=layout)
    return fig


def plot_point_cloud(obj, mesh, samples):
    verts = mesh.vertices
    I, J, K = mesh.faces.transpose()
    trace = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                       i=I, j=J, k=K, alphahull=5, opacity=0.4, color='cyan'),
             go.Scatter3d(x=samples[:, 0], y=samples[:, 1], z=samples[:, 2], mode='markers', marker=dict(size=3))]
    fig = go.Figure(data=trace)
    fig.update_layout(title=f'{samples.shape[0]} scatters',
                      scene=get_scene_dict(obj), width=1400, height=1400, showlegend=False)
    return fig


def sample_simplex_uniform(K, shape=(), dtype=torch.float32, device="cpu"):
    x = torch.sort(torch.rand(shape + (K,), dtype=dtype, device=device))[0]
    x = torch.cat([torch.zeros(*shape, 1, dtype=dtype, device=device),x,
        torch.ones(*shape, 1, dtype=dtype, device=device)], dim=-1)
    diffs = x[..., 1:] - x[..., :-1]
    return diffs


def create_mesh(obj):
    v, f = igl.read_triangle_mesh(f"{abs_path}/data/{obj}/{obj}_simp.obj")
    if obj == "bunny": v = v / 250.

    mesh_simple = trimesh.Trimesh(vertices=v, faces=f)
    mesh_simple.export(f"{abs_path}/data/{obj}/{obj}_mesh_simple.ply", 'ply')

    v_np1, f_np1 = igl.upsample(mesh_simple.vertices, mesh_simple.faces, 1)
    mesh_simple1 = trimesh.Trimesh(vertices=v_np1, faces=f_np1)
    mesh_simple1.export(f"{abs_path}/data/{obj}/{obj}_mesh_simple1.ply", 'ply')
    
    v_np2, f_np2 = igl.upsample(mesh_simple.vertices, mesh_simple.faces, 2)
    mesh_simple2 = trimesh.Trimesh(vertices=v_np2, faces=f_np2)
    mesh_simple2.export(f"{abs_path}/data/{obj}/{obj}_mesh_simple2.ply", 'ply')
             
    v_np3, f_np3 = igl.upsample(mesh_simple.vertices, mesh_simple.faces, 3)
    mesh_complex = trimesh.Trimesh(vertices=v_np3, faces=f_np3)
    mesh_complex.export(f"{abs_path}/data/{obj}/{obj}_mesh_complex.ply", 'ply')


def cal_eigen_val_fn(obj, idx):
    numeigs = idx + 20

    mesh_complex = trimesh.load(f"{abs_path}/data/{obj}/{obj}_mesh_complex.ply", 'ply')
    v_np, f_np = mesh_complex.vertices, mesh_complex.faces
    print("#vertices: ", v_np.shape, "#faces: ", f_np.shape)

    # preprocess_eigenfunctions
    M = igl.massmatrix(v_np, f_np, igl.MASSMATRIX_TYPE_VORONOI)
    L = -igl.cotmatrix(v_np, f_np)
    eigvals, eigfns = sp.sparse.linalg.eigsh(L, numeigs + 1, M, sigma=0, which="LM", maxiter=100000)
    
    print(eigvals, eigfns[:, 0])
    print(f"largest eigval: {eigvals.max().item()}, smallest eigval: {eigvals.min().item()}")
    np.save(f"{abs_path}/data/{obj}/{obj}_eigfns.npy", eigfns)
    np.save(f"{abs_path}/data/{obj}/{obj}_eigvals.npy", eigvals)


def sample_simplex_uniform(K, shape=(), dtype=torch.float32, device="cpu"):
    x = torch.sort(torch.rand(shape + (K,), dtype=dtype, device=device))[0]
    x = torch.cat([torch.zeros(*shape, 1, dtype=dtype, device=device),x,
        torch.ones(*shape, 1, dtype=dtype, device=device)], dim=-1)
    diffs = x[..., 1:] - x[..., :-1]
    return diffs


def gen_sample_from_eigen(obj, idx_list, weight_list, name="mix"):
    mesh_complex = trimesh.load(f"{abs_path}/data/{obj}/{obj}_mesh_complex.ply", 'ply')
    v_np, f_np = mesh_complex.vertices, mesh_complex.faces
    v, f = torch.tensor(v_np), torch.tensor(f_np)
    print("#vertices: ", v.shape[0], "#faces: ", f.shape[0])

    eigfns = np.load(f"{abs_path}/data/{obj}/{obj}_eigfns.npy")
    eigvals = np.load(f"{abs_path}/data/{obj}/{obj}_eigvals.npy")

    eigfns = torch.tensor(eigfns).clamp(min=0.0000)
    eigfns[:, 0] += 1.0
    print(f"largest eigval: {eigvals.max().item()}, smallest eigval: {eigvals.min().item()}")
    
    # start sample
    set_seed_everywhere(12345)
    num_samples = 60000
    weights = torch.tensor(weight_list, dtype=torch.float)
    num_per_idx = torch.multinomial(weights, num_samples, replacement=True)

    samples_list = []
    for i, idx in enumerate(idx_list):
        nsamples_temp = (num_per_idx == i).sum().item()

        vals = eigfns[:, idx]
        vals = torch.mean(vals[f], dim=1)
        vals = vals * torch.tensor(igl.doublearea(v_np, f_np)).reshape(-1) / 2
    
        f_idx = torch.multinomial(vals, nsamples_temp, replacement=True)
        barycoords = sample_simplex_uniform(2, (nsamples_temp,))
        samples_temp = torch.sum(v[f[f_idx]] * barycoords[..., None], axis=1)
        samples_list.append(samples_temp.float())
    samples = torch.cat(samples_list, dim=0)
    
    samples = samples[torch.randperm(samples.shape[0], generator=torch.Generator().manual_seed(12345))].numpy()
    np.save(f"{abs_path}/data/{obj}/{obj}_{name}.npy", samples)

    # plot
    fig = plot_point_cloud(obj, mesh_complex, samples)
    offline.plot(fig, filename=f'{abs_path}/data/figs/mesh/{obj}_{name}_point.html', auto_open=False)
    fig.write_image(f"{abs_path}/data/figs/mesh/{obj}_{name}_point.png")
    
    mesh_simple1 = trimesh.load(f"{abs_path}/data/{obj}/{obj}_mesh_simple1.ply", 'ply')
    fig = plot_histogram_on_surface(obj, mesh_simple1, samples)
    offline.plot(fig, filename=f'{abs_path}/data/figs/mesh/{obj}_{name}_hist.html', auto_open=False)
    fig.write_image(f"{abs_path}/data/figs/mesh/{obj}_{name}_hist.png")

    return


def gen_sample_trancate(obj, idx_list, weight_list, name="mix_tanc"):
    mesh_complex = trimesh.load(f"{abs_path}/data/{obj}/{obj}_mesh_complex.ply", 'ply')
    v_np, f_np = mesh_complex.vertices, mesh_complex.faces
    v, f = torch.tensor(v_np), torch.tensor(f_np)
    print("#vertices: ", v.shape[0], "#faces: ", f.shape[0])
    
    eigfns = np.load(f"{abs_path}/data/{obj}/{obj}_eigfns.npy")
    eigvals = np.load(f"{abs_path}/data/{obj}/{obj}_eigvals.npy")
    eigfns = torch.tensor(eigfns).clamp(min=0.0000)
    eigfns[:, 0] += 1.0
    print(f"largest eigval: {eigvals.max().item()}, smallest eigval: {eigvals.min().item()}")
    
    # start sample
    set_seed_everywhere(12345)
    num_samples = 90000
    weights = torch.tensor(weight_list, dtype=torch.float)
    num_per_idx = torch.multinomial(weights, num_samples, replacement=True)
    
    samples_list = []
    for i, idx in enumerate(idx_list):
        nsamples_temp = (num_per_idx == i).sum().item()
        
        vals = eigfns[:, idx]
        vals = torch.mean(vals[f], dim=1)
        vals = vals * torch.tensor(igl.doublearea(v_np, f_np)).reshape(-1) / 2
        
        f_idx = torch.multinomial(vals, nsamples_temp, replacement=True)
        barycoords = sample_simplex_uniform(2, (nsamples_temp,))
        samples_temp = torch.sum(v[f[f_idx]] * barycoords[..., None], axis=1)
        samples_list.append(samples_temp.float())
    samples = torch.cat(samples_list, dim=0)
    
    condition = (samples[:, 1] <= 0.7) & (samples[:, 2] <= 0.92)
    samples_f = samples[condition]
    samples_f = samples_f[torch.randperm(samples_f.shape[0], generator=torch.Generator().manual_seed(12345))][:60000].numpy()
    
    np.save(f"{abs_path}/data/{obj}/{obj}_{name}.npy", samples_f)
    
    # plot
    fig = plot_point_cloud(obj, mesh_complex, samples_f)
    offline.plot(fig, filename=f'{abs_path}/data/figs/mesh/{obj}_{name}_point.html', auto_open=False)
    fig.write_image(f"{abs_path}/data/figs/mesh/{obj}_{name}_point.png")
    
    mesh_simple1 = trimesh.load(f"{abs_path}/data/{obj}/{obj}_mesh_simple1.ply", 'ply')
    fig = plot_histogram_on_surface(obj, mesh_simple1, samples_f)
    offline.plot(fig, filename=f'{abs_path}/data/figs/mesh/{obj}_{name}_hist.html', auto_open=False)
    fig.write_image(f"{abs_path}/data/figs/mesh/{obj}_{name}_hist.png")
    
    return



if __name__ == "__main__":
    create_mesh("bunny")
    cal_eigen_val_fn("bunny", idx=1000)
    
    create_mesh("spot")
    cal_eigen_val_fn("spot", idx=1000)
    
    gen_sample_trancate(obj="spot", idx_list=[0, 500, 1000], weight_list=[1, 1, 1], name="mixfil")
    
    obj_list = ["bunny", "spot"]
    for obj in obj_list:
    
        idx_list = [0, 500, 1000]
        weight_list = [1, 1, 1]
        gen_sample_from_eigen(obj, idx_list, weight_list, name="mix")

