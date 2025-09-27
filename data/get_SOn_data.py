import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../")
from utils import set_seed_everywhere
from manifolds.SOn import Manifold_SOn
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f'{abs_path}/data/figs/SOn', exist_ok=True)


def plot(samples, savename=None):
    trace_list = []
    power_list = [1, 2, 4, 5]
    for i in range(4):
        samples_pow = torch.matrix_power(samples, power_list[i])
        trace = samples_pow.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        trace_list.append(trace)
    statistics = torch.cat(trace_list, dim=1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        ax.hist(statistics[:, i].numpy(), bins=200, alpha=1., density=True)
    plt.savefig(f"{abs_path}/data/figs/SOn/{savename}.png", dpi=300)
    plt.close(fig)

def generate_SOn(well_num):
    set_seed_everywhere(777)
    
    std_list = torch.ones(well_num) * 0.05
    samples_num = 50000
    SOn = Manifold_SOn(10)
    dataset_name = f'SO10_{well_num}w'

    # get center
    basic1 = np.array([[np.cos(np.pi / 3), np.sin(np.pi / 3)],
                       [-np.sin(np.pi / 3), np.cos(np.pi / 3)]])

    mat0 = np.eye(10)
    mat_list = []
    for i in range(5):
        mat0 = mat0.copy()
        mat0[2*i:2*(i+1), 2*i:2*(i+1)] = basic1
        mat_list.append(mat0)

    center_ori = torch.tensor(np.stack(mat_list[:well_num]), dtype=torch.float)
    mat_P = SOn.uniform_sample(well_num).reshape(well_num, 10, 10)
    center = torch.bmm(mat_P.transpose(dim0=1, dim1=2), torch.bmm(center_ori, mat_P))

    # center = torch.tensor(np.stack(mat_list[:well_num]), dtype=torch.float)

    # get data
    center = center.reshape(-1, 10 * 10)
    random_idx = torch.randint(low=0, high=well_num, size=(samples_num,))

    base_point = center[random_idx]
    gaussian_whole = torch.randn(samples_num, 10 * 10) * std_list[random_idx].reshape(-1, 1)
    gaussian_proj = SOn.project_onto_tangent_space(gaussian_whole, base_point=base_point)
    data_gen = SOn.exp(gaussian_proj, base_point=base_point)

    np.save(f"{abs_path}/data/SOn/{dataset_name}.npy", data_gen.detach().cpu().numpy())
    plot(data_gen.reshape(-1, 10, 10), f"hist_{well_num}w")

    return


if __name__ == "__main__":
    
    generate_SOn(well_num=5)


