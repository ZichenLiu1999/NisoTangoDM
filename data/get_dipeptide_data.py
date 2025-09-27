import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from manifolds.MD import Manifold_MD
from utils import set_seed_everywhere, refine_dataset_md
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f'{abs_path}/data/figs/dipeptide', exist_ok=True)


def plot_phi_psi_hist(phi, psi, savefig=""):
    fig = plt.figure(figsize=(10, 5))
    print(phi.shape[0])
    bins = int(phi.shape[0] / 100)

    ax = plt.subplot(1, 2, 1)
    ax.hist(phi.reshape(-1).numpy()/ torch.pi * 180, bins=bins, alpha=1.0, density=True)
    ax.set_title("phi")

    ax = plt.subplot(1, 2, 2)
    ax.hist(psi.reshape(-1).numpy()/ torch.pi * 180, bins=bins, alpha=1.0, density=True)
    ax.set_title("psi")
    plt.savefig(f'{abs_path}/data/figs/dipeptide/dipeptide_Hist_{savefig}.png', dpi=300)
    #plt.show()
    plt.close(fig)


def generate_dipeptide_data_long():
    set_seed_everywhere(777)
    xvec = torch.tensor(np.load(f'{abs_path}/data/dipeptide/dipeptide_long.npy')).float()

    manifold = Manifold_MD()

    phi, psi = manifold.angle_phi(xvec), manifold.angle_psi(xvec)
    plot_phi_psi_hist(phi, psi, savefig="ori_long")

    sdf = Manifold_MD()
    # 1e-4, 1e-2
    xvec_refined_all, xvec_refined = refine_dataset_md(sdf, xvec.reshape(-1, 30),
                                                    tol=1e-5,
                                                    max_iter_n=10000,
                                                    step_size=1e-3,
                                                    rk4=True,
                                                    keep_quiet=False)
    np.save(f'{abs_path}/data/dipeptide/dipeptide_long_refined.npy', xvec_refined.numpy())
    # np.save(f'{abs_path}/data/dipeptide/dipeptide_ref.npy', xvec_refined[0].numpy())

    print(sdf.constrain_fn(xvec_refined).abs().max())
    phi_refined, psi_refined = manifold.angle_phi(xvec_refined), manifold.angle_psi(xvec_refined)
    plot_phi_psi_hist(phi_refined, psi_refined, savefig="refined_long")

    return


if __name__ == "__main__":
    
    generate_dipeptide_data_long()




