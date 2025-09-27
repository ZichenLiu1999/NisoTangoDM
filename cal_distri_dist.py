import torch
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import ot as pot
from functools import partial
import math
from utils import get_RMSD
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


def cal_wasser_dist(x0: torch.Tensor,x1: torch.tensor, power=1, dist_fn=None, method="exact"):
    assert power == 1 or power == 2
    if method == "exact":
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=0.05)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    M = dist_fn(x0, x1) if dist_fn is not None else torch.cdist(x0, x1)

    if power == 2: M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2: ret = math.sqrt(ret)
    return ret


def cal_sliced_wasser_dist(p: torch.Tensor, q: torch.Tensor):
    assert p.shape[1] == q.shape[1]
    n_features = p.shape[1]
    num_projections = 2000

    directions = torch.randn(num_projections, n_features)
    norms = torch.norm(directions, dim=1, keepdim=True)
    norms = torch.where(norms < 1e-8, torch.tensor(1e-8), norms)
    directions = directions / norms
    # directions = directions / torch.norm(directions, dim=1, keepdim=True)

    p_projections = torch.matmul(p, directions.T)  # shape: [p_n_samples, num_projections]
    q_projections = torch.matmul(q, directions.T)  # shape: [q_n_samples, num_projections]
    # print(directions.shape, p_projections.shape, q_projections.shape)
    wasserstein_distances = []
    for i in range(num_projections):
        p_sorted = torch.sort(p_projections[:, i])[0]
        q_sorted = torch.sort(q_projections[:, i])[0]
        wasserstein_distance = torch.mean(torch.abs(p_sorted - q_sorted))
        wasserstein_distances.append(wasserstein_distance)
    return torch.mean(torch.tensor(wasserstein_distances)).numpy()


def cal_mmd_dist(a, b, dist_fn=None):
    def gaussian_kernel(x, y, sigma):
        if dist_fn is not None:
            M = dist_fn(x, y) ** 2
            return np.exp(- M / (2 * (sigma ** 2)))
        else:
            return rbf_kernel(x, y, sigma)

    def mmd_distance(x, y, gamma):
        xx = gaussian_kernel(x, x, gamma)
        xy = gaussian_kernel(x, y, gamma)
        yy = gaussian_kernel(y, y, gamma)
        return xx.mean() + yy.mean() - 2 * xy.mean()

    a, b = a.numpy(), b.numpy()
    gammas = [0.01, 0.1, 1, 10, 100]
    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd
    return np.mean(list(map(lambda gamma0: safe_mmd(a, b, gamma0), gammas)))


def cal_JS_dist(x0, x1):
    js_distance = jensenshannon(x0, x1)
    return js_distance

def cal_entropy_dist(x0, x1):
    return entropy(x1, x0)

def cal_TVD_dist(x0, x1):
    tv_distance = 0.5 * np.sum(np.abs(x0 - x1))
    return tv_distance

def distance_Kabsch(x0, x1):
    if not isinstance(x0, torch.Tensor):
        x0, x1 = torch.tensor(x0).reshape(-1, 10, 3), torch.tensor(x1).reshape(-1, 10, 3)
    else:
        x0, x1 = x0.reshape(-1, 10, 3), x1.reshape(-1, 10, 3)
    
    dist_M = []
    for i in range(x0.shape[0]):
        dist_M.append(get_RMSD(x1, x0[i]))
    dist_M1 = torch.stack(dist_M, dim=0)
    return dist_M1


if __name__ == "__main__":
    a = torch.randn(100, 2)
    b = torch.randn(200, 2)
    cal_mmd_dist(a, b)
    cal_wasser_dist(a, b)
