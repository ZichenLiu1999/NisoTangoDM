import torch
import numpy as np
import trimesh
from torch.func import vmap


def project_edge(p, a, b):
    x = p - a
    v = b - a
    r = torch.sum(x * v, dim=-1, keepdim=True) / torch.sum(v * v, dim=-1, keepdim=True)
    r = r.clamp(max=1.0, min=0.0)
    projx = v * r
    return projx + a


def face_normal(a, b, c):
    """Computes face normal based on three vertices. Ordering matters.

    Inputs:
        a, b, c: (N, 3)
    """
    u = b - a
    v = c - a
    n = torch.linalg.cross(u, v)
    n = n / torch.linalg.norm(n, dim=-1, keepdim=True)
    return n


def closest_point(p, v, f):
    """Returns the point on the mesh closest to the query point p.
    Algorithm follows https://www.youtube.com/watch?v=9MPr_XcLQuw&t=204s.
    Inputs:
        p : (#query, 3)
        v : (#vertices, 3)
        f : (#faces, 3)
    Return:
        A projected tensor of size (#query, 3) and an index (#query,) indicating the closest triangle.
    """
    
    orig_p = p

    nq = p.shape[0]
    nf = f.shape[0]

    vs = v[f]
    a, b, c = vs[:, 0], vs[:, 1], vs[:, 2]

    # calculate normal of each triangle
    n = face_normal(a, b, c)

    n = n.reshape(1, nf, 3)
    p = p.reshape(nq, 1, 3)

    a = a.reshape(1, nf, 3)
    b = b.reshape(1, nf, 3)
    c = c.reshape(1, nf, 3)

    # project onto the plane of each triangle
    p = p + (n * (a - p)).sum(-1, keepdim=True) * n

    # if barycenter coordinate is negative,
    # then point is outside of the edge on the opposite side of the vertex.
    bc = barycenter_coordinates(p, a, b, c)

    # for each outside edge, project point onto edge.
    p = torch.where((bc[..., 0] < 0)[..., None], project_edge(p, b, c), p)
    p = torch.where((bc[..., 1] < 0)[..., None], project_edge(p, c, a), p)
    p = torch.where((bc[..., 2] < 0)[..., None], project_edge(p, a, b), p)

    # compute distance to all points and take the closest one
    idx = torch.argmin(torch.linalg.norm(orig_p[:, None] - p, dim=-1), dim=-1)
    p_idx = vmap(lambda p_, idx_: torch.index_select(p_, 0, idx_))(p, idx.reshape(-1, 1)).reshape(nq, 3)
    return p_idx, idx


def barycenter_coordinates(p, a, b, c):
    """Assumes inputs are (N, D).
    Follows https://ceng2.ktu.edu.tr/~cakir/files/grafikler/Texture_Mapping.pdf
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = torch.sum(v0 * v0, dim=-1)
    d01 = torch.sum(v0 * v1, dim=-1)
    d11 = torch.sum(v1 * v1, dim=-1)
    d20 = torch.sum(v2 * v0, dim=-1)
    d21 = torch.sum(v2 * v1, dim=-1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return torch.stack([u, v, w], dim=-1)


class Manifold_Mesh:
    def __init__(self, mesh_path, device="cpu"):
        self.out_dim = 3
        self.inner_dim = 2
        self.mesh = trimesh.load(mesh_path)
        v_np, f_np = self.mesh.vertices, self.mesh.faces
        self.device = device
        self.v, self.f = torch.tensor(v_np, dtype=torch.float32).to(device), torch.tensor(f_np).to(self.device)
        self.vs = self.v[self.f]
        self.face_normal = face_normal(self.vs[:, 0], self.vs[:, 1], self.vs[:, 2])
    
    def closest_point(self, p, return_p_idx=True):
        orig_p = p
        
        nq, nf = p.shape[0], self.f.shape[0]
        a, b, c = self.vs[:, 0].reshape(1, nf, 3), self.vs[:, 1].reshape(1, nf, 3), self.vs[:, 2].reshape(1, nf, 3)
        n = self.face_normal.reshape(1, nf, 3)
        p = p.reshape(nq, 1, 3)
        # print(a.device, p.device, n.device)
        p = p + (n * (a - p)).sum(-1, keepdim=True) * n
        bc = barycenter_coordinates(p, a, b, c)
        
        p = torch.where((bc[..., 0] < 0)[..., None], project_edge(p, b, c), p)
        p = torch.where((bc[..., 1] < 0)[..., None], project_edge(p, c, a), p)
        p = torch.where((bc[..., 2] < 0)[..., None], project_edge(p, a, b), p)
        
        idx = torch.argmin(torch.linalg.norm(orig_p[:, None] - p, dim=-1), dim=-1)
        if return_p_idx:
            p_idx = vmap(lambda p_, idx_: torch.index_select(p_, 0, idx_))(p, idx.reshape(-1, 1)).reshape(nq, 3)
            return p_idx, idx
        else:
            return idx
    
    @torch.no_grad()
    def constrain_fn(self, samples):
        y = self.project_onto_manifold(samples)
        return torch.norm(samples-y, dim=1)

    @torch.no_grad()
    def constrain_grad_fn(self, samples):
        # determine which face the point is on
        _, f_idx = self.closest_point(samples)
        vs = self.v[self.f[f_idx]]
        return face_normal(a=vs[:, 0], b=vs[:, 1], c=vs[:, 2])

    @torch.no_grad()
    def project_onto_tangent_space(self, y, base_point):
        n = self.constrain_grad_fn(base_point)
        return y - (n * y).sum(-1, keepdim=True) * n

    @torch.no_grad()
    def project_onto_manifold(self, y):
        y, _ = self.closest_point(y)
        return y

    def project_onto_manifold_with_base(self, y, base_point, **kwargs):
        return self.project_onto_manifold(base_point + y)

    def project_onto_manifold_SDE(self, y, base_point):
        return self.project_onto_manifold(base_point + y)


if __name__ == "__main__":
    pass