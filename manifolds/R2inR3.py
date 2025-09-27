import torch


class Manifold_R2inR3:
    """
    z=0
    """
    def __init__(self):
        self.out_dim = 3

    def constrain_fn(self, samples):
        return samples[:, 2:]

    def constrain_grad_fn(self, samples):
        vec = torch.tensor([[0., 0., 1.]]).to(samples)
        return samples * 0. + vec

    def project_onto_tangent_space(self, y, base_point):
        vec = torch.tensor([[1., 1., 0.]]).to(y)
        return y * vec

    def project_onto_manifold(self, y):
        vec = torch.tensor([[1., 1., 0.]]).to(y)
        return y * vec

    def project_onto_manifold_with_base(self, y, base_point, **kwargs):
        return self.project_onto_manifold(y), torch.ones(y.shape[0], dtype=torch.bool).to(y)

    def project_onto_manifold_SDE(self, y, base_point):
        return self.project_onto_manifold(base_point + y)
