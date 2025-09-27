import torch
import logging


def dihedral(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x).float()
        raise NotImplementedError

    r12 = x[:, 1, :] - x[:, 0, :]
    r23 = x[:, 2, :] - x[:, 1, :]
    r34 = x[:, 3, :] - x[:, 2, :]
    n1 = torch.linalg.cross(r12, r23)
    n2 = torch.linalg.cross(r23, r34)
    cos_phi = (n1*n2).sum(dim=1, keepdim=True)
    sin_phi = (n1*r34).sum(dim=1, keepdim=True) * torch.norm(r23, dim=1, keepdim=True)
    return torch.atan2(sin_phi, cos_phi) # / torch.pi * 180


class Manifold_MD:
    def __init__(self):
        self.natom = 10
        self.out_dim = 3 * self.natom
        self.constrain_grad_fn_is_norm = False

    def angle_phi(self, x):
        x = x.reshape(x.shape[0], self.natom, 3)
        atom_indices = [1, 3, 4, 6]
        return dihedral(x[:, atom_indices,:])

    def angle_psi(self, x):
        x = x.reshape(-1, self.natom, 3)
        atom_indices = [3, 4, 6, 8]
        return dihedral(x[:, atom_indices,:])

    def constrain_fn(self, samples):
        return self.angle_phi(samples) - (- 70 / 180 * torch.pi)

    @torch.enable_grad()
    def constrain_grad_fn(self, samples):
        # samples = samples.reshape(-1, self.natom*3)
        samples.requires_grad_(True)
        gradients = torch.autograd.grad(
            outputs=self.constrain_fn(samples).sum(),
            inputs=samples,
            create_graph=True,
            retain_graph=True)[0]
        return gradients.detach()

    def project_onto_tangent_space(self, y, base_point):
        norm_vec = self.constrain_grad_fn(base_point)
        coeff = torch.sum(y * norm_vec, dim=1) / (norm_vec**2).sum(dim=1)
        return y - coeff.reshape(-1, 1) * norm_vec
    
    def project_onto_manifold_RSDE(self, y, threshold=1e-5, n_iters=10):

        @torch.enable_grad()
        def constrain_val_Jacobi(mu, active_idx):
            mu_tmp = mu[active_idx].clone()
            mu_tmp.requires_grad_(True)
            x = mu_tmp[:, :self.out_dim]

            grad_x = torch.autograd.grad(outputs=self.constrain_fn(x).sum(),inputs=x, retain_graph=True)[0]
            tmp1 = x - y[active_idx] + mu_tmp[:, -1:] * grad_x
            tmp2 = self.constrain_fn(x)
            value = torch.cat((tmp1, tmp2), dim=1)

            Jacobian = torch.zeros(value.shape[0], value.shape[1], mu.shape[1]).to(y.device)
            for j in range(value.shape[1]):
                Jacobian[:, j, :] = torch.autograd.grad(outputs=value[:, j].sum(), inputs=mu_tmp, retain_graph=True)[0]
            return  value.detach(), Jacobian.detach()
        
        mu = torch.cat((y.detach().clone(), torch.zeros(y.shape[0], 1).to(y.device)), dim=1)
        active_idx = torch.arange(0, y.shape[0], dtype=torch.int64).to(y.device)

        for i in range(n_iters):
            value, Jacobian_mat = constrain_val_Jacobi(mu, active_idx)
            bad_idx = (value.abs().max(dim=1)[0] >= threshold)
            active_idx = active_idx[bad_idx]
            if bad_idx.sum() == 0:
                break
            mu[active_idx] = mu[active_idx] - torch.linalg.solve(Jacobian_mat[bad_idx], value[bad_idx].unsqueeze(-1)).squeeze(-1)
        logging.info(f'Total steps of the projection: {i}!')

        mask = torch.ones(mu.shape[0], dtype=bool)
        mask[active_idx] = False
        return mu[mask, :self.out_dim].detach()
    
    def project_onto_manifold(self, y):
        return self.project_onto_manifold_RSDE(y)

    @torch.no_grad()
    def project_onto_manifold_with_base(self, y, base_point, threshold=1e-5, n_iters=10, keep_quiet=False, **kwargs):
        """
            Here, y is a tangent vector. Find mu such that xi(y + base_point + mu grad xi(base_point)) = 0.
        """

        grad_vec = self.constrain_grad_fn(base_point)
        mu = torch.zeros(y.shape[0], 1).to(y.device)
        active_idx = torch.arange(0, y.shape[0], dtype=torch.int64).to(y.device)

        for i in range(n_iters):
            temp = y[active_idx] + base_point[active_idx] - grad_vec[active_idx] * mu[active_idx]

            value = self.constrain_fn(temp)

            bad_idx = (value.abs() > threshold).squeeze(dim=1)
            active_idx = active_idx[bad_idx]
            if bad_idx.sum() == 0:
                break

            mu_grad = - (self.constrain_grad_fn(temp[bad_idx]) * grad_vec[active_idx]).sum(dim=1, keepdim=True)
            mu[active_idx] = mu[active_idx] - value[bad_idx] / mu_grad
        projected_pt = y + base_point - grad_vec * mu
        value = self.constrain_fn(projected_pt).abs().squeeze()

        non_converged_flag = (value > threshold) | (~torch.isfinite(value))
        non_converged_num = non_converged_flag.sum() 

        if (not keep_quiet) and (non_converged_flag.sum() > 0):
            logging.info(f'total steps: {i}, max_error: {value.max():.1e}, {non_converged_num} states not converged!')
        return projected_pt[~non_converged_flag, :].detach(), torch.logical_not(non_converged_flag).to(y)

    def project_onto_manifold_SDE(self, y, base_point):
        return self.project_onto_manifold_with_base(y, base_point)[0]


if __name__ == "__main__":
    pass





