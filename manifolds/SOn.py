import torch
import numpy as np


def get_matrix_from_euler_angle(theta1, theta2, theta3):
    c1, s1 = torch.cos(theta1), torch.sin(theta1)
    c2, s2 = torch.cos(theta2), torch.sin(theta2)
    c3, s3 = torch.cos(theta3), torch.sin(theta3)
    matrix = torch.tensor([[c2 * c3, -c2 * s3, s2],
                       [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                       [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2]])
    return matrix


def get_euler_angle_from_matrix(matrix):
    """
    Tait–Bryan angles: XYZ
    theta1: [-pi, pi]
    theta2: [-pi/2, pi/2]
    theta3: [-pi, pi]
    """

    r11, r12, r13 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    r21, r22, r23 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    r31, r32, r33 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]

    if isinstance(matrix, torch.Tensor):
        theta1 = torch.arctan2(-r23, r33)
        theta2 = torch.arcsin(r13)
        theta3 = torch.arctan2(-r12, r11)
    else:
        theta1 = np.arctan2(-r23, r33)
        theta2 = np.arcsin(r13)
        theta3 = np.arctan2(-r12, r11)
    return [theta1, theta2, theta3]


class Manifold_SOn:
    def __init__(self, dim):
        self.mat_dim = dim
        self.out_dim = self.mat_dim * self.mat_dim
        self.inner_dim = int(self.mat_dim * (self.mat_dim - 1) / 2)

        self.triu_indices = torch.triu_indices(self.mat_dim, self.mat_dim)
        self.constraint_num = self.triu_indices.shape[1]
        self.grad_coeff_matrix_normalized = self.get_grad_coeff_matrix(normalized=True)
        self.grad_coeff_matrix = self.get_grad_coeff_matrix(normalized=False)

    def get_grad_coeff_matrix(self, normalized):
        mat = torch.zeros(self.constraint_num, self.mat_dim, self.mat_dim)

        alpha = np.sqrt(0.5)
        for idx in range(self.constraint_num):
            i, j = self.triu_indices[:, idx]
            if normalized:
                if i == j:
                    mat[idx, i, j] += 1.
                else:
                    mat[idx, i, j] += alpha
                    mat[idx, j, i] += alpha
            else:
                mat[idx, i, j] += 1.0
                mat[idx, j, i] += 1.0
        return mat

    def constrain_fn(self, samples):
        samples = samples.reshape(-1, self.mat_dim, self.mat_dim)
        temp = torch.bmm(samples, torch.transpose(samples, dim0=1, dim1=2)) - torch.eye(self.mat_dim, self.mat_dim).to(samples)
        return temp[:, self.triu_indices[0, :], self.triu_indices[1, :]]

    def constrain_grad_fn(self, samples, normalized=True):
        samples = samples.reshape(-1, self.mat_dim, self.mat_dim).unsqueeze(1)
        if normalized:
            temp = torch.matmul(self.grad_coeff_matrix_normalized.to(samples), samples).flatten(start_dim=-2)
        else:
            temp = torch.matmul(self.grad_coeff_matrix.to(samples), samples).flatten(start_dim=-2)
        return temp

    def project_onto_tangent_space(self, y, base_point):
        """
        P_X(U) = (U-X U^T X)/2
        """
        y = y.reshape(-1, self.mat_dim, self.mat_dim)
        base_point = base_point.reshape(-1, self.mat_dim, self.mat_dim)
        out = (y - torch.bmm(base_point, torch.bmm(torch.transpose(y, dim0=1, dim1=2), base_point))) * 0.5
        return out.reshape(-1, self.out_dim)

    def project_onto_manifold(self, y):
        """
        P(X)=U V^T for X = U D V^T
        """
        y = y.reshape(-1, self.mat_dim, self.mat_dim)
        U, S, VT = torch.linalg.svd(y)
        # torch.bmm(U, torch.bmm(torch.diag_embed(S), Vh)) - y should be skew-symmetric
        return torch.bmm(U, VT).reshape(-1, self.out_dim)

    @torch.no_grad()
    def project_onto_manifold_with_base(self, y, base_point, threshold=1e-6, n_iters=10, **kwargs):
        mu = torch.zeros((y.shape[0], self.out_dim-self.inner_dim)).to(y)
        for i in range(n_iters):
            temp = y + base_point - torch.einsum('ijk,ij->ik', self.constrain_grad_fn(base_point, normalized=False), mu)
            value = self.constrain_fn(temp)
            # if (value.abs().max() < threshold) and (i > 0):
            if value.abs().max() < threshold:
                if i > 5: print(f'Newton steps > 5.')
                break
            mu_grad = - torch.bmm(self.constrain_grad_fn(temp, normalized=False),
                                  self.constrain_grad_fn(base_point, normalized=False).transpose(1, 2))
            mu = mu - torch.linalg.solve(mu_grad, value)
        projected_pt = y + base_point - torch.einsum('ijk,ij->ik', self.constrain_grad_fn(base_point, normalized=False), mu)
        return projected_pt.detach(), torch.ones(y.shape[0], dtype=torch.bool).to(y)

    def uniform_sample(self, sample_num):
        """
        Ensure the matrices are in the correct component
        """
        sample = torch.tensor([])
        while sample.shape[0] < sample_num:
            Z = torch.randn(sample_num, self.mat_dim, self.mat_dim)
            idx1 = torch.where(torch.linalg.det(Z).abs() > 1e-4)[0]
            Q, R = torch.linalg.qr(Z[idx1], mode="complete")
            # Z = QR
            diag = torch.diag_embed(R.diagonal(dim1=-2, dim2=-1).sign())
            Q = torch.bmm(Q, diag)
            idx2 = torch.where(torch.linalg.det(Q) > 0)[0]
            sample = torch.cat((sample, Q[idx2]), dim=0)
        return sample[:sample_num].reshape(-1, self.out_dim)

    def exp_from_identity(self, y):
        y = y.reshape(-1, self.mat_dim, self.mat_dim)
        return torch.matrix_exp(y).reshape(-1, self.out_dim)

    def exp(self, y, base_point):
        y = y.reshape(-1, self.mat_dim, self.mat_dim)
        base_point = base_point.reshape(-1, self.mat_dim, self.mat_dim)
        lie_algebra = torch.bmm(torch.transpose(base_point, dim0=1, dim1=2), y)
        temp = self.exp_from_identity(lie_algebra).reshape(-1, self.mat_dim, self.mat_dim)
        shifted = torch.bmm(base_point, temp)
        return shifted.reshape(-1, self.out_dim)

    def project_onto_manifold_SDE(self, y, base_point):
        return self.project_onto_manifold(base_point + y)


