import torch
from torch.func import vmap


def cal_div(y, x):
    div = 0.
    for i in range(y.shape[1]):
        div += torch.autograd.grad(
            outputs=y[:, i].sum(),
            inputs=x,
            create_graph=True,
            allow_unused=True)[0][:, i]
    return div


def get_norm_normal_vec(normal_vec):
    if normal_vec.dim() == 2: normal_vec = normal_vec.unsqueeze(dim=1)
    tmp_mat = torch.bmm(normal_vec, torch.transpose(normal_vec, dim0=1, dim1=2))
    L, Q = torch.linalg.eigh(tmp_mat)
    normalize_mat = torch.bmm(torch.bmm(Q, torch.diag_embed(1.0 / torch.sqrt(L))), torch.transpose(Q, dim0=1, dim1=2))
    norm_normal_vec = torch.bmm(normalize_mat, normal_vec)
    return norm_normal_vec


def loss_vesde(sde, score_fn, batch, config):
    eps = config.training.eps
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    std = sde.marginal_prob(batch, t)[1]

    z = torch.randn_like(batch, device=batch.device)
    perturbed_data = batch + std[:, None] * z

    if config.training.mode == 'dsm':
        scores = score_fn(perturbed_data, t)
        losses = 0.5 * torch.norm(scores, dim=1) ** 2 + torch.sum(scores * z / std[:, None], dim=1)

    elif config.training.mode == 'ism':
        perturbed_data = perturbed_data.detach().clone()
        perturbed_data.requires_grad_(True)
        scores = score_fn(perturbed_data, t)
        div = cal_div(scores, perturbed_data)
        losses = 0.5 * torch.norm(scores, dim=1) ** 2 + div

    weight = std ** config.training.ReweightPower
    return torch.mean(losses * weight)


def loss_vesde_noniso(sde, score_fn, batch, config, manifold):
    """
    cov_mat_inv:
    """
    device = batch.device
    eps = config.training.eps
    t = torch.rand(batch.shape[0], device=device) * (sde.T - eps) + eps
    bsz, dim = batch.shape
    c = torch.tensor(config.model.c).to(device)
    std = sde.marginal_prob(batch, t)[1]

    if config.problem.manifold == "SOn":
        norm_normal_vec = manifold.constrain_grad_fn(batch, normalized=True)
    else:
        normal_vec = manifold.constrain_grad_fn(batch)
        norm_normal_vec = get_norm_normal_vec(normal_vec)

    norm_normal_vec_tensor = torch.bmm(torch.transpose(norm_normal_vec, dim0=1,dim1=2), norm_normal_vec)
    cov_mat_inv = (1 / std[:,None,None] ** 2) * (torch.eye(dim, device=device).repeat(bsz, 1, 1) - c ** 2 / (c ** 2 + std[:,None,None] ** 2) * norm_normal_vec_tensor)
    non_isotropic_noise = torch.randn_like(batch, device=device) * std[:, None] + torch.matmul(torch.randn(bsz, 1, norm_normal_vec.shape[1], device=device), norm_normal_vec).squeeze(dim=1) * c
    perturbed_data = batch + non_isotropic_noise

    if config.training.mode == 'dsm':
        scores = score_fn(perturbed_data, t)
        target = - torch.bmm(cov_mat_inv, non_isotropic_noise.reshape(-1, dim, 1)).reshape(-1, dim)
        losses = 0.5 * torch.norm(scores, dim=1) ** 2 - (scores * target).sum(dim=-1)

    elif config.training.mode == 'ism':
        perturbed_data = perturbed_data.detach().clone()
        perturbed_data.requires_grad_(True)
        scores = score_fn(perturbed_data, t)
        div = cal_div(scores, perturbed_data)
        losses = 0.5 * torch.norm(scores, dim=1) ** 2 + div

    weight = std ** config.training.ReweightPower
    return torch.mean(losses * weight)



def loss_vesde_projected(sde, score_fn, batch, config, manifold):
    
    eps = config.training.eps
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch, device=batch.device)
    c = torch.tensor(config.model.c).to(batch)
    std = sde.marginal_prob(batch, t)[1]
    idx_wo_proj = torch.where(std >= c)[0]
    idx_w_proj = torch.where(std < c)[0]
    perturbed_data = batch + std[:, None] * z

    if config.training.mode == 'dsm':
        '------------------dsm---------------------'
        scores_wo = score_fn(perturbed_data[idx_wo_proj], t[idx_wo_proj])
        losses_wo = 0.5 * torch.norm(scores_wo, dim=1) ** 2 + \
                    torch.sum(scores_wo * z[idx_wo_proj] / std[idx_wo_proj, None], dim=1)
        '------------------proj_dsm---------------------'
        base_point = perturbed_data[idx_w_proj]
        # scores_w = score_fn(base_point, t[idx_w_proj])
        # target_w = manifold.project_onto_tangent_space(z[idx_w_proj] / std[idx_w_proj, None], base_point=base_point)
        scores_w = manifold.project_onto_tangent_space(score_fn(base_point, t[idx_w_proj]), base_point=base_point)
        target_w = manifold.project_onto_tangent_space(z[idx_w_proj] / std[idx_w_proj, None], base_point=base_point)
        losses_w = 0.5 * torch.norm(scores_w, dim=1) ** 2 + torch.sum(scores_w * target_w, dim=1)
    elif config.training.mode == 'ism':
        '------------------ism---------------------'
        samples_wo = perturbed_data[idx_wo_proj].detach().clone()
        samples_wo.requires_grad_(True)
        scores_wo = score_fn(samples_wo, t[idx_wo_proj])
        div_wo = 0.
        for i in range(scores_wo.shape[1]):
            div_wo += torch.autograd.grad(
                outputs=scores_wo[:, i].sum(),
                inputs=samples_wo,
                create_graph=True,
                allow_unused=True)[0][:, i]
        losses_wo = 0.5 * torch.norm(scores_wo, dim=1) ** 2 + div_wo
        '------------------proj_ism---------------------'
        samples_w = perturbed_data[idx_w_proj].detach().clone()
        samples_w.requires_grad_(True)
        scores_w = manifold.project_onto_tangent_space(score_fn(samples_w, t[idx_w_proj]), base_point=samples_w)
        div_w = 0.
        for i in range(scores_w.shape[1]):
            div_w += torch.autograd.grad(
                outputs=scores_w[:, i].sum(),
                inputs=samples_w,
                create_graph=True,
                allow_unused=True)[0][:, i]
        losses_w = 0.5 * torch.norm(scores_w, dim=1) ** 2 + div_w

    weight = std ** config.training.ReweightPower
    return (torch.sum(losses_wo * weight[idx_wo_proj]) + torch.sum(losses_w * weight[idx_w_proj]))/batch.shape[0]


def loss_vesde_rescale(sde, score_fn, batch, config, net_rescale_fn):
    eps = config.training.eps
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    std = sde.marginal_prob(batch, t)[1]
    z = torch.randn_like(batch, device=batch.device)
    perturbed_data = batch + std[:, None] * z
    
    scores = score_fn(perturbed_data, t)
    losses = 0.5 * torch.norm(scores, dim=1) ** 2 \
             + torch.sum(scores * z / std[:, None], dim=1)
    scal_coeff = net_rescale_fn(t)
    # weight = (scal_coeff ** 2) * (std ** config.training.ReweightPower)
    weight = scal_coeff ** 2
    return torch.mean(losses * weight)


def loss_vesde_noniso_rescale(sde, score_fn, batch, config, manifold, net_rescale_fn):
    device = batch.device
    eps = config.training.eps
    t = torch.rand(batch.shape[0], device=device) * (sde.T - eps) + eps
    bsz, dim = batch.shape
    c = torch.tensor(config.model.c).to(device)
    std = sde.marginal_prob(batch, t)[1]

    if config.problem.manifold == "SOn":
        norm_normal_vec = manifold.constrain_grad_fn(batch, normalized=True)
    elif config.problem.manifold == "Mesh":
        norm_normal_vec = manifold.constrain_grad_fn(batch).unsqueeze(1)
    else:
        normal_vec = manifold.constrain_grad_fn(batch)
        norm_normal_vec = get_norm_normal_vec(normal_vec)

    norm_normal_vec_tensor = torch.bmm(torch.transpose(norm_normal_vec, dim0=1,dim1=2), norm_normal_vec)
    cov_mat_inv = (1 / std.reshape(-1, 1, 1) ** 2) * (torch.eye(dim, device=device).repeat(bsz, 1, 1) - c ** 2 / (c ** 2 + std.reshape(-1, 1, 1) ** 2) * norm_normal_vec_tensor)
    non_isotropic_noise = torch.randn_like(batch, device=device) * std.reshape(-1, 1) + torch.matmul(torch.randn(bsz, 1, norm_normal_vec.shape[1], device=device), norm_normal_vec).squeeze(dim=1) * c
    perturbed_data = batch + non_isotropic_noise
    
    scores = score_fn(perturbed_data, t)
    target = - torch.bmm(cov_mat_inv, non_isotropic_noise.reshape(-1, dim, 1)).reshape(-1, dim)
    losses = 0.5 * torch.norm(scores, dim=1) ** 2 - (scores * target).sum(dim=-1)
    
    scal_coeff = net_rescale_fn(t)
    # weight = (scal_coeff ** 2) * ((std/scal_coeff) ** config.training.ReweightPower)
    weight = scal_coeff * std
    return torch.mean(losses * weight)


def loss_vesde_proj_rescale(sde, score_fn, batch, config, manifold, net_rescale_fn):
    eps = config.training.eps
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch, device=batch.device)
    c = torch.tensor(config.model.c).to(batch)
    std = sde.marginal_prob(batch, t)[1]
    idx_wo_proj = torch.where(std >= c)[0]
    idx_w_proj = torch.where(std < c)[0]
    perturbed_data = batch + std[:, None] * z
    
    '------------------dsm---------------------'
    scores_wo = score_fn(perturbed_data[idx_wo_proj], t[idx_wo_proj])
    losses_wo = 0.5 * torch.norm(scores_wo, dim=1) ** 2 + \
                torch.sum(scores_wo * z[idx_wo_proj] / std[idx_wo_proj, None], dim=1)
    '------------------proj_dsm---------------------'
    base_point = perturbed_data[idx_w_proj]
    scores_w = manifold.project_onto_tangent_space(score_fn(base_point, t[idx_w_proj]), base_point=base_point)
    target_w = manifold.project_onto_tangent_space(z[idx_w_proj] / std[idx_w_proj, None], base_point=base_point)
    losses_w = 0.5 * torch.norm(scores_w, dim=1) ** 2 + torch.sum(scores_w * target_w, dim=1)

    scal_coeff = net_rescale_fn(t)
    weight = scal_coeff * std
    # weight = (scal_coeff ** 2) * (std ** config.training.ReweightPower)

    return (torch.sum(losses_wo * weight[idx_wo_proj]) + torch.sum(losses_w * weight[idx_w_proj])) / batch.shape[0]



if __name__ == "__main__":
    pass

