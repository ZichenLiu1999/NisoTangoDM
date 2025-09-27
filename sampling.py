import logging
from torchdiffeq import odeint
import numpy as np
import torch
from tqdm import tqdm
import time

@torch.no_grad()
def SDE_sampler_two_stage(config, score_net, sde, manifold, mode, threshold=None,
                          return_hist=False, keep_quiet=True, **kwargs):
    assert mode in ['Reverse-sde', 'Early-stop', 'Corrector'], "The parameter 'mode' is false."
    device = config.device
    x = sde.prior_sampling((config.sample.sample_num, manifold.out_dim)).to(device)
    timesteps = torch.linspace(sde.T, 0., sde.N, device=device)
    rsde = sde.reverse(score_net, probability_flow=False)
    dt = -sde.T / rsde.N # dt < 0 !!!

    def update_fn_reverse_sde(x, t_tmp):
        t = torch.ones(x.shape[0], device=device) * t_tmp
        z = torch.randn_like(x, device=device)
        drift, diffusion = rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None] * np.sqrt(-dt) * z
        return x.detach()

    def update_fn_corrector(x, t_tmp):
        for i in range(config.sample.corrector_step):
            snr = config.sample.corrector_snr
            t = torch.ones(x.shape[0], device=device) * t_tmp
            grad = - manifold.project_onto_tangent_space(score_net(x, t), base_point=x)
            z = manifold.project_onto_tangent_space(torch.randn_like(x, device=device), base_point=x)

            noise_norm = torch.norm(z, dim=-1).mean()
            grad_norm = torch.norm(grad, dim=-1).mean()
            # grad_norm = torch.norm(grad, dim=-1, keepdim=True)
            step_size = 2 * (snr * noise_norm / grad_norm) ** 2 # step_size > 0 !!!
            vec = - step_size * grad + torch.sqrt(step_size * 2) * z
            x = manifold.project_onto_manifold_SDE(vec, base_point=x)
        return x.detach()

    sampling_time = []
    start_time = time.time()

    x_hist = torch.zeros(sde.N+1, *x.shape).to(device)
    x_hist[0] = x.clone()
    for i in tqdm(range(sde.N), mininterval=2., disable=keep_quiet):
        t_tmp = timesteps[i]

        if mode == 'Reverse-sde':
            x = update_fn_reverse_sde(x, t_tmp)
        elif mode == 'Early-stop':
            if sde.marginal_prob(0, t_tmp)[1] >= threshold:
                x = update_fn_reverse_sde(x, t_tmp)
            else:
                return (x, x_hist[:i]) if return_hist else x
        elif mode == 'Corrector':
            if sde.marginal_prob(0, t_tmp)[1] >= threshold:
                x = update_fn_reverse_sde(x, t_tmp)
                Corrector_proj = True
            else:
                if Corrector_proj:
                    print("Project on manifold!!!")
                    x = manifold.project_onto_manifold(x)
                    Corrector_proj = False
                else:
                    x = update_fn_corrector(x, t_tmp)

        if return_hist: x_hist[i + 1] = x.clone()

        end_time = time.time()
        sampling_time.append(end_time - start_time)
        
    return (x, x_hist) if return_hist else x


@torch.no_grad()
def ode_sampler(config, score_net, sde, manifold, **kwargs):
    logging.info(f'start sampling by ode_sampler.')

    device = config.device
    init = sde.prior_sampling((config.sample.sample_num, manifold.out_dim)).to(device)
    rsde = sde.reverse(score_net, probability_flow=True)

    def ode_func(t, state):
        if not torch.is_tensor(t) or t.numel() == 1:
            t = t * torch.ones(state.shape[0]).to(state)
        t = t.to(state)
        drift, diffusion = rsde.sde(state, t)
        return drift

    state0 = init
    T = torch.tensor([1., 0.]).to(device)
    _, samples = odeint(ode_func, state0, T, atol=1e-5, rtol=1e-5)
    return samples


