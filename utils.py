import shutil
import logging
import sys
import os
import torch
import numpy as np
import torch.optim as optim
import torch.distributions as D



def get_logger(log_dir, verbose='info'):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    # setup logger
    level = getattr(logging, verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)
    return logger


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_optimizer(parameters, config):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))


def save_model(workdir, network, name):
    save_path = os.path.join(workdir, name)
    torch.save(network.state_dict(), save_path)


def load_model(config):
    if config.load_model_path != "no_path":
        model_path = os.path.join(config.workdir, config.load_model_path)
        if os.path.exists(model_path):
            return torch.load(model_path, map_location=torch.device('cpu'))
    elif config.load_model_prefix != "no_prefix":
        path_temp0 = os.path.join(config.workdir, "..")
        for entry in os.listdir(path_temp0):
            path_temp1 = os.path.join(path_temp0, entry)
            prefix = config.load_model_prefix + f"-{config.seed}-"
            if os.path.isdir(path_temp1) and entry.startswith(prefix):
                path_temp2 = os.path.join(path_temp1, "model.pt")
                if os.path.exists(path_temp2):
                    return torch.load(path_temp2, map_location=torch.device('cpu'))

    logging.info(f'No trained model is found !!!!!!!!!!!!!!!!!!!!!!')
    return None


def split_dataset(data_ori, data_seed, test_rate=0.2):
    # no val_set !!!
    total_size = data_ori.shape[0]
    test_size = int(total_size * test_rate)
    training_size = total_size - test_size
    train_data, test_data = torch.utils.data.random_split(data_ori, [training_size, test_size],
                                                          generator=torch.Generator().manual_seed(data_seed))
    training_set = data_ori[train_data.indices]
    test_set = data_ori[test_data.indices]
    logging.info(f"size of the training set: {training_set.shape[0]}, size of the test set: {test_set.shape[0]}, split seed: {data_seed}.")
    return training_set, test_set, None


def get_GMMDist(mean, cov, mix):
    mix = D.Categorical(mix)
    comp = D.Independent(D.MultivariateNormal(mean, cov), 0)
    return D.mixture_same_family.MixtureSameFamily(mix, comp)


def run_func_in_batches(func, x, max_batch_size, out_dim=None):
    k = int(np.ceil(x.shape[0] / max_batch_size))
    if out_dim is None:
        out = torch.empty(x.shape[0])
    else:
        out = torch.empty(x.shape[0], out_dim)
    for i in range(k):
        x_i = x[i * max_batch_size: (i+1) * max_batch_size]
        out[i * max_batch_size: (i+1) * max_batch_size] = func(x_i).detach().view(x_i.shape[0], out_dim)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out


def get_density_mix_gaussian(x, mean, cov, mix):
    """
    N: the number of samples
    k: the number of well
    dim:
    """
    assert len(cov.shape) == 3 # k*dim*dim
    assert mix.shape[0] == cov.shape[0] # k

    # k*d*d
    cov_t_inv = torch.linalg.inv(cov)

    # k
    det = torch.linalg.det(cov_t_inv)

    # N*k*d
    tmp = x.unsqueeze(dim=1)-mean
    # N*k
    tmp1 = torch.matmul(tmp.unsqueeze(dim=2), torch.matmul(cov_t_inv, tmp.unsqueeze(dim=-1))).squeeze(dim=-1).squeeze(dim=-1)
    # N*k
    wi = torch.pow(2* torch.tensor(torch.pi), -x.shape[1]*0.5) * mix * det**0.5 * torch.exp(-0.5 * tmp1)

    return wi.sum(dim=1)


def get_score_mix_gaussian(x, mean, cov, mix):
    """
    N: the number of samples
    k: the number of well
    dim:
    """
    assert len(cov.shape) == 3 # k*dim*dim
    assert mix.shape[0] == cov.shape[0] # k

    # k*d*d
    cov_t_inv = torch.linalg.inv(cov)

    # k
    det = torch.linalg.det(cov_t_inv)

    # N*k*d
    tmp = x.unsqueeze(dim=1)-mean
    # N*k
    tmp1 = torch.matmul(tmp.unsqueeze(dim=2), torch.matmul(cov_t_inv, tmp.unsqueeze(dim=-1))).squeeze(dim=-1).squeeze(dim=-1)

    # !!! maybe it is nan!
    # N*k
    wi = mix * det**0.5 * (torch.exp(-0.5 * tmp1) + 1e-6)
    # wi = mix * det**0.5 * torch.exp(-0.5 * tmp1)

    # N*2
    score = -1.0 * torch.matmul(wi.unsqueeze(dim=1), torch.matmul(cov_t_inv, tmp.unsqueeze(dim=-1)).squeeze(dim=-1)).squeeze(dim=1) / wi.sum(dim=1, keepdim=True)

    return score


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) /
                        (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)
    
    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']


def cal_trace(mat):
    return mat.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)


def check_memory(data=None):
    if data is not None:
        memory_bytes = data.element_size() * data.nelement()
        logging.info(f"The data (shape {data.shape}) occupy {memory_bytes / (1024 ** 3):.3f} G of memory on {data.device}.")
    
    if torch.cuda.is_available():
        logging.info(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.3f}G")
        torch.cuda.empty_cache()


@torch.no_grad()
def Kabsch(x, xref):
    """
    align states by translation and rotation.
    This method implements the Kabsch algorithm.
    """
    assert isinstance(x, torch.Tensor), 'Input x is not a torch tensor'

    # device = x.device
    xref = xref.to(x)
    align_atom_indices = list(range(0, xref.shape[0]))

    xref = xref - torch.mean(xref[align_atom_indices, :], 0, True)
    b = torch.mean(x[:, align_atom_indices, :], 1, True)
    x_notran = x[:, align_atom_indices, :] - b

    if not torch.isfinite(x_notran).all():
        print("Some point go to the infinity!!")

    xtmp = x_notran.permute((0, 2, 1))

    prod = torch.matmul(xtmp, xref)  # batched matrix multiplication, output dimension: traj_length x 3 x 3
    # u, s, vh = torch.linalg.svd(prod)

    try:
        u, s, vh = torch.linalg.svd(prod)
    except torch._C._LinAlgError as e:
        error_idx = torch.where(~torch.isfinite(prod))[0].unique()
        print(f"Index {error_idx} of prod is NAN! It was changed into zero before SVD.")
        prod = torch.nan_to_num(prod, nan=0.0, posinf=0.0, neginf=0.0)
        u, s, vh = torch.linalg.svd(prod)

    diag_mat = torch.diag(torch.ones(3)).unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device, dtype=u.dtype)
    sign_vec = torch.sign(torch.linalg.det(torch.matmul(u, vh))).detach()
    diag_mat[:, 2, 2] = sign_vec

    R = torch.bmm(torch.bmm(u, diag_mat), vh)
    return R.transpose(1, 2), b


def get_RMSD(xvec, xref):
    xvec = xvec.reshape(-1, 10, 3)
    R, b = Kabsch(xvec, xref)
    b0 = torch.mean(xref, 0, True)
    error = xvec - b - torch.matmul(xref - b0, R)
    return torch.sqrt(torch.sum(error ** 2, dim=(1, 2))/xref.shape[0])


@torch.no_grad()
def refine_dataset_md(sdf, samples, tol=1e-5, max_iter_n=1000, step_size=1e-1, rk4=True, keep_quiet=False):

    print("refine data set...")

    if isinstance(samples, torch.Tensor):
        # use double instead of float in order to improve convergence!
        samples = samples.clone().double()
        device = samples.device
    else:
        samples = torch.tensor(samples, dtype=torch.float64)
        device = torch.device("cpu")

    active_idx = torch.arange(0, samples.shape[0], dtype=torch.int64).to(device)

    iter_n = 0
    while iter_n < max_iter_n:
        xi_vals = sdf.constrain_fn(samples[active_idx, :])
        error = torch.abs(xi_vals).squeeze(dim=1)
        bad_idx = (error >= tol)
        if bad_idx.sum() == 0:
            print("break")
            break
        else:
            if iter_n % 500 == 0 and not keep_quiet:
                print(f'iter {iter_n}: max_err={torch.max(error):.3e}, {bad_idx.sum()} bad states, tol={tol:.3e}')
        active_idx = active_idx[bad_idx]

        if rk4:
            # remove [:,:,None] in ki s !!!
            k1 = sdf.constrain_fn(samples[active_idx,:]) * sdf.constrain_grad_fn(samples[active_idx, :])
            tmp = samples[active_idx,:] + k1 * step_size * 0.5
            k2 = sdf.constrain_fn(tmp) * sdf.constrain_grad_fn(tmp)
            tmp = samples[active_idx,:] + k2 * step_size * 0.5
            k3 = sdf.constrain_fn(tmp) * sdf.constrain_grad_fn(tmp)
            tmp = samples[active_idx,:] + k3 * step_size 
            k4 = sdf.constrain_fn(tmp) * sdf.constrain_grad_fn(tmp)
            samples[active_idx, :] = samples[active_idx, :] - (k1 + 2*k2 + 2*k3 + k4) * step_size / 6.0
        else:
            samples[active_idx, :] = samples[active_idx, :] - sdf.constrain_fn(samples[active_idx,:])[:,:,None] * sdf.constrain_grad_fn(samples[active_idx, :]) * step_size

        iter_n += 1

    if iter_n == 0:
        print(f'total steps 0.')
        return samples.detach().cpu().float(), samples.detach().cpu().float()


    xi_vals = sdf.constrain_fn(samples)
    max_error = torch.max(torch.abs(xi_vals).squeeze())

    print(f'total steps={iter_n}, final error: {max_error: .3e}.')
    if max_error > tol * 1.1:
        print(f'Warning: tolerance ({tol: .3e}) not reached!')

    mask = torch.ones(samples.shape[0], dtype=torch.bool)
    mask[active_idx] = False
    samples_abridged = samples[mask, ...]

    return samples.detach().cpu().float(), samples_abridged.detach().cpu().float()


@torch.no_grad()
def refine_dataset_SDF(sdf, samples0, tol=1e-5, max_iter_n=1000, step_size=1e-1, keep_quiet=False):
    @torch.enable_grad()
    def sdf_grad(samples):
        samples.requires_grad_(True)
        gradients = torch.autograd.grad(
            outputs=sdf(samples).sum(),
            inputs=samples,
            create_graph=True,
            retain_graph=True)[0]
        return gradients.detach()

    device = next(sdf.parameters()).device

    if isinstance(samples0, torch.Tensor):
        samples = samples0.to(device)
    else:
        samples = torch.tensor(samples0, dtype=torch.float32).to(device)

    active_idx = torch.arange(0, samples.shape[0], dtype=torch.int64).to(device)

    iter_n = 0
    while iter_n < max_iter_n:
        xi_vals = sdf(samples[active_idx, :])
        error = torch.abs(xi_vals).squeeze(dim=1)
        bad_idx = (error >= tol)
        if bad_idx.sum() == 0:
            break
        else:
            if iter_n % 50 == 0 and not keep_quiet:
                print(f'iter {iter_n}: max_err={torch.max(error):.3e}, {bad_idx.sum()} bad states, tol={tol:.3e}')
        active_idx = active_idx[bad_idx]
        samples[active_idx, :] = samples[active_idx, :] - xi_vals[bad_idx,:] * sdf_grad(samples[active_idx,:]) * step_size
        iter_n += 1

    xi_vals = sdf(samples)
    max_error = torch.max(torch.abs(xi_vals).squeeze())

    print(f'Total steps={iter_n}, final error: {max_error: .3e}.')
    if max_error > tol * 1.1:
        print(f'Warning: tolerance ({tol: .3e}) not reached!')

    # new
    mask = torch.ones(samples.shape[0], dtype=torch.bool)
    mask[active_idx] = False
    samples_abridged = samples[mask, ...]

    return samples.detach(), samples_abridged.detach()

def get_scene_dict(obj):
    scene_dict = {"bunny": dict(xaxis=dict(range=(-1.05, 1.05), autorange=False),
                               yaxis=dict(range=(-1.05, 1.05), autorange=False),
                               zaxis=dict(range=(-1.05, 1.05), autorange=False),
                               aspectratio=dict(x=1, y=1, z=1),
                               camera=dict(eye=dict(x=-0.5, y=0, z=-2),
                                            up=dict(x=0, y=1, z=0),
                                            center=dict(x=0, y=0, z=0))),
    "spot": dict(xaxis=dict(range=(-1.05, 1.05), autorange=False),
                               yaxis=dict(range=(-1.05, 1.05), autorange=False),
                               zaxis=dict(range=(-1.05, 1.05), autorange=False),
                               aspectratio=dict(x=1, y=1, z=1),
                               camera=dict(eye=dict(x=-1, y=1, z=1),
                                            up=dict(x=0, y=1, z=0),
                                            center=dict(x=0, y=0, z=0)))}
    return scene_dict[obj]



if __name__ == "__main__":
    pass


