import logging
import losses
from sde_lib import VESDE
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import networks_lib
import tensorboardX
import time
from utils import save_model, load_model, ExponentialMovingAverage
from cal_distri_dist import cal_wasser_dist, cal_mmd_dist, cal_sliced_wasser_dist, cal_JS_dist, cal_entropy_dist, cal_TVD_dist
import manifolds
import yaml
import pickle


class BasicRunner:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        """------------------------get directory------------------------"""
        self.workdir = self.config.workdir
        self.savefig_dir = os.path.join(self.workdir, 'figs')
        self.samples_dir = os.path.join(self.workdir, 'samples')
        self.validate_dir = os.path.join(self.workdir, 'validate')
        os.makedirs(self.savefig_dir)
        os.makedirs(self.samples_dir)
        os.makedirs(self.validate_dir)
        self.dataset_name = self.config.problem.dataset

        """------------------------get manifolds------------------------"""
        if self.config.problem.manifold == "R2inR3":
            self.manifold = manifolds.Manifold_R2inR3()
        elif self.config.problem.manifold == "SOn":
            self.manifold = manifolds.Manifold_SOn(self.config.problem.mat_dim)
        elif self.config.problem.manifold == "Mesh":
            self.obj = self.dataset_name.split('_')[0]
            mesh_path = f"./data/{self.obj}/{self.obj}_mesh_simple.ply"
            self.manifold = manifolds.Manifold_Mesh(mesh_path=mesh_path, device=self.device)
        elif self.config.problem.manifold == "SDF":
            self.obj = self.dataset_name.split('_')[0]
            self.sdf_model_path = f'./constraint/model/{self.config.sdf_model_name}'
            mesh_path = f"./data/{self.obj}/{self.obj}_mesh_simple.ply"
            self.manifold = manifolds.Manifold_SDF(model_path=self.sdf_model_path, mesh_path=mesh_path)
            torch.save(self.manifold.model.state_dict(), os.path.join(self.workdir + "/sdf_constraint_model.pt"))
            self.manifold.model.to(self.device)
            logging.info(f"Constrant Network Xi:\n {self.manifold.model.__str__()}")
        elif self.config.problem.manifold == "MD":
            self.manifold = manifolds.Manifold_MD()
        else:
            raise NotImplementedError

        """------------------------get sde------------------------"""
        self.sde = VESDE(sigma_min=self.config.model.sigma_min,
                         sigma_max=self.config.model.sigma_max,
                         N=self.config.model.N)
        self.plot_sde_info()
        if self.config.training.algo in ['vesde', 'vesde_rescale', 'vesde_ana']:
            self.config.model.c = 0.
        """------------------------others------------------------"""
        self.tb_logger = tensorboardX.SummaryWriter(log_dir=self.workdir)
        self.dist_dist_fn = None
        if self.config.training.algo in ['vesde_projected', 'vesde_proj_rescale'] and self.config.problem.manifold != "R2inR3":
            self.config.sample.Reverse_sde = False
            self.config.sample.Reverse_ode = False
        self.distri_dist_rec_dict = {self.config.seed:{'true':{}}}

    def get_mlp_network(self):
        network_mode = self.config.network.network_mode
        activation = self.config.network.activation
        scale = self.config.network.scale
        layers = [self.manifold.out_dim+1] + self.config.network.hidden_layers + [self.manifold.out_dim]
        
        if network_mode == "MLP":
            return networks_lib.MLP(layers, scale=scale, activation=activation)
        elif network_mode == "EMLP":
            return networks_lib.EMLP(layers, xref=self.xref, scale=scale, activation=activation)
    
    def get_score_net(self, network):
    
        if "rescale" in self.config.training.algo:
            # network: score * rescale_fn(t)
            
            if self.config.training.algo == "vesde_proj_rescale":
                self.config.training.rescale_fn = "max"
                
            if self.config.training.rescale_fn == "exact":
                logging.info('rescale_fn is exact!!!')
                self.net_rescale_fn = lambda t: torch.sqrt(self.sde.marginal_prob(0, t)[1] ** 2 + self.config.model.c **2)
            elif self.config.training.rescale_fn == "max":
                self.net_rescale_fn = lambda t: torch.maximum(self.sde.marginal_prob(0, t)[1], torch.tensor(self.config.model.c))
            else:
                self.net_rescale_fn = lambda t: self.sde.marginal_prob(0, t)[1] + self.config.model.c
            
            def score_fn(x, t):
                out = network(x, t)
                return out / self.net_rescale_fn(t).reshape(-1, 1)
            
            return score_fn
        else:
            self.net_rescale_fn = None
            return network
            
    def run(self):
        self.network = self.get_mlp_network()
        if self.config.if_train:
            self.score_net = self.get_score_net(self.network)
            self.train_step()
        else:
            state_dict = load_model(self.config)
            self.network.load_state_dict(state_dict)
            self.network.to(self.device)
            self.score_net = self.get_score_net(self.network)
        save_model(self.workdir, self.network, name="model.pt")

        if self.config.if_test:
            self.test()

        if self.config.if_sample:
            self.sample()

    def train_step(self):
        logging.info('Start training...')
        logging.info(f"Network:\n {self.network.__str__()}")
        logging.info(f"Param. No.: {sum(p.numel() for p in self.network.parameters() if p.requires_grad)}")

        self.network.to(self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.optim.lr, weight_decay=0.000,
                                     betas=(0.9, 0.999), amsgrad=False)

        training_loader = torch.utils.data.DataLoader(self.training_set, batch_size=self.config.training.batch_size,
                                                      shuffle=True, drop_last=True)

        self.total_epochs = self.config.training.n_epochs
        step = 0
        loss_train_list = []
        for epoch in range(self.total_epochs+1):
            for _, samples in enumerate(training_loader):
                step += 1
                samples = samples.to(self.device)

                if self.config.training.algo == 'vesde':
                    loss = losses.loss_vesde(self.sde, self.score_net, samples, self.config)
                elif self.config.training.algo == 'vesde_noniso':
                    loss = losses.loss_vesde_noniso(self.sde, self.score_net, samples, self.config, self.manifold)
                elif self.config.training.algo == 'vesde_projected':
                    loss = losses.loss_vesde_projected(self.sde, self.score_net, samples, self.config, self.manifold)
                    
                elif self.config.training.algo == 'vesde_rescale':
                    loss = losses.loss_vesde_rescale(self.sde, self.score_net, samples, self.config, self.net_rescale_fn)
                elif self.config.training.algo == 'vesde_noniso_rescale':
                    loss = losses.loss_vesde_noniso_rescale(self.sde, self.score_net, samples, self.config, self.manifold, self.net_rescale_fn)
                elif self.config.training.algo == 'vesde_proj_rescale':
                    loss = losses.loss_vesde_proj_rescale(self.sde, self.score_net, samples, self.config, self.manifold, self.net_rescale_fn)
                else:
                    raise NotImplementedError

                optimizer.zero_grad()
                loss.backward()
                if self.config.training.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.config.training.clip_threshold)
                optimizer.step()

                if epoch > 0: self.ema.update(self.network.parameters())

            "-------------------------------------validate---------------------------------------"
            if epoch == 0: self.ema = ExponentialMovingAverage(self.network.parameters(), 0.999)

            self.tb_logger.add_scalar(f'loss_{self.dataset_name}', loss, global_step=epoch)
            loss_train_list.append(loss.detach().cpu().numpy())

            if epoch % (int(self.total_epochs / 10)) == 0:
                save_model(self.validate_dir, self.network, name=f"model_temp.pt")
                logging.info(f'epoch: {epoch}/{self.total_epochs}, step: {step}, loss: {loss.item():.6f}')
                if self.config.if_validate:
                    self.ema.store(self.network.parameters())
                    self.ema.copy_to(self.network.parameters())
                    self.validate(epoch=epoch)
                    self.ema.restore(self.network.parameters())

                fig = plt.figure()
                plt.plot(loss_train_list, c='b', label='loss')
                plt.savefig(self.savefig_dir + f"/aa_loss_training.png", dpi=300, bbox_inches='tight')
                plt.close(fig)

        self.ema.copy_to(self.network.parameters())

    def plot_constraint_hist(self, samples, savefig=None):
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples)
        fig = plt.figure()
        constrain = self.manifold.constrain_fn(samples).detach().cpu().numpy()
        plt.hist(constrain, bins=50, alpha=1.0, density=True)

        plt.savefig(self.savefig_dir + f"/constraint_hist_{savefig}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_sde_info(self):
        t = torch.linspace(0, 1, 1001)
        g_t = self.sde.sde(torch.zeros_like(t), t)[1]
        std = self.sde.marginal_prob(0, t)[1]

        fig = plt.figure(figsize=(10,5))

        ax = fig.add_subplot(121)
        ax.plot(t, std)
        ax.plot(t, self.config.model.c * torch.ones_like(t), linestyle='--')
        ax.set_xlabel('t')
        ax.set_ylabel('sigma')
        ax.set_title('Noise scale for sde')

        ax = fig.add_subplot(122)
        ax.plot(t, g_t)
        ax.set_xlabel('t')
        ax.set_ylabel('g_t')
        ax.set_title('g_t in sde')

        plt.savefig(self.savefig_dir + "/sde_info.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def calculate_constrain(self, samples):
        samples_constrain = self.manifold.constrain_fn(samples)
        logging.info(f'Samples constrain: {samples_constrain.min().item():.2e} - {samples_constrain.max().item():.2e}')

    def sample(self):
        logging.info(f'Start sampling under training algorithm {self.config.training.algo}.')
        self.network.to(self.device)

        if self.config.sample.Reverse_sde:
            logging.info(f'start sampling by Reverse-sde.')
            self.distri_dist_rec_dict[self.config.seed]['Reverse-sde'] = {}
            self.generate_new_samples(mode='Reverse-sde')
        if self.config.sample.Reverse_ode:
            logging.info(f'start sampling by Reverse-ode.')
            self.distri_dist_rec_dict[self.config.seed]['Reverse-ode'] = {}
            self.generate_new_samples(mode='Reverse-ode')
        if self.config.sample.corrector:
            logging.info(f'start sampling by Corrector.')
            self.distri_dist_rec_dict[self.config.seed]['Corrector'] = {}
            self.generate_new_samples(mode='Corrector', threshold=self.config.sample.sample_threshold)
        if self.config.sample.Early_stop:
            logging.info(f'start sampling by Early-stop.')
            self.distri_dist_rec_dict[self.config.seed]['Early-stop'] = {}
            self.generate_new_samples(mode='Early-stop', threshold=self.config.sample.sample_threshold)

        print(self.distri_dist_rec_dict)
        with open(os.path.join(self.workdir, 'logs', 'distri_dist_rec_dict.pkl'), "wb") as file:
            pickle.dump(self.distri_dist_rec_dict, file)
        print(self.distri_dist_rec_dict)
        with open(os.path.join(self.workdir, 'logs', 'distri_dist_rec_dict.yml'), "w") as file:
            yaml.dump(self.distri_dist_rec_dict, file, default_flow_style=False, allow_unicode=True)


    def cal_distri_dist_fn(self, samples_list, mode, sample_mode):
        def cal_dist(a, b):
            if mode == "JS":
                dist = cal_JS_dist(a, b)
                return dist
            if mode == "entropy":
                return cal_entropy_dist(a, b)
            if mode == "TVD":
                return cal_TVD_dist(a, b)
                
            sample_num = self.config.cal_dist_sample_num
            a, b = a[:sample_num], b[:sample_num]

            start_time = time.time()
            if mode == "wasser1":
                dist = cal_wasser_dist(a, b, power=1, dist_fn=self.dist_dist_fn)
            elif mode == "wasser2":
                dist = cal_wasser_dist(a, b, power=2, dist_fn=self.dist_dist_fn)
            elif mode == "mmd":
                dist = cal_mmd_dist(a, b, dist_fn=self.dist_dist_fn)
            elif mode == "sliced_wasser":
                dist = cal_sliced_wasser_dist(a, b)
            else:
                raise NotImplementedError
            end_time = time.time()
            run_time = end_time - start_time
            if run_time > 1800: logging.info(f"Runtime: {run_time/3600: .2f} h.")
            return dist
        
        wasserstein_dist_list = []
        for _, samples in enumerate(samples_list):
            statistics2 = self.data_for_cal_dist(samples)
            wasserstein_dist_list.append(cal_dist(self.statistics_true, statistics2))
        wasserstein_dist_list = np.array(wasserstein_dist_list).astype(np.float32)

        self.distri_dist_rec_dict[self.config.seed][sample_mode][mode] = wasserstein_dist_list.tolist()

        mean = wasserstein_dist_list.mean()
        std = np.sqrt(np.mean((wasserstein_dist_list-mean)**2))
        logging.info(f'Distance between distribution mode {mode}: {mean:.5f}({std:.5f})')
        logging.info(f'Info for grep:__{sample_mode}__{mode}__{mean:.5f}__{std:.5f}__')


if __name__ == "__main__":
    pass

