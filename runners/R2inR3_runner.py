import logging
from runners.Basic_runner import BasicRunner
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from utils import split_dataset, get_GMMDist, get_score_mix_gaussian, get_density_mix_gaussian
from torch.func import vmap
from sampling import SDE_sampler_two_stage, ode_sampler


class R2inR3Runner(BasicRunner):
    def __init__(self, config):
        super().__init__(config)
        self.load_data()
        
        "-------------------------------exhibit dataset-----------------------------------"
        self.plot_scores_norm_ana(label=0., savefig=None)
        samples_test = self.training_set[:10000].clone()
        self.plot_sample(samples_test.numpy(), savefig="true")

    def load_data(self):
        
        self.x_bound, self.y_bound = [-self.config.bd_lim, self.config.bd_lim], [-self.config.bd_lim, self.config.bd_lim]
        self.mix = torch.ones(9) / 9.
        self.mean = torch.tensor([[-1., -1.], [-1., 0.], [-1., 1.],
                                    [0., -1.], [0., 0.], [0., 1.],
                                    [1., -1.], [1., 0.], [1., 1.]])
        self.well_center = self.mean
        cov = self.config.well_sigma ** 2
        self.cov = torch.tensor([[[cov, 0], [0, cov]]]).repeat(9, 1, 1)
        self.dist_inner = get_GMMDist(self.mean, self.cov, mix=self.mix)
    
        sample_num = 50000
        data_inner = self.dist_inner.sample((sample_num,))
        data_ori = torch.cat((data_inner, torch.zeros(sample_num, 1)), dim=1).float()
        np.save(f"./data/R2inR3/{self.dataset_name}.npy", data_ori.cpu().numpy())
        
        self.data_set = data_ori.clone()
        
        self.config.data_seed = self.config.seed
        self.training_set, self.test_set, _ = split_dataset(data_ori, self.config.data_seed)

        if self.config.if_cal_distri_dist:
            self.statistics_true = self.data_for_cal_dist(self.test_set[:self.config.sample.sample_num])
            self.dist_dist_fn = None
            
            logging.info(f'Calculating distance: sampling mode: __true__, training algorithm: __{self.config.training.algo}')
            statistics = self.data_for_cal_dist(self.training_set[:self.config.sample.sample_num])
            self.cal_distri_dist_all_fn([statistics], sample_mode="true")

    def get_ref_density(self, x, t):
        assert t.dim() == 0, 'time t should be a scalar (all x are at the same time)' 
        _, std = self.sde.marginal_prob(0, t)
        mean_temp = torch.cat((self.mean, torch.zeros(self.mean.shape[0], 1)), dim=1)
        cov_temp = torch.zeros(self.cov.shape[0], 3, 3)
        cov_temp[:, :2, :2] = self.cov
        cov_temp[:, 2, 2] = self.config.model.c ** 2 * torch.ones(self.cov.shape[0])
        cov_temp = cov_temp + std**2 * torch.eye(3).unsqueeze(dim=0) 
        return get_density_mix_gaussian(x, mean=mean_temp, cov=cov_temp, mix=self.mix)

    def get_ref_score(self, x, t):
        device = x.device
        assert t.dim() == 0, 'time t should be a scalar , all x are at the same time.'
        std = self.sde.marginal_prob(0, t)[1]

        mean_temp = torch.cat((self.mean, torch.zeros(self.mean.shape[0], 1)), dim=1).to(device)
        cov_temp = torch.zeros(self.cov.shape[0], 3, 3)
        cov_temp[:, :2, :2] = self.cov
        cov_temp[:, 2, 2] = self.config.model.c ** 2 * torch.ones(self.cov.shape[0])
        cov_temp = cov_temp.to(device) + std**2 * torch.eye(3, device=device).unsqueeze(dim=0) 
        return get_score_mix_gaussian(x, mean=mean_temp, cov=cov_temp, mix=self.mix.to(device))

    def plot_sample(self, samples, savefig=None):
        fig = plt.figure()
        plt.scatter(samples[:, 0], samples[:, 1], s=0.5, c='green', alpha=0.8)
        if self.well_center is not None:
            plt.scatter(self.well_center[:, 0], self.well_center[:, 1], s=10.0, c='red', alpha=1.0)
        plt.title('Samples projected in R2')
        plt.xticks([self.x_bound[0], -1, 0, 1, self.x_bound[1]])
        plt.yticks([self.y_bound[0], -1, 0, 1, self.y_bound[1]])
        plt.savefig(self.savefig_dir + f"/samples_{savefig}.png", bbox_inches='tight')
        plt.close(fig)
    
    @torch.no_grad()
    def plot_scores_norm_ana(self, label=0., savefig=None):
        r_grid = torch.linspace(-0.2, 0.2, 100 + 1)
        
        grid_dense = 30
        x = torch.linspace(self.x_bound[0], self.x_bound[1], grid_dense)
        y = torch.linspace(self.y_bound[0], self.y_bound[1], grid_dense)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        mesh_ori = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1), torch.zeros(grid_dense ** 2, 1)), dim=1)
        
        scores_normal_norm_list = []
        for r in r_grid:
            mesh = mesh_ori + r * torch.tensor([[0.0, 0.0, 1.0]])
            scores = vmap(lambda x, y: self.get_ref_score(x.reshape(1, 3), y))(mesh, label * torch.ones(mesh.shape[0])).squeeze(1)
            scores_normal = scores * torch.tensor([[0.0, 0.0, 1.0]])
            scores_normal_norm = scores_normal.norm(dim=1)
            scores_normal_norm_list.append(scores_normal_norm.mean().detach().cpu().numpy())
        
        fig = plt.figure(figsize=(5, 5))
        
        ax = fig.add_subplot(111)
        ax.plot(r_grid.detach().cpu().numpy(), scores_normal_norm_list)
        ax.set_xlabel('r')
        ax.set_ylabel('norm')
        ax.set_title('normal')
        
        plt.suptitle(f'Norm of the ana score for label={label :.2f}')
        plt.savefig(self.savefig_dir + f"/score_norm_ana.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    @torch.no_grad()
    def plot_scores_norm(self, score_net, label, savefig=None, device='cpu'):
        
        r_grid = torch.linspace(-0.2, 0.2, 100+1)

        grid_dense = 30
        x = torch.linspace(self.x_bound[0], self.x_bound[1], grid_dense)
        y = torch.linspace(self.y_bound[0], self.y_bound[1], grid_dense)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        mesh_ori = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1), torch.zeros(grid_dense**2, 1)), dim=1).to(device)

        scores_normal_norm_list = []
        scores_tangent_norm_list = []
        for r in r_grid:
            mesh = mesh_ori + r * torch.tensor([[0.0, 0.0, 1.0]], device=device)
            t = label * torch.ones(mesh.shape[0], device=device)
            scores = score_net(mesh, t)
            
            scores_normal = scores * torch.tensor([[0.0, 0.0, 1.0]], device=device)
            scores_tangent = scores - scores_normal

            scores_normal_norm = scores_normal.norm(dim=1)
            scores_tangent_norm = scores_tangent.norm(dim=1)
            scores_normal_norm_list.append(scores_normal_norm.mean().detach().cpu().numpy())
            scores_tangent_norm_list.append(scores_tangent_norm.mean().detach().cpu().numpy())

        fig = plt.figure(figsize=(10,5))

        ax = fig.add_subplot(121)
        ax.plot(r_grid.detach().cpu().numpy(), scores_normal_norm_list)
        ax.set_xlabel('r')
        ax.set_ylabel('norm')
        ax.set_title('normal')

        ax = fig.add_subplot(122)
        ax.plot(r_grid.detach().cpu().numpy(), scores_tangent_norm_list)
        ax.set_xlabel('r')
        ax.set_ylabel('norm')
        ax.set_title('tangent')

        plt.suptitle(f'Norm of the different components of the score for label={label :.2f}')
        plt.savefig(self.savefig_dir + f"/score_norm_{savefig}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    def plot_scores_manifolds(self, t_temp, savefig=None, device='cpu'):
        fig = plt.figure(figsize=(15, 5))

        grid_N = 30
        x = torch.linspace(self.x_bound[0], self.x_bound[1], grid_N)
        y = torch.linspace(self.y_bound[0], self.y_bound[1], grid_N)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        grid = torch.cat([x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)], dim=-1)
        grid_extend = torch.cat((grid, torch.zeros(grid.shape[0], 1)), dim=1)
        tt = torch.ones(grid.shape[0]) * t_temp

        score_ana = self.get_ref_score(grid_extend, t_temp)[:, :2].reshape(grid_N, grid_N, 2).detach().numpy()
        score = self.score_net(grid_extend.to(device), tt.to(device))[:, :2].reshape(grid_N, grid_N, 2).detach().cpu().numpy()

        x_grid, y_grid = x_grid.detach().numpy(), y_grid.detach().numpy()

        ax = fig.add_subplot(131)
        ax.quiver(x_grid, y_grid, score[:, :, 0], score[:, :, 1], angles='xy')
        if self.well_center is not None:
            ax.scatter(self.well_center[:, 0], self.well_center[:, 1], s=10.0, c='green', alpha=0.7)
        ax.set_title('Learned')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax = fig.add_subplot(132)
        ax.quiver(x_grid, y_grid, score_ana[:,:,0], score_ana[:,:,1], angles='xy')
        if self.well_center is not None:
            ax.scatter(self.well_center[:, 0], self.well_center[:, 1], s=10.0, c='green', alpha=0.7)
        ax.set_title('Analytic')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax = fig.add_subplot(133)
        score_diff = score - score_ana
        ax.quiver(x_grid, y_grid, score_diff[:,:,0], score_diff[:,:,1], angles='xy')
        if self.well_center is not None:
            ax.scatter(self.well_center[:, 0], self.well_center[:, 1], s=10.0, c='green', alpha=0.7)
        ax.set_title('Difference')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        plt.savefig(self.savefig_dir + f"/score_tang_{savefig}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


    @torch.no_grad()
    def plot_scores_error_vs_t(self):
        bsz = self.test_set.shape[0]
        t_list = torch.linspace(0., 1., 20 + 1)
        c = self.config.model.c

        tangent_error_list = []
        tangent_relative_error_list = []
        normal_error_list = []
        normal_relative_error_list = []
        
        for t_temp in tqdm(t_list):
            tt = torch.ones(bsz) * t_temp
            std = self.sde.marginal_prob(0, tt)[1]

            noise = torch.randn(bsz, 3) * std.reshape(-1, 1)\
                    + torch.randn(bsz, 1) * torch.tensor([[0., 0., 1.]]) * c
            samples = self.test_set + noise

            score_ana = self.get_ref_score(samples, t_temp) 
            score = self.score_net(samples, tt).detach()
            score_diff = score - score_ana

            tangent_score_diff_norm = score_diff[:, :2].norm(dim=1)
            tangent_score_ana_norm = score_ana[:, :2].norm(dim=1)
            normal_score_diff_norm = score_diff[:, 2:].norm(dim=1)
            normal_score_ana_norm = score_ana[:, 2:].norm(dim=1)

            tangent_error = tangent_score_diff_norm.mean()
            tangent_relative_error = tangent_score_diff_norm.sum()/tangent_score_ana_norm.sum()
            normal_error = normal_score_diff_norm.mean()
            normal_relative_error = normal_score_diff_norm.sum()/normal_score_ana_norm.sum()

            tangent_error_list.append(tangent_error.numpy())
            tangent_relative_error_list.append(tangent_relative_error.numpy())
            normal_error_list.append(normal_error.numpy())
            normal_relative_error_list.append(normal_relative_error.numpy())

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_subplot(221)
        ax.plot(t_list.numpy(), tangent_error_list)
        ax.set_xlabel('t')
        ax.set_title('Tangent error vs t')

        ax = fig.add_subplot(222)
        ax.plot(t_list.numpy(), tangent_relative_error_list)
        ax.set_xlabel('t')
        ax.set_title('Tangent relative error vs t')
        
        ax = fig.add_subplot(223)
        ax.plot(t_list.numpy(), normal_error_list)
        ax.set_xlabel('t')
        ax.set_title('Normal error vs t')

        ax = fig.add_subplot(224)
        ax.plot(t_list.numpy(), normal_relative_error_list)
        ax.set_xlabel('t')
        ax.set_title('Normal relative error vs t')

        plt.suptitle('Error of scores')
        plt.savefig(self.savefig_dir + f"/error_score.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    @torch.no_grad()
    def plot_scores_error_in_normal_and_t(self):
        """
        scores_ana is NAN when c=0, t=0!!!
        """
        r_left, r_right = -0.1, 0.1
        t_list = torch.linspace(0., self.sde.T, 100 + 1)

        r_grid = torch.linspace(r_left, r_right, 50 + 1)
        grid_dense = 50
        x = torch.linspace(self.x_bound[0], self.x_bound[1], grid_dense)
        y = torch.linspace(self.y_bound[0], self.y_bound[1], grid_dense)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        mesh_ori = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1), torch.zeros(grid_dense**2, 1)), dim=1)

        error_list = torch.zeros(t_list.shape[0], r_grid.shape[0])
        error_tangent_list = torch.zeros(t_list.shape[0], r_grid.shape[0])

        for i, t_temp in enumerate(t_list):
            tt = torch.ones(mesh_ori.shape[0]) * t_temp

            for j, r in enumerate(r_grid):
                std = self.sde.marginal_prob(0, t_temp)[1]
                mask = torch.abs(r) > std * 2.5
                if mask:
                    error_list[i, j] = torch.nan
                    error_tangent_list[i, j] = torch.nan
                else:
                    mesh = mesh_ori + r * torch.tensor([[0.0, 0.0, 1.0]])
                    scores = self.score_net(mesh, tt)
                    scores_ana = self.get_ref_score(mesh, t_temp)

                    error_list[i, j] = torch.norm(scores-scores_ana, dim=1).mean()
                    error_tangent_list[i, j] = torch.norm(scores[:, :2]-scores_ana[:, :2], dim=1).mean()
        
        if self.config.save_plot_grid:
            error_tangent = error_tangent_list
            print(error_tangent, error_tangent.shape)
            np.save(f"{self.samples_dir}/{self.config.training.algo}_error_tangent.npy", error_tangent.cpu().numpy())
        
        fig, axs = plt.subplots(2, 1)

        ax = axs[0]
        cs = ax.imshow(error_list.T, interpolation='nearest', cmap='rainbow',
                        extent=[0, self.sde.T, r_left, r_right])
        cbar = fig.colorbar(cs, ax=ax)
        ax.set_title('Error')
        ax.set_xlabel('t')
        ax.set_ylabel('z')

        ax = axs[1]
        cs = ax.imshow(error_tangent_list.T, interpolation='nearest', cmap='rainbow',
                        extent=[0, self.sde.T, r_left, r_right])
        cbar = fig.colorbar(cs, ax=ax)
        ax.set_title('Error in tangent space')
        ax.set_xlabel('t')
        ax.set_ylabel('z')

        plt.tight_layout()
        plt.savefig(self.savefig_dir + f"/error_in_z-t_space.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def validate(self, epoch=0):
        if epoch < self.total_epochs * 0.6: return
        logging.info(f"-------------------------Start validating: Epoch {epoch}/{self.total_epochs}-------------------------")
        mode = 'Reverse-sde'
        samples = SDE_sampler_two_stage(self.config, self.score_net, self.sde, self.manifold,
                                        mode=mode, threshold=self.config.sample.sample_threshold)
        self.calculate_constrain(samples)
        samples = self.manifold.project_onto_manifold(samples)
        self.plot_sample(samples.cpu().numpy(), savefig=f'val_{epoch}_generated_{mode}')
        logging.info("-------------------------End validating.-------------------------")
    
        for t_temp in tqdm(torch.linspace(0., 0.1, 2)):
            self.plot_scores_manifolds(t_temp=t_temp, savefig=f'val_{epoch}_{t_temp :.2f}', device=self.device)
            self.plot_scores_norm(self.score_net, label=t_temp, savefig=f'val_{epoch}_{t_temp :.2f}', device=self.device)

    def test(self):
        logging.info('Start testing')
        device = torch.device('cpu') 
        self.network.to(device)

        logging.info('Plot force: ')
        for t_temp in tqdm(torch.linspace(0., 1., 10 + 1)):
            self.plot_scores_manifolds(t_temp=t_temp, savefig=f'{t_temp :.2f}', device=device)
            self.plot_scores_norm(self.score_net, label=t_temp, savefig=f'{t_temp :.2f}', device=device)
        
        self.plot_scores_error_in_normal_and_t()
        self.plot_scores_error_vs_t()
        
        return

    def generate_new_samples(self, mode, threshold=None):
        logging.info('----------------------------------------------------------')

        samples_list = []
        for _ in range(self.config.sample.sample_epoch):
            if mode == 'Reverse-ode':
                samples = ode_sampler(self.config, self.score_net, self.sde, self.manifold)
            else:
                samples = SDE_sampler_two_stage(self.config, self.score_net, self.sde, self.manifold,
                                                mode=mode, threshold=threshold)
            # self.calculate_constrain(samples)
            samples = self.manifold.project_onto_manifold(samples)
            samples_list.append(samples.detach().cpu())

        if self.config.if_cal_distri_dist:
            logging.info(f'Calculating distance: sampling mode: __{mode}__, training algorithm: __{self.config.training.algo}')
            self.cal_distri_dist_all_fn(samples_list, sample_mode=mode)
            # np.save(f"{self.samples_dir}/samples_{mode}.npy", samples.cpu().numpy())

        self.plot_sample(samples_list[0], savefig=mode)
        logging.info('----------------------------------------------------------')

    def data_for_cal_dist(self, A):
        return A[:, :2]

    def cal_distri_dist_all_fn(self, samples_list, sample_mode=None):
        logging.info(f"Start calculating 1-wasserstein distance")
        self.cal_distri_dist_fn(samples_list, mode="wasser1", sample_mode=sample_mode)
        logging.info(f"Start calculating 2-wasserstein distance")
        self.cal_distri_dist_fn(samples_list, mode="wasser2", sample_mode=sample_mode)
        logging.info(f"Start calculating MMD")
        self.cal_distri_dist_fn(samples_list, mode="mmd", sample_mode=sample_mode)



if __name__ == "__main__":
    pass


