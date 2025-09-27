import logging
import matplotlib.pyplot as plt
import torch
import numpy as np
from runners.Basic_runner import BasicRunner
from utils import split_dataset
from sampling import SDE_sampler_two_stage, ode_sampler


class SOnRunner(BasicRunner):
    def __init__(self, config):
        super().__init__(config)
        self.load_data()

        """---------------------------------------exhibit dataset----------------------------------------"""
        test_set_samples = self.training_set[:self.config.sample.sample_num]
        self.test_set_statistics = self.get_statistics(test_set_samples).cpu().numpy()
        self.plot_hist(self.test_set_statistics, savefig="true")
        if self.config.if_save_sample: 
            np.save(f"{self.samples_dir}/statistics_true.npy", self.test_set_statistics)

    def load_data(self):
        self.power_list = [1, 2, 4, 5]
        data_ori = torch.tensor(np.load(f"./data/SOn/{self.dataset_name}.npy")).reshape(-1, self.manifold.out_dim)
        self.data_set = data_ori.clone()

        self.config.data_seed = self.config.seed
        self.training_set, self.test_set, _ = split_dataset(self.data_set, self.config.data_seed)

        if self.config.if_cal_distri_dist:
            self.statistics_true = self.data_for_cal_dist(self.test_set[:self.config.sample.sample_num])
            self.dist_dist_fn = None
            
            logging.info(f'Calculating distance: sampling mode: __true__, training algorithm: __{self.config.training.algo}')
            statistics = self.data_for_cal_dist(self.training_set[:self.config.sample.sample_num])
            self.cal_distri_dist_all_fn([statistics], sample_mode="true")

    def filter_sample(self, samples):
        indices = torch.where(samples.abs() > 1.01)
        if len(indices[0]) > 0:
            explode_index = torch.unique(indices[0], sorted=False)
            mask = torch.ones(samples.shape[0], dtype=torch.bool)
            mask[explode_index] = False
            samples = samples[mask, ...]
        all_mat = samples.reshape(-1, self.manifold.mat_dim, self.manifold.mat_dim)
        manifolds_idx = torch.where(torch.linalg.det(all_mat) > 0)[0]
        logging.info(f"The number of samples on the correct connected component: {manifolds_idx.shape[0]}/{samples.shape[0]}, the others are dropped.")
        return samples[manifolds_idx]
    
    def get_statistics(self, samples):
        if not isinstance(samples, torch.Tensor): samples = torch.tensor(samples)
        samples = samples.reshape(-1, self.manifold.mat_dim, self.manifold.mat_dim)
        trace_list = []
        for i in range(4):
            samples_pow = torch.matrix_power(samples, self.power_list[i])
            trace = samples_pow.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
            trace_list.append(trace)
        return torch.cat(trace_list, dim=1)

    def plot_hist(self, statistics, savefig=None, compare=True):
        bins = int(statistics.shape[0]/100)
        fig = plt.figure(figsize=(10, 10))
        for i in range(4):
            ax = plt.subplot(2, 2, i + 1)
            if compare:
                ax.hist(statistics[:, i], bins=bins, histtype='stepfilled', alpha=0.5, density=True, color='green')
                ax.hist(self.test_set_statistics[:, i], bins=bins, histtype='stepfilled', alpha=0.5, density=True, color='red')
            else:
                ax.hist(statistics[:, i], bins=bins, alpha=1.0, density=True)
            ax.set_title(rf"$tr(S^{self.power_list[i]})$")

        plt.suptitle(f"Histogram of {statistics.shape[0]} samples")
        plt.savefig(self.savefig_dir + f"/Hist_Statistics_{savefig}.png", dpi=300)
        plt.close(fig)

    def validate(self, epoch=0):
        if epoch < self.total_epochs * 0.6: return
        logging.info(f"-------------------------Start validating: Epoch {epoch}/{self.total_epochs}-------------------------")
        mode = 'Reverse-sde'
        samples = SDE_sampler_two_stage(self.config, self.score_net, self.sde, self.manifold,
                                        mode=mode, threshold=self.config.sample.sample_threshold)
        self.calculate_constrain(samples)
        samples = self.manifold.project_onto_manifold(samples)
        samples = self.filter_sample(samples)

        statistics = self.get_statistics(samples).cpu().numpy()
        self.plot_hist(statistics, savefig=f'val_{epoch}_generated_{mode}')
        logging.info("-------------------------End validating.-------------------------")

    def test(self):
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
            if mode != 'Corrector':
                samples = self.manifold.project_onto_manifold(samples)

            samples = self.filter_sample(samples)
            # self.calculate_constrain(samples)
            samples_list.append(samples.detach().cpu())

        if self.config.if_cal_distri_dist:
            logging.info(f'Calculating distance: sampling mode: __{mode}__, training algorithm: __{self.config.training.algo}')
            self.cal_distri_dist_all_fn(samples_list, sample_mode=mode)

        statistics = self.get_statistics(samples_list[0]).numpy()
        self.plot_hist(statistics, savefig=mode)

        if self.config.if_save_sample:
            np.save(f"{self.samples_dir}/samples_{mode}_{self.config.seed}.npy", samples.cpu().numpy())
        logging.info('----------------------------------------------------------')

    def data_for_cal_dist(self, A):
        return A.reshape(-1, 100)

    def cal_distri_dist_all_fn(self, samples_list, sample_mode=None):
        logging.info(f"Start calculating sliced wasserstein distance")
        self.cal_distri_dist_fn(samples_list, mode="sliced_wasser", sample_mode=sample_mode)


if __name__ == "__main__":

    pass


