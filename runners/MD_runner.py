import logging
from runners.Basic_runner import BasicRunner
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import split_dataset, check_memory, get_RMSD
from sampling import SDE_sampler_two_stage, ode_sampler
from cal_distri_dist import distance_Kabsch


class MDRunner(BasicRunner):
    def __init__(self, config):
        super().__init__(config)
        
        self.load_data()
        
        "---------------------------exhibit dataset--------------------------"
        samples_test = self.training_set[:self.config.sample.sample_num].detach()
        self.plot_sample_hist(samples_test.cpu().numpy(), savefig="training_set")
        self.plot_angle_and_RMSD_hist(samples_test.cpu().numpy(), savefig="training_set")
        
        samples_test1 = samples_test + torch.randn_like(samples_test) * self.config.model.sigma_min
        self.plot_sample_hist(samples_test1.cpu().numpy(), savefig="perturbed")
        self.plot_angle_and_RMSD_hist(samples_test1.cpu().numpy(), savefig="perturbed")
        
    def move_to_origin(self, samples):
        shape = samples.shape
        if samples.shape[-1] != 3:
            samples = samples.reshape(-1, self.natom, 3)
        samples = samples - samples.mean(dim=-2, keepdim=True)
        return samples.reshape(shape)

    def load_data(self):
        self.natom = 10
        self.xref = torch.tensor(np.load(f'./data/dipeptide/dipeptide_ref.npy')).float()
        self.x_center_1 = torch.tensor(np.load(f'./data/dipeptide/dipeptide_center_1.npy')).float()
        self.x_center_2 = torch.tensor(np.load(f'./data/dipeptide/dipeptide_center_2.npy')).float()
        
        data_ori = self.move_to_origin(torch.tensor(np.load(f'./data/dipeptide/dipeptide_long_refined.npy'))).float()  # shape: -1, 10 ,3
        self.data_set = data_ori.view(-1, 3 * self.natom)
        
        self.config.data_seed = self.config.seed
        
        self.training_set, self.test_set, _ = split_dataset(self.data_set, self.config.data_seed)
        check_memory()
        
        if self.config.if_cal_distri_dist:

            sample_true = self.test_set.clone()
            self.statistics_true = self.data_for_cal_dist(sample_true)
            self.dist_dist_fn = distance_Kabsch
            # logging.info(f'Calculating distance: sampling mode: __true__, training algorithm: __{self.config.training.algo}')
            # self.cal_distri_dist_all_fn([sample_true[:self.config.sample.sample_num]], sample_mode="true")

    def plot_sample_hist(self, samples, savefig=None):
        if not isinstance(samples, torch.Tensor): samples = torch.tensor(samples).float()
        phi, psi = self.manifold.angle_phi(samples), self.manifold.angle_psi(samples)
        phi_true, psi_true = self.manifold.angle_phi(self.data_set), self.manifold.angle_psi(self.data_set)
        fig = plt.figure(figsize=(10, 5))
        bins = int(phi.shape[0] / 100)

        ax = plt.subplot(1, 2, 1)
        ax.hist(phi.reshape(-1).numpy() / torch.pi * 180, bins=bins, alpha=0.5, density=True, histtype='stepfilled')
        ax.set_title("phi")

        ax = plt.subplot(1, 2, 2)
        ax.hist(psi.reshape(-1).numpy() / torch.pi * 180, bins=bins, alpha=0.5, density=True, histtype='stepfilled')
        ax.hist(psi_true.reshape(-1).numpy() / torch.pi * 180, bins=bins, alpha=0.5, density=True, histtype='stepfilled')
        ax.set_title("psi")

        plt.suptitle(f"Histogram for {samples.shape[0]} samples.")
        plt.savefig(self.savefig_dir + f"/Hist_psi_phi_{savefig}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_angle_and_RMSD_hist(self, samples, savefig=None):
        if not isinstance(samples, torch.Tensor): samples = torch.tensor(samples).float()
        samples = samples.reshape(-1, self.natom, 3)

        psi = self.manifold.angle_psi(samples)
        psi_true = self.manifold.angle_psi(self.data_set)
        psi_angle = psi.reshape(-1).numpy() / np.pi * 180
        psi_angle_true = psi_true.reshape(-1).numpy() / np.pi * 180

        psi_center_1 = self.manifold.angle_psi(self.x_center_1.unsqueeze(0)) / torch.pi * 180
        psi_center_2 = self.manifold.angle_psi(self.x_center_2.unsqueeze(0)) / torch.pi * 180

        RMSD1 = get_RMSD(samples, self.x_center_1).numpy()
        RMSD1_true = get_RMSD(self.data_set, self.x_center_1).numpy()

        RMSD2 = get_RMSD(samples, self.x_center_2).numpy()
        RMSD2_true = get_RMSD(self.data_set, self.x_center_2).numpy()

        fig = plt.figure(figsize=(15, 5))
        bins = int(psi.shape[0] / 100)

        ax = plt.subplot(1, 3, 1)
        ax.hist(psi_angle, bins=bins, alpha=0.5, density=True, color='green')
        ax.hist(psi_angle_true, bins=bins, alpha=0.5, density=True, color='red')

        ax.axvline(x=psi_center_1.reshape(-1).numpy(), color='black', linestyle='--')
        ax.axvline(x=psi_center_2.reshape(-1).numpy(), color='black', linestyle='--')
        ax.set_title("psi")

        ax = plt.subplot(1, 3, 2)
        ax.hist(RMSD1, bins=bins, alpha=0.5, density=True, color='green')
        ax.hist(RMSD1_true, bins=bins, alpha=0.5, density=True, color='red')
        ax.set_title("RMSD1")

        ax = plt.subplot(1, 3, 3)
        ax.hist(RMSD2, bins=bins, alpha=0.5, density=True, color='green')
        ax.hist(RMSD2_true, bins=bins, alpha=0.5, density=True, color='red')
        ax.set_title("RMSD2")

        plt.suptitle(f"Histogram for {samples.shape[0]} samples.")
        plt.savefig(self.savefig_dir + f"/Hist_angel_RMSD_{savefig}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def filter_sample(self, samples):
        RMSD = get_RMSD(samples, self.x_center_1.to(samples.device))
        constrain = self.manifold.constrain_fn(samples)
        find_error = (constrain.reshape(-1) > 1e-5) | (RMSD > 1.4)
        idx_error = torch.unique(torch.where(find_error)[0])
        mask = torch.ones(samples.shape[0], dtype=torch.bool)
        mask[idx_error] = False
        samples_new = samples[mask, ...]
        logging.info(f'{idx_error.shape[0]} of {samples.shape[0]} samples are dropped.')
        if idx_error.shape[0] == samples.shape[0]:
            logging.info(f'All the samples are dropped.')
            return torch.zeros_like(samples)
        return samples_new

    def validate(self, epoch=0):
        if epoch < self.total_epochs * 0.6: return
        logging.info(f"-------------------------Start validating: Epoch {epoch}/{self.total_epochs}-------------------------")
        mode = 'Reverse-sde'
        samples = SDE_sampler_two_stage(self.config, self.score_net, self.sde, self.manifold,
                                        mode=mode, threshold=self.config.sample.sample_threshold)
        self.calculate_constrain(samples)
        samples = self.manifold.project_onto_manifold(samples).cpu().numpy()
        self.plot_angle_and_RMSD_hist(samples, savefig=f'val_{epoch}_generated_{mode}')
        self.plot_sample_hist(samples, savefig=f'val_{epoch}_generated_{mode}')
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

        self.plot_angle_and_RMSD_hist(samples_list[0], savefig=mode)
        
        if self.config.if_save_sample:
            np.save(f"{self.samples_dir}/samples_{mode}_{self.config.seed}.npy", samples.detach().cpu().numpy())
        logging.info('----------------------------------------------------------')

    def data_for_cal_dist(self, A):
        return A

    def cal_distri_dist_all_fn(self, samples_list, sample_mode=None):
        logging.info(f"Start calculating 1-wasserstein distance")
        self.cal_distri_dist_fn(samples_list, mode="wasser1", sample_mode=sample_mode)
        logging.info(f"Start calculating 2-wasserstein distance")
        self.cal_distri_dist_fn(samples_list, mode="wasser2", sample_mode=sample_mode)


if __name__ == "__main__":

    pass
