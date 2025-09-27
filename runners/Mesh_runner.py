import logging
import torch
import numpy as np
from runners.Basic_runner import BasicRunner
from utils import split_dataset, check_memory, get_scene_dict
from sampling import SDE_sampler_two_stage, ode_sampler
import trimesh
import plotly.graph_objs as go
import plotly.offline as offline


class MeshRunner(BasicRunner):
    def __init__(self, config):
        super().__init__(config)
        self.config.data_seed = self.config.seed
        self.load_data()

        """---------------------------exhibit dataset--------------------------"""
        samples_test = self.training_set.detach()
        self.plot_sample(samples_test.cpu().numpy(), savefig="true")
        self.plot_histogram_on_surface(samples=samples_test.cpu().numpy(), savefig="true")

        if self.config.if_save_sample:
            np.save(f"{self.samples_dir}/samples_{self.config.training.algo}_true.npy", self.training_set[:self.config.sample.sample_num].numpy())


    def load_data(self):
        data_ori = torch.tensor(np.load(f"./data/{self.obj}/{self.dataset_name}.npy"))
        self.data_set = data_ori.clone()
        self.config.data_seed = self.config.seed
        
        self.training_set, self.test_set, _ = split_dataset(self.data_set, self.config.data_seed)
        self.filter_sample(self.test_set)
        check_memory()

        self.mesh = trimesh.load(f"./data/{self.obj}/{self.obj}_mesh_simple1.ply")
        self.scene_dict = get_scene_dict(self.obj)

        if self.config.if_cal_distri_dist:
            self.statistics_true = self.data_for_cal_dist(self.data_set)
            self.dist_dist_fn = None
            logging.info(f'Calculating distance: sampling mode: __true__, training algorithm: __{self.config.training.algo}')
            self.cal_distri_dist_all_fn([self.training_set[:self.config.sample.sample_num]], sample_mode="true")

    def plot_sample(self, samples, savefig=None):
        verts = self.mesh.vertices
        I, J, K = self.mesh.faces.transpose()

        trace = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, alphahull=5, opacity=0.4, color='cyan'),
                 go.Scatter3d(x=samples[:, 0], y=samples[:, 1], z=samples[:, 2], mode='markers', marker=dict(size=3))]
        fig = go.Figure(data=trace)
        fig.update_layout(title=f'{samples.shape[0]} scatters',
                          scene=self.scene_dict, width=1400, height=1400, showlegend=True)

        filename0 = self.savefig_dir + f"/Samples_{savefig}"
        offline.plot(fig, filename=f'{filename0}.html', auto_open=False)
        fig.write_image(f"{filename0}.png")

    def plot_histogram_on_surface(self, samples, colorscale=None, savefig=None):

        verts = self.mesh.vertices
        I, J, K = self.mesh.faces.transpose()

        closest_points, _, closest_faces = trimesh.proximity.closest_point(self.mesh, samples)
        unique_faces, counts = np.unique(closest_faces, return_counts=True)
        probs = np.zeros(len(self.mesh.faces))
        probs[unique_faces] = counts / len(samples)
        densities = probs / self.mesh.area_faces
        densities[np.isnan(densities)] = 0
        cmin, cmax = -0.1, np.percentile(densities, 95) if colorscale is None else colorscale

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='Samples_hist',
                            opacity=1.0, intensity=densities, intensitymode="cell", colorscale="Viridis",
                            cmin=cmin, cmax=cmax)]
        layout = go.Layout(title=f'Histgram of {samples.shape[0]} scatters', scene=self.scene_dict, width=1400, height=1400, showlegend=True)
        fig = go.Figure(data=traces, layout=layout)

        filename0 = self.savefig_dir + f"/Histgram_{savefig}"
        offline.plot(fig, filename=f'{filename0}.html', auto_open=False)
        fig.write_image(f"{filename0}.png")

    def filter_sample(self, samples):
        find_error = torch.isnan(samples) | (torch.abs(samples) > 2.)
        idx_error = torch.unique(torch.where(find_error)[0])
        mask = torch.ones(samples.shape[0], dtype=torch.bool)
        mask[idx_error] = False
        samples_new = samples[mask, ...]
        logging.info(f'{idx_error.shape[0]} of {samples.shape[0]} samples are dropped.')
        if idx_error.shape[0] == samples.shape[0]:
            logging.info(f'All the samples are dropped.')
            samples_new = torch.zeros_like(samples)
        return samples_new

    def validate(self, epoch=0):
        if epoch < self.total_epochs * 0.6: return
        logging.info(f"-------------------------Start validating: Epoch {epoch}/{self.total_epochs}-------------------------")
        mode = 'Reverse-sde'
        samples = SDE_sampler_two_stage(self.config, self.score_net, self.sde, self.manifold,
                            mode=mode, threshold=self.config.sample.sample_threshold)
        self.calculate_constrain(samples)

        samples = self.filter_sample(samples)
        samples = self.manifold.project_onto_manifold(samples)

        self.plot_sample(samples.cpu().numpy(), savefig=f'val_{epoch}_generated_{mode}')
        self.plot_histogram_on_surface(samples.cpu().numpy(), savefig=f'val_{epoch}_generated_{mode}')
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
            # self.calculate_constrain(samples)
            samples = self.filter_sample(samples)
            if mode != 'Corrector':
                samples = self.manifold.project_onto_manifold(samples)
            samples_list.append(samples.detach().cpu())

        if self.config.if_cal_distri_dist:
            logging.info(f'Calculating distance: sampling mode: __{mode}__, training algorithm: __{self.config.training.algo}')
            self.cal_distri_dist_all_fn(samples_list, sample_mode=mode)
            
        samples_plot = torch.cat(samples_list, dim=0)
        self.plot_sample(samples_plot.numpy(), savefig=mode)
        self.plot_histogram_on_surface(samples_plot.numpy(), savefig=mode)

        if self.config.if_save_sample: 
            np.save(f"{self.samples_dir}/samples_{self.config.training.algo}_{mode}.npy", samples_plot.numpy())
        logging.info('----------------------------------------------------------')

    def data_for_cal_dist(self, A):
        _, _, closest_faces = trimesh.proximity.closest_point(self.mesh, A)
        unique_faces, counts = np.unique(closest_faces, return_counts=True)
        probs = np.zeros(len(self.mesh.faces))
        probs[unique_faces] = counts / len(A)
        return probs

    def cal_distri_dist_all_fn(self, samples_list, sample_mode=None):
        logging.info(f"Start calculating TVD distance")
        self.cal_distri_dist_fn(samples_list, mode="TVD", sample_mode=sample_mode)
        logging.info(f"Start calculating JS distance")
        self.cal_distri_dist_fn(samples_list, mode="JS", sample_mode=sample_mode)
        # logging.info(f"Start calculating entropy distance")
        # self.cal_distri_dist_fn(samples_list, mode="entropy", sample_mode=sample_mode)


if __name__ == "__main__":
    pass
