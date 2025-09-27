

<h3 align="center">Improving the Euclidean Diffusion Generation of Manifold Data by Mitigating Score Function Singularity</h3>


## Experiments


### R2inR3

Execute the training process with:
```
python main.py --multirun experiment=R2inR3_9w \
    training.algo=vesde,vesde_noniso,vesde_projected,vesde_rescale,vesde_noniso_rescale,vesde_proj_rescale  \
    if_cal_distri_dist=True seed=0,1,2,3,4
```

### SOn

To generate the dataset, run the following command:

```
python data/get_SOn_data.py
```

Then, execute the training process with:

```
python main.py --multirun experiment=SO10_5w \
    training.algo=vesde,vesde_noniso,vesde_projected,vesde_rescale,vesde_noniso_rescale,vesde_proj_rescale  \
    if_cal_distri_dist=True seed=0,1,2,3,4
```

### Mesh Data

To generate the dataset, run the following command:

```
python data/get_mesh_data.py
```

Then, execute the training process with:

```
python main.py --multirun experiment=bunny_mix,spot_mix \
    training.algo=vesde,vesde_noniso,vesde_projected,vesde_rescale,vesde_noniso_rescale,vesde_proj_rescale  \
    if_cal_distri_dist=True seed=0,1,2,3,4
```

### Alanine Dipeptide

To generate the dataset, run the following command:

```
python data/get_dipeptide_data.py
```

Then, execute the training process with:

```
python main.py --multirun experiment=dipeptide_l \
    training.algo=vesde,vesde_noniso,vesde_projected,vesde_rescale,vesde_noniso_rescale,vesde_proj_rescale  \
    if_cal_distri_dist=True seed=0,1,2,3,4
```
