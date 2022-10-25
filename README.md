# Leveraging Demonstrations with Latent Space Priors

This is the source code release for the paper [Leveraging Demonstrations with
Latent Space Priors](TODO LINK TO ARXIV). It contains pre-training code (VAE,
latent space prior, low-level policy) and code to train high-level policies with
different prior integrations. Pre-trained policies are provided as well (see
below).

## Prerequisites

Install PyTorch according to the [official
instructions](https://pytorch.org/get-started), for example in a new conda
environment. This code-base was tested with PyTorch 1.10 and 1.12. Python 3.9 is
required.

Install Python dependencies via
```sh
pip install -r requirements.txt
```

We provide a pre-trained low-level policy which was trained with MuJoCo 2.0. For
maximum compatibility, we recommend to perform a manual install of the 2.0.0
which is [available at roboti.us](https://www.roboti.us/download.html), along
with an unrestricted license key.

For pre-training, you will have to build the C++ extension, which in turn
requires `cmake` and `pybind11`. These can be installed via conda or your
system-wide package manager of choice. You'll also require the [CUDA
SDK](https://docs.nvidia.com/cuda/). The extension can be built as follows:
```sh
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release \
  -DMUJOCO_INCLUDE_DIR=${HOME}/.mujoco/mujoco200_linux/include \
  -DMUJOCO_LIBRARY=${HOME}/.mujoco/mujoco200_linux/bin/libmujoco200.so
make -C cpp/build
```
Naturally, the MuJoCo location in the above command might have to be adjusted.

For optimal performance, we also recommend installing NVidia's
[PyTorch extensions](https://github.com/NVIDIA/apex).


## Usage

**NOTE** Depending on your MuJoCo installation, rendering environments may
require additional configuration. For all the commands below, passing
`video=null eval.video=null` on the command-line disables video generation.


### With Pre-trained Models

Download and extract the pre-trained models:
```sh
wget https://dl.fbaipublicfiles.com/latent-space-priors/pretrained-models.tar.gz
tar xvzf pretrained-models.tar.gz
```

Experiments from the paper can be launched as follows:
```sh
# High-level policy training without a prior
python train.py -cn zprior_explore policy_lo=pretrained/policy-lo.pt prior=null
# z-prior Explore
python train.py -cn zprior_explore policy_lo=pretrained/policy-lo.pt prior=pretrained/prior.pt
# z-prior Regularize
python train.py -cn zprior_regularize policy_lo=pretrained/policy-lo.pt prior=pretrained/prior.pt
# z-prior Options 
python train.py -cn zprior_options policy_lo=pretrained/policy-lo.pt prior=pretrained/prior.pt
# Planning
python train.py -cn zprior_plan policy_lo=pretrained/policy-lo.pt prior=pretrained/prior.pt
```


### Pre-training

<details>
<summary>Mocap data preparation</summary>

You'll need to download the SMPL-H model from [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) and place it in `hucc/envs/assets` like so:

```
hucc/envs/assets/smplh/
├── LICENSE.txt
├── female
│   └── model.npz
├── info.txt
├── male
│   └── model.npz
└── neutral
    └── model.npz
```

The AMASS data can be obtained at
[amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/). First, extract the
`CMU.tar.bz2` archive to a location of choice. Afterwards, prepare a training
data file as follows:
```sh
find <path-to-cmu-clips> -name '*_poses.npz' | PYTHONPATH=$PWD python scripts/gen_amass_jpos.py -f - --subset cmu_locomotion_small --output-path lcs2-train.h5
find <path-to-cmu-clips> -name '*_poses.npz' | PYTHONPATH=$PWD python scripts/gen_amass_jpos.py -f - --subset cmu_locomotion_small_valid --output-path lcs2-valid.h5
python scripts/merge_h5_corpora.py --output-path lcs2.h5 --train lcs2-train --valid lcs2-valid --test ''
```
</details>

<details>
<summary>VAE training</summary>

Let's specify a dedicated job directory (`hydra.run.dir`) so that we can easily
refer to the resulting checkpoint and configuration.

```sh
python train_zprior.py -cn lcs2_vae dataset.path=$PWD/lcs2.h5 hydra.run.dir=vae
```
</details>

<details>
<summary>Prior training</summary>

As a first step, the training data file has to be augmented with latent states
and distributions from the VAE:

```sh
PYTHONPATH=$PWD python scripts/add_latents_to_hdf5.py --checkpoint vae/checkpoint.pt lcs2.h5 lcs2-with-latents.h5
```

The prior can then be trained as follows:
```sh
python train_zprior.py -cn lcs2_prior dataset.path=$PWD/lcs2-with-latents.h5 hydra.run.dir=prior
```
</details>

<details>
<summary>Low-level policy training</summary>

The low-level policy training code requires a dedicated data file which can be
produced as follows:

```sh
find <path-to-cmu-clips> -name '*_poses.npz' | PYTHONPATH=$PWD python scripts/gen_amass_mjbox.py -f - --subset cmu_locomotion_small --checkpoint vae/checkpoint.pt --output-path lcs2-lo.h5
```

Training the policy itself is computationally demanding. In our experiments, we
use 2 machines, each consisting of 80 CPUs and 8 GPUs. We provide several
configurations:
```sh
# Single-GPU training
python train.py -cn lcs2_policy_lo ref_path=lcs2-lo.h5 hydra.run.dir=policy_lo

# Multi-GPU training, single machine
python train_dist.py -cn lcs2_policy_lo ref_path=lcs2-lo.h5 agent.distributed.size=<num-gpus> hydra.run.dir=policy_lo
```

`train_dist.py` supports multi-machine training with slurm. Manual invocation is
also possible: the example below expects two 8-GPU machines.  A shared
directory is required to perform the initial rendez-vous. The following can then
be run on each machine (where `<machine-id>` is either 0 or 1):
```sh
# Multi-GPU training on 2 machines, on each machine:
SLURM_NODEID=<machine-id> SLURM_NNODES=2 SLURM_JOBID=<random-string> \
python train_dist.py -cn lcs2_policy_lo ref_path=lcs2-lo.h5 \
  agent.distributed.size=16 \
  agent.distributed.rdvu_path=<shared_directory> \
  hydra.run.dir=policy_lo
```
</details>


### Environments 

Environments can be selected with the `env.name` option. The table below shows
environment names as used in the paper:

Environment | `env.name`
--- | ---
GoToTargets | `BiskGoToTargetsC-v1`
Gaps | `BiskGapsC-v1`
Butterflies | `BiskButterfliesC-v1`
Stairs | `BiskStairsContC-v1`


## License
The majority of latent-space-priors is licensed under CC-BY-NC, however portions
of the project are available under separate license terms: the transformer
modeling code in hucc/models/prior.py is originally licensed under the Apache
2.0 license; the gym vector environment code in hucc/envs/thmp_vector_env.py is
license under the MIT license.
