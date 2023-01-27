#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import wandb
import numpy as np
import ml_collections
from pathlib import Path
from tqdm.auto import tqdm
from functools import partial
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets import (
    get_dataset, data_transform, inverse_data_transform
)

from models import eval_models
from models.ema import EMAHelper
from models.unet import UNet_SMLD, UNet_DDPM
from models.fvd.fvd import (
    get_fvd_feats, frechet_distance, load_i3d_pretrained
)
from models import (
    ddpm_sampler,
    ddim_sampler,
    FPNDM_sampler,
    anneal_Langevin_dynamics,
    anneal_Langevin_dynamics_consistent,
    anneal_Langevin_dynamics_inpainting,
    anneal_Langevin_dynamics_interpolation
)
from models.better.ncsnpp_more import UNetMore_DDPM

from losses import get_optimizer, warmup_lr
from losses.dsm import anneal_dsm_score_estimation

from load_model_from_ckpt import init_samples as initialize_samples
from runners.ncsn_runner import conditioning_fn


# In[ ]:


def get_data_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.channels = 1
    config.dataset = 'StochasticMovingMNIST'
    config.gaussian_dequantization = False
    config.image_size = 64
    config.logit_transform = False
    config.num_digits = 2
    config.num_frames = 5
    config.num_frames_cond = 5
    config.num_frames_future = 0
    config.num_workers = 0
    config.prob_mask_cond = 0.0
    config.prob_mask_future = 0.0
    config.prob_mask_sync = False
    config.random_flip = True
    config.rescaled = True
    config.step_length = 0.1
    config.uniform_dequantization = False

    return config


def get_fast_fid_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.batch_size = 1000
    config.begin_ckpt = 5000
    config.end_ckpt = 300000
    config.ensemble = False
    config.freq = 5000
    config.n_steps_each = 0
    config.num_samples = 1000
    config.pr_nn_k = 3
    config.step_lr = 0.0
    config.verbose = False

    return config


def get_model_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.arch = 'unetmore'
    config.attn_resolutions = [8, 16, 32]
    config.ch_mult = [1, 2, 3, 4]
    config.cond_emb = False
    config.conditional = True
    config.depth = 'deep'
    config.dropout = 0.1
    config.ema = True
    config.ema_rate = 0.999
    config.gamma = False
    config.n_head_channels = 64
    config.ngf = 64
    config.noise_in_cond = False
    config.nonlinearity = 'swish'
    config.normalization = 'InstanceNorm++'
    config.num_classes = 1000
    config.num_res_blocks = 2
    config.output_all_frames = False
    config.sigma_begin = 0.02
    config.sigma_dist = 'linear'
    config.sigma_end = 0.0001
    config.spade = False
    config.spade_dim = 128
    config.spec_norm = False
    config.time_conditional = True
    config.type = 'v1'
    config.scheduler = 'DDPM'

    return config


def get_optim_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.amsgrad = False
    config.beta1 = 0.9
    config.eps = 1e-08
    config.grad_clip = 1.0
    config.lr = 0.0002
    config.optimizer = 'Adam'
    config.warmup = 1000
    config.weight_decay = 0.0

    return config


def get_sampling_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.batch_size = 100
    config.ckpt_id = 0
    config.clip_before = True
    config.consistent = True
    config.data_init = False
    config.denoise = True
    config.fid = False
    config.final_only = True
    config.fvd = True
    config.init_prev_t = -1.0
    config.inpainting = False
    config.interpolation = False
    config.max_data_iter = 100000
    config.n_interpolations = 15
    config.n_steps_each = 0
    config.num_frames_pred = 20
    config.num_samples4fid = 10000
    config.num_samples4fvd = 10000
    config.one_frame_at_a_time = False
    config.preds_per_test = 1
    config.ssim = True
    config.step_lr = 0.0
    config.subsample = 1000
    config.train = False

    return config


def get_test_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.batch_size = 100
    config.begin_ckpt = 5000
    config.end_ckpt = 300000

    return config


def get_training_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.L1 = False
    config.batch_size = 64
    config.checkpoint_freq = 100
    config.log_all_sigmas = False
    config.log_freq = 50
    config.n_epochs = 500
    config.n_iters = 3000001
    config.sample_freq = 50000
    config.snapshot_freq = 1000
    config.snapshot_sampling = True
    config.val_freq = 100
    config.checkpoint_dir = "smmnist_cat"
    config.checkpoint_freq = 50

    return config


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    
    config.data = get_data_configs()
    config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config.fast_fid = get_fast_fid_configs()
    config.model = get_model_configs()
    config.optim = get_optim_configs()
    config.sampling = get_sampling_configs()
    config.test = get_test_configs()
    config.training = get_training_configs()
    config.start_at = 0
    
    return config


# In[ ]:


def ls(path): 
    return sorted(list(path.iterdir()))


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# In[ ]:


config = get_config()
config_dict = config.to_dict()
config_dict.pop("device", None)

wandb.init(
    project="masked-conditional-video-diffusion",
    entity="wandb",
    job_type="test",
    config=config_dict
)

wandb_config = wandb.config

artifact = wandb.use_artifact('capecape/gtc/np_dataset:v0', type='dataset')
artifact_dir = artifact.download()


# In[ ]:


class CloudDataset:
    
    def __init__(self, files, num_frames=4, scale=True, size=64):
        self.num_frames = num_frames
        self.size = size
        self.tfms = T.Compose([
            T.Resize((size, int(size * 1.7))),
            T.CenterCrop(size)
        ])
        data = []
        for file in tqdm(files):
            one_day = np.load(file)
            one_day = 0.5 - self._scale(one_day) if scale else one_day
            wds = np.lib.stride_tricks.sliding_window_view(
                one_day.squeeze(), 
                num_frames, 
                axis=0
            ).transpose((0, 3, 1, 2))
            data.append(wds)
        self.data = np.concatenate(data, axis=0)
            
    @staticmethod
    def _scale(arr):
        "Scales values of array in [0,1]"
        m, M = arr.min(), arr.max()
        return (arr - m) / (M - m)
    
    def __getitem__(self, idx):
        data = self.tfms(torch.from_numpy(self.data[idx]))
        data = torch.unsqueeze(data, dim=-3)
        return data, data
    
    def __len__(self):
        return len(self.data)

    def save(self, fname="cloud_frames.npy"):
        np.save(fname, self.data)


# In[ ]:


def plot_results(images, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1)
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()


# In[ ]:


files = ls(Path(artifact_dir))
train_ds = CloudDataset(
    files,
    num_frames=config.data.num_frames + config.data.num_frames_cond,
    size=config.data.image_size
)


# In[ ]:


plot_results([
    torch.squeeze(train_ds[0][0]).numpy()[i]
    for i in range(config.data.num_frames + config.data.num_frames_cond)
])
plot_results([
    torch.squeeze(train_ds[0][1]).numpy()[i]
    for i in range(config.data.num_frames + config.data.num_frames_cond)
])


# In[ ]:


train_loader = DataLoader(
    train_ds,
    batch_size=config.training.batch_size,
    shuffle=True,
    num_workers=config.data.num_workers
)
x, y = next(iter(train_loader))
x.shape, y.shape


# In[ ]:


scorenet = UNetMore_DDPM(config).to(config.device)
scorenet = torch.nn.DataParallel(scorenet)
optimizer = get_optimizer(config, scorenet.parameters())

wandb.log({
    "Parameters": count_parameters(scorenet),
    "Trainable Parameters": count_trainable_parameters(scorenet)
}, commit=False)


# In[ ]:


if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"Number of GPUs : {num_devices}")
    for i in range(num_devices):
        print(torch.cuda.get_device_properties(i))
else:
    print(f"Running on CPU!")


# In[ ]:


if config.model.ema:
    ema_helper = EMAHelper(mu=config.model.ema_rate)
    ema_helper.register(scorenet)

net = scorenet.module if hasattr(scorenet, 'module') else scorenet


# In[ ]:


conditional = config.data.num_frames_cond > 0
cond, test_cond = None, None
future = getattr(config.data, "num_frames_future", 0)
n_init_samples = min(36, config.training.batch_size)
init_samples_shape = (
    n_init_samples,
    config.data.channels * config.data.num_frames,
    config.data.image_size,
    config.data.image_size
)


# In[ ]:


if config.model.scheduler == "SMLD":
    init_samples = data_transform(
        config,
        torch.rand(init_samples_shape, device=config.device)
    )
elif config.model.scheduler in ["DDPM", "DDIM", "FPNDM"]:
    if getattr(config.model, 'gamma', False):
        used_k, used_theta = net.k_cum[0], net.theta_t[0]
        z = torch.distributions.gamma(
            torch.full(init_samples_shape, used_k),
            torch.full(init_samples_shape, 1 / used_theta)
        ).sample().to(config.device)
        init_samples = z - used_k * used_theta
    else:
        init_samples = torch.randn(init_samples_shape, device=config.device)


# In[ ]:


if config.model.scheduler == "SMLD":
    consistent = getattr(config.sampling, 'consistent', False)
    sampler = anneal_Langevin_dynamics_consistent if consistent else anneal_Langevin_dynamics
elif config.model.scheduler == "DDPM":
    sampler = partial(ddpm_sampler, config=config)
elif config.model.scheduler == "DDIM":
    sampler = partial(ddim_sampler, config=config)
elif config.model.scheduler == "FPNDM":
    sampler = partial(FPNDM_sampler, config=config)


# In[ ]:


def train_step(x, y, step):
    optimizer.zero_grad()
    lr = warmup_lr(
        optimizer, step,
        getattr(config.optim, 'warmup', 0),
        config.optim.lr
    )
    scorenet.train()
    
    x = x.to(config.device)
    x = data_transform(config, x)
    x, cond, cond_mask = conditioning_fn(
        config, x, num_frames_pred=config.data.num_frames,
        prob_mask_cond=getattr(config.data, 'prob_mask_cond', 0.0),
        prob_mask_future=getattr(config.data, 'prob_mask_future', 0.0),
        conditional=conditional
    )
    
    loss = anneal_dsm_score_estimation(
        scorenet, x, labels=None, cond=cond, cond_mask=cond_mask,
        loss_type=getattr(config.training, 'loss_type', 'a'),
        gamma=getattr(config.model, 'gamma', False),
        L1=getattr(config.training, 'L1', False), hook=None,
        all_frames=getattr(config.model, 'output_all_frames', False)
    )
    loss.backward()
    
    grad_norm = torch.nn.utils.clip_grad_norm_(
        scorenet.parameters(), getattr(config.optim, 'grad_clip', np.inf)
    )
    optimizer.step()
    
    if config.model.ema:
        ema_helper.update(scorenet)
    
    return loss.item(), grad_norm.item(), lr


# In[ ]:


def validation_step(epoch):
    test_scorenet = ema_helper.ema_copy(scorenet) if config.model.ema else scorenet
    test_scorenet.eval()
    x, y = next(iter(test_loader))
    x = x.to(config.device)
    x = data_transform(config, x)
    x, test_cond, test_cond_mask = conditioning_fn(
        config, x, num_frames_pred=config.data.num_frames,
        prob_mask_cond=getattr(config.data, 'prob_mask_cond', 0.0),
        prob_mask_future=getattr(config.data, 'prob_mask_future', 0.0),
        conditional=conditional
    )
    with torch.no_grad():
        test_dsm_loss = anneal_dsm_score_estimation(
            test_scorenet, x, labels=None,
            cond=test_cond, cond_mask=test_cond_mask,
            loss_type=getattr(config.training, 'loss_type', 'a'),
            gamma=getattr(config.model, 'gamma', False),
            L1=getattr(config.training, 'L1', False), hook=None,
            all_frames=getattr(config.model, 'output_all_frames', False)
        )
        if wandb.run is not None:
            wandb.log({
                "validation/epoch": epoch,
                "validation/loss": test_dsm_loss.item(),
            }, step=epoch)


# In[ ]:


def save_model(scorenet, epoch):
    if not os.path.isdir(config.training.checkpoint_dir):
        os.makedirs(config.training.checkpoint_dir)
    states = [scorenet.state_dict(), optimizer.state_dict(), epoch, step]
    if config.model.ema:
        states.append(ema_helper.state_dict())
    checkpoint_path = os.path.join(config.training.checkpoint_dir, 'checkpoint.pt')
    torch.save(states, checkpoint_path)
    if wandb.run is not None:
        artifact = wandb.Artifact(
            f'checkpoint-{wandb.run.name}-{wandb.run.id}', type='model'
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact, aliases=["latest", f"epoch-{epoch}"])


# In[ ]:


step = 0

for epoch in range(1, config.training.n_epochs + 1):
    train_pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Training Epoch {epoch}"
    )
    for batch, (x, y) in train_pbar:
        loss, grad_norm, lr = train_step(x, y, step)
        if wandb.run is not None:
            wandb.log({
                "train/step": step,
                "lr": lr,
                "grad_norm": grad_norm,
                "train/loss": loss,
            }, step=step)
            step += 1
    if epoch % config.training.checkpoint_freq == 0:
        save_model(scorenet, epoch)


# In[ ]:


wandb.finish()

