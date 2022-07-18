import os.path as osp
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pretty_errors
import forge
from forge import flags
import forge.experiment_tools as fet
from forge.experiment_tools import fprint

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.utils import compute_performance, batch_statistic, mIOU
from datasets.unity_object_room import S3DISDataset

torch.set_printoptions(threshold=3000, linewidth=200)

def main_flags():

    flags.DEFINE_boolean('resume', True, 'Tries to resume a job if True.')

    flags.DEFINE_integer('report_loss_every', 100, 'Number of iterations between reporting minibatch loss.')

    flags.DEFINE_integer('run_validation_every', 5000, 'Number of iterations between running validation.')

    flags.DEFINE_integer('dash_plot_every', 100, 'Number of iterations between dash point cloud visualization')

    flags.DEFINE_integer('tfb_update_every', 50, 'Number of iterations between tfb update')

    flags.DEFINE_integer('ckpt_freq', 5000, 'Number of iterations between saving model checkpoints')

    flags.DEFINE_integer('num_test', 128, 'Number of iterations to test model')

    flags.DEFINE_integer('train_iter', 100000, 'Number of training iterations.')

    flags.DEFINE_string('results_dir', '/mnt/checkpoints/SPAIR3D', 'Top directory for all experimeontal results.')

    flags.DEFINE_string('run_name', 'S3DIS', 'Name of this job and name of results folder.')

    flags.DEFINE_boolean('multi_gpu', False, 'Use multiple GPUs if available.')

    flags.DEFINE_string('data_config', 'datasets/unity_object_room.py', 'Path to a data config file.')

    flags.DEFINE_string('model_config', 'models/SS3D.py', 'Path to a model config file.')

    flags.DEFINE_integer('batch_size', 1, 'Batch size. Point cloud operation does not support batch size > 1')

    flags.DEFINE_float('grad_max_norm', 1, 'Clip gradient of norm larger than 1.')

# max_radius = 1.25/16

main_flags()

config = forge.config()
logdir = osp.join(config.results_dir, config.run_name)
logdir, resume_checkpoint = fet.init_checkpoint(logdir, config.data_config, config.model_config, ["models/submodels/spair3d_modules.py"], config.resume)
checkpoint_name = osp.join(logdir, 'model.ckpt')

dash_logdir = osp.join(logdir, 'dash_data')
if not osp.exists(dash_logdir):
    os.mkdir(dash_logdir)

dataset = S3DISDataset("/mnt/S3DIS/")
train_loader = DataLoader(dataset, pin_memory=True, shuffle=True)

iter_idx = 0

writter = SummaryWriter(logdir)

model = fet.load(config.model_config, config).to("cuda:0")

# model = SS3D_SPAIR(max_radius)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, amsgrad=True)
# optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.0001)

if resume_checkpoint is not None:
    fprint(f"Restoring checkpoint from {resume_checkpoint}")

    checkpoint = torch.load(resume_checkpoint, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimiser_state_dict'])

    iter_idx = checkpoint['iter_idx'] + 1

model.iter = iter_idx

fprint(f"Starting training at iter = {iter_idx}")

model.train()

# while iter_idx <= config.train_iter:

while True:

    for pos, Id, batch, Layer in train_loader: # xyz, rgb, batch, Id, layer

        pos = pos.to('cuda:0')
        batch = batch.to('cuda:0')

        pos = pos[0]*4/3
        batch = batch[0]
        Id = Id[0]
        Layer = Layer[0]
        
        optimizer.zero_grad()

        # point clouds have varying sizes, in case one used up all the VRAM
        try:

            (NLL_forward, NLL_backward, DKL, 
            glimpse__size, 
            glimpse__batch, 
            glimpse__center, 
            glimpse__voxel_center,
            _,                        # glimpse z_what
            _,                        # glimpse_z_mask
            glimpse__log_z_pres,
            glimpse__logit_pres,
            glimpse__center_diff,     # glimpse__center_log_likelihood,
            glimpse_member__log_mask,
            glimpse_member__local_pos,
            glimpse_member__normalized_log_alpha,
            glimpse_member__batch, 
            glimpse_member__glimpse_index,
            glimpse_member__point_index,
            glimpse_predict__local_pos, 
            glimpse_predict__glimpse_index, 
            bg_predict__pos, 
            bg_predict__batch,
            bg_log_alpha,
            NLL_backward_fg,
            NLL_backward_bg,
            glimpse_chamfer_predict__local_pos,
            bg_chamfer_predict__pos) = model(pos, None, None, batch)

            NLL = NLL_forward + NLL_backward
            # NLL = NLL_forward

            loss = NLL + DKL

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = config.grad_max_norm)

            optimizer.step()

        except RuntimeError:

            continue

        if iter_idx % config.dash_plot_every == 0:
            segmentation, ARI, sc, msc, _, _ = compute_performance(dash_logdir, iter_idx, config.max_radius,
                                                            Id, pos, batch,
                                                            glimpse__batch, 
                                                            glimpse__center, 
                                                            glimpse_member__normalized_log_alpha,
                                                            glimpse_member__batch, 
                                                            glimpse_member__glimpse_index,
                                                            glimpse_member__point_index,
                                                            glimpse_predict__glimpse_index, 
                                                            bg_log_alpha,
                                                            glimpse_chamfer_predict__local_pos,
                                                            bg_chamfer_predict__pos)

            msc_1, N_1 = mIOU(Id = Id.cuda(), Seg = torch.tensor(segmentation).cuda(), Layer = Layer.cuda(), cls = 1)
            msc_2, N_2 = mIOU(Id = Id.cuda(), Seg = torch.tensor(segmentation).cuda(), Layer = Layer.cuda(), cls = 2)
            msc_3, N_3 = mIOU(Id = Id.cuda(), Seg = torch.tensor(segmentation).cuda(), Layer = Layer.cuda(), cls = 3)

            fprint(f"{iter_idx}: mIoU={(msc_1/N_1 if N_1 > 0 else 0):.4f}, mIoU={(msc_2/N_2) if N_2 > 0 else 0:.4f}, mIoU={(msc_3/N_3) if N_3 > 0 else 0:.4f}")
            fprint(f"{iter_idx}: N={N_1}, N={N_2}, N={N_3}")

        z_pres = None
        D_center = None

        if iter_idx % config.tfb_update_every == 0:
            z_pres = batch_statistic(torch.exp(glimpse__log_z_pres), glimpse__batch)
            # L_center = batch_statistic(torch.exp(glimpse__center_log_likelihood), glimpse__batch)
            D_center = torch.mean(glimpse__center_diff)
            bg_alpha = torch.mean(torch.exp(bg_log_alpha))
            writter.add_scalar('loss', loss.item(), iter_idx)
            writter.add_scalar('NLL', NLL.item(), iter_idx)
            writter.add_scalar('NLL_forward', NLL_forward.item(), iter_idx)
            writter.add_scalar('NLL_backward', NLL_backward.item(), iter_idx)
            writter.add_scalar('DKL', DKL.item(), iter_idx)
            writter.add_scalar('z_pres', z_pres.item(), iter_idx)
            writter.add_scalar('fg_alpha', 1 - bg_alpha.item(), iter_idx)
            writter.add_scalar('NLL_backward_fg', NLL_backward_fg.item(), iter_idx)
            writter.add_scalar('NLL_backward_bg', NLL_backward_bg.item(), iter_idx)
            writter.add_scalar('D_center', D_center, iter_idx)


        if iter_idx % config.report_loss_every == 0:
            if z_pres is None:
                z_pres = batch_statistic(torch.exp(glimpse__log_z_pres), glimpse__batch)
            if D_center is None:
                D_center = torch.mean(glimpse__center_diff)

            fprint(f"{iter_idx}: NLL={NLL.item():.4f},",
            f" NLL_F={NLL_forward.item():.4f},",
            f" NLL_B={NLL_backward.item():.4f},",
            f" NLL_backward_fg={NLL_backward_fg.item():.4f}",
            f" NLL_backward_bg={NLL_backward_bg.item():.4f}",
            f" DKL={DKL.item():.4f},",
            f" Loss={loss.item():.4f},",
            )

        if iter_idx % config.ckpt_freq == 0:

            ckpt_file = '{}-{}'.format(checkpoint_name, iter_idx)
            fprint(f"Saving model training checkpoint to: {ckpt_file}")
            model_state_dict = model.state_dict()
            ckpt_dict = {'iter_idx': iter_idx, 'model_state_dict': model_state_dict, 'optimiser_state_dict': optimizer.state_dict(), 'elbo': loss.item()}
            torch.save(ckpt_dict, ckpt_file)
            writter.close()

        if iter_idx + 1 % config.ckpt_freq == 0:
            writter = SummaryWriter(logdir)

        iter_idx += 1
