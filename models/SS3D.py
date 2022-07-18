import torch
import torch.nn.functional as F

from forge import flags

from torch.distributions.kl import kl_divergence
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal

from torch_geometric.nn import knn, radius
from torch_geometric.nn.pool.consecutive import consecutive_cluster

from torch_scatter import scatter_sum, scatter_log_softmax, scatter_logsumexp

from models.submodels.spair3d_modules import (SPAIRPointFeatureNetwork, SPAIRGridFeatureNetwork, SPAIRGlimpseVAE, CoarseVAE)

from models.utils import diagonal_distance, linear_anneal

flags.DEFINE_float('max_radius', 1/8, 'max radius of glimpse ball or glimpse box')
flags.DEFINE_float('min_radius', 0.25/8, 'min radius of glimpse ball or glimpse box')
flags.DEFINE_float('grid_size', 1/8, 'voxel grid size')
flags.DEFINE_float('glimpse_center_offset_max_ratio', 1, 'glimpse_center_offset_max/grid_size')
flags.DEFINE_string('glimpse_type', 'box', '{box, ball}')

# soft boundary loss
flags.DEFINE_float('boundary_ratio', 0.5, 'the width ratio of the soft boundary')
flags.DEFINE_string('boundary_func', 'linear', '{linear, parabola}')

flags.DEFINE_string('alpha_temp_nmpq', '(10, 10, 0, 15000)', 'annealing parameter in the form of (n, m, p, q)')
flags.DEFINE_string('pos_reconstruct_sigma_nmpq', '(0.1, 0.01, 10000, 15000)', 'annealing parameter in the form of (n, m, p, q)')

flags.DEFINE_string('pres_prior_nmpq', '(0.01, 0.0001, 0, 15000)', 'annealing parameter in the form of (n, m, p, q)')
flags.DEFINE_string('pres_temp_nmpq', '(2.5, 0.5, 0, 10000)', 'annealing parameter in the form of (n, m, p, q)')
flags.DEFINE_string('radius_nmpq', '(2, -1, 10000, 20000)', 'annealing parameter in the form of (n, m, p, q)')

flags.DEFINE_string('pres_kl_weight_nmpq', '(5, 5, 0, 25000)', 'annealing parameter in the form of (n, m, p, q)')
flags.DEFINE_string('pos_kl_weight_nmpq', '(1, 1, 0, 10000)', 'annealing parameter in the form of (n, m, p, q)')
flags.DEFINE_string('size_ratio_kl_weight_nmpq', '(0.5, 0.5, 0, 10000)', 'annealing parameter in the form of (n, m, p, q)')
flags.DEFINE_string('what_mask_kl_weight_nmpq', '(1, 1, 0, 10000)', 'annealing parameter in the form of (n, m, p, q)')

flags.DEFINE_float('fg_distance_scale', 1/8, 'foreground distance scale for likleihood calculation')
flags.DEFINE_float('bg_distance_scale', 1, 'background distance scale for likleihood calculation')

flags.DEFINE_boolean('fix_boundary', True, 'symmetical boundary weight')
flags.DEFINE_boolean('fix_scale', False, 'fix scale')
flags.DEFINE_boolean('test_boundary_weight', False, 'symmetical boundary weight')

flags.DEFINE_string('radius_sigma_nmpq', '(0.5, 0.5, 0, 20000)', 'annealing parameter in the form of (n, m, p, q)')
flags.DEFINE_string('pos_offset_sigma_nmpq', '(0.5, 0.5, 0, 20000)', 'annealing parameter in the form of (n, m, p, q)')

# grid encoder
flags.DEFINE_boolean('grid_encoder_ar', True, 'flag for auto registration layer')
flags.DEFINE_boolean('grid_encoder_ln', True, 'flag for layer norm')

# glimpse VAE
flags.DEFINE_float('extra_predict_ratio', 0, 'the number of prediction / the number of ground truth - 1')
flags.DEFINE_boolean('glimpse_encoder_ar', True, 'flag for auto registration layer')
flags.DEFINE_boolean('glimpse_encoder_ln', True, 'flag for layer norm')

def load(cfg):
    return SS3D_SPAIR_v1(cfg)


class SS3D_SPAIR_v1(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.iter = 0

        self.glimpse_type = cfg.glimpse_type # {ball, box}
        self.fix_boundary = cfg.fix_boundary
        self.fix_scale = cfg.fix_scale

        self.boundary_ratio = cfg.boundary_ratio
        self.boundary_func = cfg.boundary_func
        self.test_boundary_weight = cfg.test_boundary_weight

        self.point_feature_network = SPAIRPointFeatureNetwork()

        self.grid_encoder = SPAIRGridFeatureNetwork(cfg)

        self.glimpse_vae = SPAIRGlimpseVAE(cfg)

        self.coarse_vae = CoarseVAE()

        self.grid_size = cfg.grid_size

        self.glimpse_center_offset_max = self.grid_size * cfg.glimpse_center_offset_max_ratio

        self.ball_radius_max = cfg.max_radius
        self.ball_radius_min = cfg.min_radius

        self.box_size_max = cfg.max_radius
        self.box_size_min = cfg.min_radius
        self.box_radius = diagonal_distance(self.box_size_max)   # half of the diagonal distance of box_size_max

        self.register_buffer("device", torch.tensor(0))

        self.standard_normal = None

        self.center_offset_prior = None
        self.pres_prior = None
        self.radius_ratio_prior = None
        self.pos_reconstruct_prob = None
        self.center_prob = None

        self.pres_temperature = 1.0
        self.alpha_temperature = 100

        self.pres_kl_weight = 1.0
        self.pos_kl_weight = 1.0
        self.size_ratio_kl_weigh = 1.0
        self.what_mask_kl_weight = 1.0

        self.pres_prior_nmpq = eval(cfg.pres_prior_nmpq)
        self.pres_temp_nmpq = eval(cfg.pres_temp_nmpq)
        self.radius_nmpq = eval(cfg.radius_nmpq)
        self.alpha_temp_nmpq = eval(cfg.alpha_temp_nmpq)
        self.pos_reconstruct_sigma_nmpq = eval(cfg.pos_reconstruct_sigma_nmpq)

        self.radius_sigma_nmpq = eval(cfg.radius_sigma_nmpq)
        self.pos_offset_sigma_nmpq = eval(cfg.pos_offset_sigma_nmpq)

        self.pres_kl_weight_nmpq = eval(cfg.pres_kl_weight_nmpq)
        self.pos_kl_weight_nmpq = eval(cfg.pos_kl_weight_nmpq)
        self.size_ratio_kl_weight_nmpq = eval(cfg.size_ratio_kl_weight_nmpq)
        self.what_mask_kl_weight_nmpq = eval(cfg.what_mask_kl_weight_nmpq)

        self.fg_distance_scale = cfg.fg_distance_scale
        self.bg_distance_scale = cfg.bg_distance_scale

    def anneal_parameters(self):

        if self.standard_normal is None:
            self.standard_normal = Normal(torch.tensor(0, device=self.device.device), torch.tensor(1, device=self.device.device))

        pos_offset_sigma_flag, pos_offset_sigma = linear_anneal(*self.pos_offset_sigma_nmpq, self.iter)

        if self.center_offset_prior is None or pos_offset_sigma_flag:
            self.center_offset_prior = Normal(torch.tensor(0, device=self.device.device), torch.tensor(pos_offset_sigma, device=self.device.device))

        pres_prior_flag, pres_beta = linear_anneal(*self.pres_prior_nmpq, self.iter)

        if self.pres_prior is None or pres_prior_flag:
            self.pres_prior = Bernoulli(torch.tensor(pres_beta, device=self.device.device))

        radius_mu_flag, radius_mu = linear_anneal(*self.radius_nmpq, self.iter)
        radius_sigma_flag, radius_sigma = linear_anneal(*self.radius_sigma_nmpq, self.iter)

        if self.radius_ratio_prior is None or radius_mu_flag or radius_sigma_flag:
            self.radius_ratio_prior = Normal(torch.tensor(radius_mu, device=self.device.device), torch.tensor(radius_sigma, device=self.device.device))

        pos_reconstruct_flag, pos_reconstruct_sigma = linear_anneal(*self.pos_reconstruct_sigma_nmpq, self.iter) 

        if self.pos_reconstruct_prob is None or pos_reconstruct_flag:
            self.pos_reconstruct_prob = Normal(torch.tensor(0, device=self.device.device), torch.tensor(pos_reconstruct_sigma, device=self.device.device))

        _, self.pres_temperature = linear_anneal(*self.pres_temp_nmpq, self.iter)
        _, self.alpha_temperature = linear_anneal(*self.alpha_temp_nmpq, self.iter)

        _, self.pres_kl_weight = linear_anneal(*self.pres_kl_weight_nmpq, self.iter)
        _, self.pos_kl_weight = linear_anneal(*self.pos_kl_weight_nmpq, self.iter)
        _, self.size_ratio_kl_weight = linear_anneal(*self.size_ratio_kl_weight_nmpq, self.iter)
        _, self.what_mask_kl_weight = linear_anneal(*self.what_mask_kl_weight_nmpq, self.iter)

        if self.center_prob is None:
            self.center_prob = Normal(torch.tensor(0.0, device=self.device.device), torch.tensor(0.3989423, device = self.device.device))

        if self.training:
            self.iter += 1

    def take_glimpse_ball(self, pos, rgb, batch, glimpse__center, voxel__center, glimpse__ball_radius_ratio, glimpse__center_offset_ratio, glimpse__batch):

        point_index = torch.arange(pos.size(0), dtype=torch.long, device=pos.device)

        # to overcome the fact that radius operation does not support different glimpse-wise radius, take glimpse with the maximum of radius first
        glimpse_member__glimpse_index, neighbor_index = radius(pos, glimpse__center, self.ball_radius_max * (1 + self.boundary_ratio), batch, glimpse__batch, max_num_neighbors=1024)

        glimpse_member__global_pos = pos[neighbor_index] 

        glimpse_member__batch = batch[neighbor_index]

        glimpse_member__point_index = point_index[neighbor_index]
        ###############################################################################

        # TODO: if rotation needs to be applied, apply here.

        glimpse_member__glimpse_center = glimpse__center[glimpse_member__glimpse_index]
        glimpse_member__local_pos = glimpse_member__global_pos - glimpse_member__glimpse_center

        # compute the glimpse-wise radius
        glimpse__ball_radius = (self.ball_radius_max - self.ball_radius_min) * glimpse__ball_radius_ratio + self.ball_radius_min

        glimpse_member__ball_radius = glimpse__ball_radius[glimpse_member__glimpse_index]

        glimpse_member__local_pos = glimpse_member__local_pos / glimpse_member__ball_radius

        ###############################################################################

        # compute 2 norm to get ball
        glimpse_member__local_euclid_norm = torch.norm(glimpse_member__local_pos, 2, dim = 1)
        
        # the definition of glimpse now include boundarys
        index = (glimpse_member__local_euclid_norm < (1 + self.boundary_ratio))
        
        glimpse_member__local_euclid_norm = glimpse_member__local_euclid_norm[index]
        glimpse_member__local_pos = glimpse_member__local_pos[index]
        glimpse_member__batch = glimpse_member__batch[index]
        glimpse_member__point_index = glimpse_member__point_index[index]
        glimpse_member__ball_radius = glimpse_member__ball_radius[index]
        glimpse_member__glimpse_index = glimpse_member__glimpse_index[index]

        if self.boundary_ratio > 0:
            glimpse_member__boundary_distance = glimpse_member__local_euclid_norm - 1.0
            # * torch.log1p is more accurate when input is close to zero (log(1)) but less accurate when input is close to -1 (log(0)) compared with torch.log
            ratio = glimpse_member__boundary_distance/self.boundary_ratio
            # assert torch.all(ratio >= 0)
            # assert torch.all(ratio <= 1)
            glimpse_member__log_boundary_weight = torch.log(1 - ratio + 1e-12)
        else:
            glimpse_member__log_boundary_weight = torch.zeros_like(glimpse_member__local_euclid_norm)

        # assert glimpse_member__log_boundary_weight.size(0) == glimpse_member__local_pos.size(0)
        if not self.fix_scale:
            glimpse_member__local_pos = glimpse_member__local_pos * glimpse_member__ball_radius / self.ball_radius_max

        # make glimplse_member__glimpse_index consequtive, sorting involved, might be slow
        # * return glimpse_member__local_pos, None, glimpse_member__glimpse_index, glimpse_member__batch, glimpse_member__point_index, glimpse__ball_radius, glimpse__batch
        # ! after take_glimpse all glimpse contains no points will be ignored
        glimpse_member__consecutive_glimpse_index, perm = consecutive_cluster(glimpse_member__glimpse_index)
        glimpse__consecutive_to_nonconsecutive_glimpse_index = glimpse_member__glimpse_index[perm]
        glimpse_consecutive__center = glimpse__center[glimpse__consecutive_to_nonconsecutive_glimpse_index]
        glimpse_consecutive__center_offset_ratio = glimpse__center_offset_ratio[glimpse__consecutive_to_nonconsecutive_glimpse_index]
        glimpse_consecutive__voxel_center = voxel__center[glimpse__consecutive_to_nonconsecutive_glimpse_index]
        glimpse_consecutive__batch = glimpse__batch[glimpse__consecutive_to_nonconsecutive_glimpse_index]
        glimpse_consecutive__ball_radius = glimpse__ball_radius[glimpse__consecutive_to_nonconsecutive_glimpse_index]

        return (glimpse_member__local_pos, None, 
                glimpse_member__consecutive_glimpse_index, 
                glimpse_member__point_index,
                glimpse_member__log_boundary_weight,
                glimpse_member__batch,
                glimpse_member__ball_radius,
                glimpse_consecutive__ball_radius, 
                glimpse_consecutive__batch, 
                glimpse_consecutive__center,
                glimpse_consecutive__voxel_center, 
                glimpse_consecutive__center_offset_ratio,
                glimpse__consecutive_to_nonconsecutive_glimpse_index)

    def take_glimpse_box(self, pos, rgb, batch, glimpse__center, voxel__center, glimpse__box_length_width_hight_ratio, glimpse__center_offset_ratio, glimpse__batch):

        # TODO: = can be further optimized to reduce the number of tensor index selection ?

        point_index = torch.arange(pos.size(0), dtype=torch.long, device=pos.device)

        # to overcome the fact that radius operation does not support different glimpse-wise radius, take glimpse with the maximum of radius first
        glimpse_member__glimpse_index, neighbor_index = radius(pos, glimpse__center, self.box_radius * (1 + self.boundary_ratio), batch, glimpse__batch, max_num_neighbors=2048)

        glimpse_member__global_pos = pos[neighbor_index] 

        glimpse_member__batch = batch[neighbor_index]

        glimpse_member__point_index = point_index[neighbor_index]
        ###############################################################################

        # TODO: = if rotation needs to be applied, apply here.

        glimpse_member__glimpse_center = glimpse__center[glimpse_member__glimpse_index]
        glimpse_member__local_pos = glimpse_member__global_pos - glimpse_member__glimpse_center

        # compute the glimpse-wise box size
        glimpse__box_length_width_hight = glimpse__box_length_width_hight_ratio * (self.box_size_max - self.box_size_min) + self.box_size_min

        glimpse_member__box_length_width_hight = glimpse__box_length_width_hight[glimpse_member__glimpse_index]
        glimpse_member__local_pos = glimpse_member__local_pos / glimpse_member__box_length_width_hight

        # scale pos local location into a cube covering -1, 1 in each dimension

        ###############################################################################

        # compute inf norm to get box
        glimpse_member__local_inf_norm = torch.norm(glimpse_member__local_pos, float('inf'), dim = 1)

        # the definition of glimpse now include boundarys
        index = (glimpse_member__local_inf_norm < (1 + self.boundary_ratio))

        # glimpse_member__local_inf_norm = glimpse_member__local_inf_norm[index]
        glimpse_member__local_pos = glimpse_member__local_pos[index]
        glimpse_member__batch = glimpse_member__batch[index]
        glimpse_member__point_index = glimpse_member__point_index[index]
        glimpse_member__box_length_width_hight = glimpse_member__box_length_width_hight[index]
        glimpse_member__glimpse_index = glimpse_member__glimpse_index[index]

        if self.training and self.boundary_ratio > 0:
            if self.fix_boundary:
                glimpse_member__boundary_distance = torch.clamp(torch.norm(glimpse_member__local_pos, float('inf'), dim = 1) - 1, min = 0)
            else:
                glimpse_member__boundary_offset = torch.clamp(glimpse_member__local_pos - 1, min = 0)
                glimpse_member__boundary_distance = torch.norm(glimpse_member__boundary_offset, float('inf'), dim = 1)
            # * torch.log1p is more accurate when input is close to zero (log(1)) but less accurate when input is close to -1 (log(0)) compared with torch.log
            x_by_r = glimpse_member__boundary_distance/self.boundary_ratio
            if self.boundary_func == 'linear':
                glimpse_member__log_boundary_weight = torch.log(1 - x_by_r + 1e-12) # y = - x/r + 1
            elif self.boundary_func == 'parabola':
                glimpse_member__log_boundary_weight = torch.log(1 + x_by_r**2 - 2*x_by_r + 1e-12) # y = x^2/r^2 - 2x/r + 1
            else:
                raise NotImplementedError
        else:
            if self.test_boundary_weight:
                # glimpse_member__log_boundary_weight = torch.zeros(glimpse_member__local_pos.size(0), device=glimpse_member__local_pos.device)
                glimpse_member__boundary_distance = torch.clamp(torch.norm(glimpse_member__local_pos, float('inf'), dim = 1) - 1, min = 0)
                x_by_r = glimpse_member__boundary_distance/self.boundary_ratio
                glimpse_member__log_boundary_weight = torch.log(1 + x_by_r**2 - 2*x_by_r + 1e-12)
                glimpse_member__log_boundary_weight[glimpse_member__log_boundary_weight > 0.5] = 1
            else:
                glimpse_member__log_boundary_weight = torch.zeros(glimpse_member__local_pos.size(0), device=glimpse_member__local_pos.device)

        # scale pos local location back
        if not self.fix_scale:
            glimpse_member__local_pos = glimpse_member__local_pos * glimpse_member__box_length_width_hight / self.box_size_max

        # make glimplse_member__glimpse_index consequtive, sorting involved, might be slow
        # * return glimpse_member__local_pos, None, glimpse_member__glimpse_index, glimpse_member__batch, glimpse_member__point_index, glimpse__box_length_width_hight, glimpse__batch
        glimpse_member__consecutive_glimpse_index, perm = consecutive_cluster(glimpse_member__glimpse_index)
        glimpse__consecutive_to_nonconsecutive_glimpse_index = glimpse_member__glimpse_index[perm]
        glimpse_consecutive__center = glimpse__center[glimpse__consecutive_to_nonconsecutive_glimpse_index]
        glimpse_consecutive__center_offset_ratio = glimpse__center_offset_ratio[glimpse__consecutive_to_nonconsecutive_glimpse_index]
        glimpse_consecutive__voxel_center = voxel__center[glimpse__consecutive_to_nonconsecutive_glimpse_index]
        glimpse_consecutive__batch = glimpse__batch[glimpse__consecutive_to_nonconsecutive_glimpse_index]
        glimpse_consecutive__box_length_width_hight = glimpse__box_length_width_hight[glimpse__consecutive_to_nonconsecutive_glimpse_index]

        return (glimpse_member__local_pos, None, 
                glimpse_member__consecutive_glimpse_index, 
                glimpse_member__point_index,
                glimpse_member__log_boundary_weight,
                glimpse_member__batch,
                glimpse_member__box_length_width_hight,
                glimpse_consecutive__box_length_width_hight,  # glimpse__size
                glimpse_consecutive__batch, 
                glimpse_consecutive__center, 
                glimpse_consecutive__voxel_center, 
                glimpse_consecutive__center_offset_ratio,
                glimpse__consecutive_to_nonconsecutive_glimpse_index)

    def compute_kl(self, pos_post, size_ratio_post, pres_post, log_z_pres, logit_pres, what_mask_post, bg__what_post, glimpse__consecutive_to_nonconsecutive_glimpse_index, glimpse__batch, flag):

        pos_kld = kl_divergence(pos_post, self.center_offset_prior.expand(pos_post.batch_shape)).sum(-1)[glimpse__consecutive_to_nonconsecutive_glimpse_index]
        size_ratio_kld = kl_divergence(size_ratio_post, self.radius_ratio_prior.expand(size_ratio_post.batch_shape)).sum(-1)[glimpse__consecutive_to_nonconsecutive_glimpse_index]
        pres_kld = kl_divergence(pres_post, self.pres_prior.expand(pres_post.batch_shape))

        if flag:
            pres_kld = pres_kld[glimpse__consecutive_to_nonconsecutive_glimpse_index]
            log_z_pres = log_z_pres[glimpse__consecutive_to_nonconsecutive_glimpse_index]
            logit_pres = logit_pres[glimpse__consecutive_to_nonconsecutive_glimpse_index]

        what_mask_kld = kl_divergence(what_mask_post, self.standard_normal.expand(what_mask_post.batch_shape)).sum(-1)
        
        bg_what_kld = kl_divergence(bg__what_post, self.standard_normal.expand(bg__what_post.batch_shape)).sum(-1)
        kld_fg = self.pres_kl_weight * pres_kld + torch.exp(log_z_pres) * (self.pos_kl_weight * pos_kld + self.size_ratio_kl_weight * size_ratio_kld + self.what_mask_kl_weight * what_mask_kld)
        batch_avg_kld_fg = torch.mean(scatter_sum(kld_fg, index = glimpse__batch, dim = 0))
        batch_avg_kld_bg = torch.mean(bg_what_kld)
            
        # ? how to weight what_coarse_kld ?
        return batch_avg_kld_fg + batch_avg_kld_bg, log_z_pres, logit_pres

    def compute_rc_loss(self, pos, rgb, batch, 
                            glimpse_member__log_mask, 
                            glimpse_member__rgb,
                            glimpse_member__local_pos,
                            glimpse_member__glimpse_index, 
                            glimpse_member__point_index,
                            glimpse_member__log_boundary_weight,
                            glimpse_member__batch,
                            glimpse_member__size,
                            glimpse_predict__local_pos, 
                            glimpse_predict__glimpse_index,
                            glimpse__center, 
                            # glimpse__center_diff,
                            glimpse__log_z_pres, 
                            glimpse__batch,
                            bg_predict__pos,
                            bg_predict__batch):

        device = pos.device

        if self.fix_scale:
            glimpse_member__local_pos = glimpse_member__local_pos * glimpse_member__size / self.box_size_max

        # combine member mask and glimpse pres to get unnormalized alpha
        glimpse_member__log_mask.squeeze_(1)
        glimpse_member__log_z_pres = glimpse__log_z_pres[glimpse_member__glimpse_index]

        full_point_index = torch.arange(pos.size(0), dtype=torch.long, device=device)

        glimpse_member__log_alpha = glimpse_member__log_mask + glimpse_member__log_z_pres + glimpse_member__log_boundary_weight #+ glimpse_member__center_log_likelihood
        glimpse_member__normalized_log_alpha = scatter_log_softmax(self.alpha_temperature * glimpse_member__log_alpha, glimpse_member__point_index, dim = 0) + glimpse_member__log_alpha

        #################################################################################################################################################
        # compute distance from truth points covered by at least one glimpse to its closest prediction points in global scale, distance is computed glimpse-wise
        # * forward 
        glimpse_member_index, glimpse_predict_index = knn(glimpse_predict__local_pos, glimpse_member__local_pos, 1, batch_x = glimpse_predict__glimpse_index, batch_y = glimpse_member__glimpse_index)   # compute foreground correspondence
        
        # * it is optional to zoom back to global distance
        glimpse_chamfer_predict__local_pos = glimpse_predict__local_pos[glimpse_predict_index]
        distance_forward_fg = torch.norm((glimpse_member__local_pos - glimpse_chamfer_predict__local_pos) * self.fg_distance_scale, p = None, dim = -1) 
        LL_forward_fg = self.pos_reconstruct_prob.log_prob(distance_forward_fg)

        glimpse_fg_chamfer_member__normalized_log_alpha = glimpse_member__normalized_log_alpha # gather the alpha value of the corresponding gt point of the selected predict point
        glimpse_chamfer_member__point_index = glimpse_member__point_index                      # gather the point index of the corresponding gt point of the selected predict point

        # ! will return -inf for index not included in index
        fg_log_alpha = scatter_logsumexp(glimpse_fg_chamfer_member__normalized_log_alpha, glimpse_chamfer_member__point_index, dim = 0, dim_size = pos.size(0))

        bg_log_alpha = torch.log(-torch.expm1(fg_log_alpha) + 1e-12)

        glimpse_bg_chamfer_member__normalized_log_alpha = bg_log_alpha
    

        index, bg_predict_index = knn(bg_predict__pos, pos, 1, batch_x = bg_predict__batch, batch_y = batch)   # compute background correspondence
        bg_chamfer_predict__pos = bg_predict__pos[bg_predict_index]
        distance_forward_bg = torch.norm((pos - bg_chamfer_predict__pos) * self.bg_distance_scale, dim = -1)
        LL_forward_bg = self.pos_reconstruct_prob.log_prob(distance_forward_bg)

        weighted_LL_forward_fg = LL_forward_fg + glimpse_fg_chamfer_member__normalized_log_alpha
        weighted_LL_forward_bg = LL_forward_bg + glimpse_bg_chamfer_member__normalized_log_alpha

        LL_forward = torch.cat((weighted_LL_forward_fg, weighted_LL_forward_bg), dim = 0)
        point_index = torch.cat((glimpse_chamfer_member__point_index, full_point_index), dim = 0)

        # sum over points that are covered by more than one glimpse and background to get point wise likelihood 
        LL_forward = scatter_logsumexp(LL_forward, index = point_index, dim = 0)
        
        # get the complete likelihood of each batch
        LL_forward = scatter_sum(LL_forward, batch, dim = 0)

        #################################################################################################################################################
        # compute distance from prediction points of each glimpse to its closest truth points in global scale, distance is computed glimpse-wise
        # * backward

        glimpse_predict_index, glimpse_member_index = knn(glimpse_member__local_pos, glimpse_predict__local_pos, 1, batch_x = glimpse_member__glimpse_index, batch_y = glimpse_predict__glimpse_index)  

        # * it is optional to zoom back to global distance
        distance_backward_fg = torch.norm((glimpse_member__local_pos[glimpse_member_index] - glimpse_predict__local_pos) * self.fg_distance_scale, p = None, dim = -1)
        LL_backward_fg = self.pos_reconstruct_prob.log_prob(distance_backward_fg)

        glimpse_fg_chamfer_member__normalized_log_alpha = glimpse_member__normalized_log_alpha[glimpse_member_index]

        bg_predict_index, index = knn(pos, bg_predict__pos, 1, batch_x = batch, batch_y = bg_predict__batch)   # compute background correspondence
        distance_bg = torch.norm((pos[index] - bg_predict__pos) * self.bg_distance_scale, dim = -1)
        LL_backward_bg = self.pos_reconstruct_prob.log_prob(distance_bg)

        glimpse_bg_chamfer_member__normalized_log_alpha = bg_log_alpha[index]

        # * backward LL 3:
        weighted_LL_backward_fg = LL_backward_fg * torch.exp(glimpse_fg_chamfer_member__normalized_log_alpha)
        weighted_LL_backward_bg = LL_backward_bg * torch.exp(glimpse_bg_chamfer_member__normalized_log_alpha)

        LL_backward_fg = scatter_sum(weighted_LL_backward_fg, glimpse__batch[glimpse_predict__glimpse_index], dim = 0)
        LL_backward_bg = scatter_sum(weighted_LL_backward_bg, bg_predict__batch, dim =0)

        LL_backward = LL_backward_fg + LL_backward_bg

        LL_backward_fg = torch.mean(LL_backward_fg)
        LL_backward_bg = torch.mean(LL_backward_bg)

        LL_avg_forward = torch.mean(LL_forward)
        LL_avg_backward = torch.mean(LL_backward)

        return (LL_avg_forward, 
                LL_avg_backward, 
                bg_log_alpha, 
                glimpse_member__normalized_log_alpha, 
                # glimpse__center_log_likelihood, 
                LL_backward_fg, 
                LL_backward_bg, 
                glimpse_chamfer_predict__local_pos, 
                bg_chamfer_predict__pos)

    def compute_alpha(self, pos, rgb, batch, 
                    glimpse_member__log_mask, 
                    glimpse_member__rgb,
                    glimpse_member__local_pos,
                    glimpse_member__glimpse_index, 
                    glimpse_member__point_index,
                    glimpse_member__log_boundary_weight,
                    glimpse_member__batch, 
                    glimpse_predict__local_pos, 
                    glimpse_predict__glimpse_index,
                    glimpse__center, 
                    glimpse__log_z_pres,
                    glimpse__logit_pres,
                    glimpse__consecutive_to_nonconsecutive_glimpse_index,
                    glimpse__batch,
                    bg_predict__pos,
                    bg_predict__batch, 
                    flag):

        if flag:
            log_z_pres = glimpse__log_z_pres[glimpse__consecutive_to_nonconsecutive_glimpse_index]
            logit_pres = glimpse__logit_pres[glimpse__consecutive_to_nonconsecutive_glimpse_index]

        device = pos.device

        # combine member mask and glimpse pres to get unnormalized alpha
        glimpse_member__log_mask.squeeze_(1)
        glimpse_member__log_z_pres = glimpse__log_z_pres[glimpse_member__glimpse_index]

        full_point_index = torch.arange(pos.size(0), dtype=torch.long, device=device)

        glimpse_member__log_alpha = glimpse_member__log_mask + glimpse_member__log_z_pres + glimpse_member__log_boundary_weight #+ glimpse_member__center_log_likelihood
        glimpse_member__normalized_log_alpha = scatter_log_softmax(self.alpha_temperature * glimpse_member__log_alpha, glimpse_member__point_index, dim = 0) + glimpse_member__log_alpha

        glimpse_member_index, glimpse_predict_index = knn(glimpse_predict__local_pos, glimpse_member__local_pos, 1, batch_x = glimpse_predict__glimpse_index, batch_y = glimpse_member__glimpse_index)   # compute foreground correspondence
        
        # * it is optional to zoom back to global distance
        glimpse_chamfer_predict__local_pos = glimpse_predict__local_pos[glimpse_predict_index]
        distance_forward_fg = torch.norm((glimpse_member__local_pos - glimpse_chamfer_predict__local_pos) * self.fg_distance_scale, p = None, dim = -1) 
        LL_forward_fg = self.pos_reconstruct_prob.log_prob(distance_forward_fg)

        glimpse_fg_chamfer_member__normalized_log_alpha = glimpse_member__normalized_log_alpha # gather the alpha value of the corresponding gt point of the selected predict point
        glimpse_chamfer_member__point_index = glimpse_member__point_index   

        fg_log_alpha = scatter_logsumexp(glimpse_fg_chamfer_member__normalized_log_alpha, glimpse_chamfer_member__point_index, dim = 0, dim_size = pos.size(0))

        bg_log_alpha = torch.log(-torch.expm1(fg_log_alpha) + 1e-12)

        glimpse_bg_chamfer_member__normalized_log_alpha = bg_log_alpha

        index, bg_predict_index = knn(bg_predict__pos, pos, 1, batch_x = bg_predict__batch, batch_y = batch)   # compute background correspondence
        bg_chamfer_predict__pos = bg_predict__pos[bg_predict_index]
        distance_forward_bg = torch.norm((pos - bg_chamfer_predict__pos) * self.bg_distance_scale, dim = -1)
        LL_forward_bg = self.pos_reconstruct_prob.log_prob(distance_forward_bg)
    
    def forward(self, pos, rgb, norm, batch):

        # TODO: normalized points coord to form the very first glimpse
        self.anneal_parameters()

        # point cloud message passing.
        pos, feature, batch = self.point_feature_network(pos, rgb, batch)

        # produce gird level features   
        ((glimpse__center_offset_ratio, glimpse__size_ratio, glimpse__log_z_pres, glimpse__logit_pres), 
        (glimpse__pos_post, glimpse__size_ratio_post, glimpse__pres_post), 
        voxel__center, voxel__out_feature, voxel__out_pos, voxel__out_batch) = self.grid_encoder(pos, feature, batch, self.pres_temperature)

        # carry out background (large scope / coarse level) prediction, since background is used to complete distributions, there is not mask prediction.
        bg_predict__pos, bg_predict__batch, bg__what_post = self.coarse_vae(voxel__out_pos, voxel__out_feature, voxel__out_batch, batch)

        glimpse__batch = voxel__out_batch
        glimpse__center = voxel__center + glimpse__center_offset_ratio * self.glimpse_center_offset_max

        # empty glimpses are taken out in take_glimpse_ball 
        (glimpse_member__local_pos, _, 
        glimpse_member__glimpse_index,
        glimpse_member__point_index, 
        glimpse_member__log_boundary_weight,
        glimpse_member__batch,
        glimpse_member__size,
        glimpse__size,                       # glimpse_consecutive__box_length_width_hight
        glimpse__batch,
        glimpse__center,
        glimpse__voxel_center,
        glimpse__center_offset_ratio,
        glimpse__consecutive_to_nonconsecutive_glimpse_index) = self.take_glimpse_ball(pos, rgb, batch, glimpse__center, voxel__center, glimpse__size_ratio, glimpse__center_offset_ratio, glimpse__batch) if self.glimpse_type == "ball" else self.take_glimpse_box(pos, rgb, batch, glimpse__center, voxel__center, glimpse__size_ratio, glimpse__center_offset_ratio, glimpse__batch)

        (glimpse__z_what,
        glimpse__z_mask,
        _glimpse__log_z_pres,
        _glimpse__logit_pres,
        glimpse_member__log_mask,
        _, 
        glimpse_predict__local_pos, 
        glimpse_predict__glimpse_index, 
        glimpse__what_mask_post,
        _glimpse__pres_post,
        glimpse__center_diff) = self.glimpse_vae(None, glimpse_member__local_pos, glimpse_member__glimpse_index, glimpse__center, glimpse__batch, self.pres_temperature)

        if self.glimpse_vae.generate_z_pres:
            glimpse__log_z_pres = _glimpse__log_z_pres
            glimpse__logit_pres = _glimpse__logit_pres
            glimpse__pres_post = _glimpse__pres_post

        # if self.training:
        if True:

            DKL, glimpse__log_z_pres, glimpse__logit_pres = self.compute_kl(glimpse__pos_post, 
                                                                            glimpse__size_ratio_post, 
                                                                            glimpse__pres_post, 
                                                                            glimpse__log_z_pres,
                                                                            glimpse__logit_pres,
                                                                            glimpse__what_mask_post, 
                                                                            bg__what_post, 
                                                                            glimpse__consecutive_to_nonconsecutive_glimpse_index,
                                                                            glimpse__batch,
                                                                            self.grid_encoder.generate_z_pres)

            (LL_forward, LL_backward, bg_log_alpha, 
            glimpse_member__normalized_log_alpha,
            # glimpse__center_log_likelihood,
            LL_backward_fg, LL_backward_bg,
            glimpse_chamfer_predict__local_pos, 
            bg_chamfer_predict__pos) = self.compute_rc_loss(pos, rgb, batch, 
                                                            glimpse_member__log_mask, 
                                                            None,
                                                            glimpse_member__local_pos,
                                                            glimpse_member__glimpse_index,
                                                            glimpse_member__point_index, 
                                                            glimpse_member__log_boundary_weight,
                                                            glimpse_member__batch,
                                                            glimpse_member__size,
                                                            glimpse_predict__local_pos, 
                                                            glimpse_predict__glimpse_index,
                                                            glimpse__center, 
                                                            # glimpse__center_diff,
                                                            # glimpse__size,
                                                            glimpse__log_z_pres,
                                                            glimpse__batch,
                                                            bg_predict__pos,
                                                            bg_predict__batch)

            return (-LL_forward, -LL_backward, DKL, 
                    glimpse__size, 
                    glimpse__batch,
                    glimpse__center, 
                    glimpse__voxel_center,
                    glimpse__z_what,
                    glimpse__z_mask,
                    glimpse__log_z_pres,
                    glimpse__logit_pres,
                    glimpse__center_diff * self.fg_distance_scale,
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
                    -LL_backward_fg,
                    -LL_backward_bg,
                    glimpse_chamfer_predict__local_pos,
                    bg_chamfer_predict__pos)