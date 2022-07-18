import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli
from torch.distributions.bernoulli import Bernoulli

from torch_geometric.nn import radius_graph, LayerNorm

from torch_scatter import scatter_mean, scatter_sum, scatter_log_softmax

from models.layers.AutoRegistration import AutoRegistrationLayer
from models.layers.PointConv import PointConv, CenterShift
from models.utils import to_sigma, find_voxel_center, voxel_mean_pool

class SPAIRPointFeatureNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.radius = 1/16          
        self.max_num_neighbors = 128

        self.conv1 = PointConv(c_in = 3, c_mid = 8, c_out = 8)
        self.conv2 = PointConv(c_in = 8, c_mid = 16, c_out = 16)
        self.conv3 = PointConv(c_in = 16, c_mid = 32, c_out = 32)

    def forward(self, pos, rgb, batch):

        out_index, in_index = radius_graph(pos, self.radius, batch, loop=True, max_num_neighbors=64, flow='target_to_source')

        out = F.celu(self.conv1(pos, pos, batch, in_index = in_index, out_index = out_index))
        out = F.celu(self.conv2(out, pos, batch, in_index = in_index, out_index = out_index))
        out = F.celu(self.conv3(out, pos, batch, in_index = in_index, out_index = out_index))

        return pos, out, batch

class SPAIRGridFeatureNetwork(torch.nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.layer_norm = cfg.grid_encoder_ln
        self.ar = cfg.grid_encoder_ar
        self.glimpse_type = cfg.glimpse_type

        # radius graph message passing
        if self.ar:
            self.ar1 = AutoRegistrationLayer(x_dim = 64, f_hidden = 64, f_out = 64, g_hidden = 64, g_out = 64)
            self.ar2 = AutoRegistrationLayer(x_dim = 128, f_hidden = 128, f_out = 128, g_hidden = 128, g_out = 128)
            self.ar3 = AutoRegistrationLayer(x_dim = 256, f_hidden = 256, f_out = 256, g_hidden = 256, g_out = 256)
            self.ar4 = AutoRegistrationLayer(x_dim = 256, f_hidden = 256, f_out = 256, g_hidden = 256, g_out = 256)
            self.ar5 = AutoRegistrationLayer(x_dim = 256, f_hidden = 256, f_out = 256, g_hidden = 256, g_out = 256)

        # grid information aggregation
        self.conv1 = PointConv(16, max_num_neighbors = 128, c_in = 32, c_mid = 32, c_out = 64)
        self.conv2 = PointConv(2/16, max_num_neighbors = 128, c_in = 64, c_mid = 64, c_out = 128)
        self.conv3 = PointConv(2/16, max_num_neighbors = 128, c_in = 128, c_mid = 128, c_out = 256)
        self.conv4 = CenterShift(c_in = 256, c_mid = 256, c_out = 256)

        # normalization
        if self.layer_norm:
            self.norm1 = LayerNorm(64)
            self.norm2 = LayerNorm(128)
            self.norm3 = LayerNorm(256)

        # TODO: wasting one digit for now if self.generate_z_pres is False
        if self.glimpse_type == "ball":
            self.linear = torch.nn.Linear(in_features = 256, out_features = 9)
        elif self.glimpse_type == "box":
            self.linear = torch.nn.Linear(in_features = 256, out_features = 13)

    def forward(self, pos, feature, batch, temperature):
        """
        clustering is done in entire point clouds, thus use batch as index.
        """

        max_pos, _ = torch.max(pos, dim = 0) 
        min_pos, _ = torch.min(pos, dim = 0)
        # add noise that ranges from 0 to voxel_gird_size
        noise = torch.rand_like(min_pos) * (1/8)
        min_pos -= noise

        (out_index, in_index), pos, batch, pos_sample, batch_sample, voxel_cluster, voxel_cluster_sample, inv = voxel_mean_pool(pos = pos, batch = batch, start = min_pos, end = max_pos, size = 0.5 / 16)
        feature = feature[inv]
        
        # # ! Debug: with voxel grid clustering, every point is assigned to one and only one cluster, torch.min(num_points) == 0 means that
        # num_points = scatter_sum(torch.ones(out_index.shape, device=in_index.device), out_index, dim = 0)
        # assert torch.min(num_points) > 0
        # # ! Debug:

        # voxel_cluster << batch
        
        feature = F.celu(self.conv1(x_in = feature, pos_in = pos, batch_in = voxel_cluster, pos_out = pos_sample, batch_out = voxel_cluster_sample, in_index = in_index, out_index = out_index))

        ################################################################################################

        pos = pos_sample
        batch = batch_sample

        edge_index = radius_graph(pos, 0.5 / 16, batch, loop=True)

        if self.ar:
            feature, _, _ = self.ar1(feature, pos, edge_index)

        if self.layer_norm:
            feature = self.norm1(feature, batch)

        (out_index, in_index), pos, batch, pos_sample, batch_sample, voxel_cluster, voxel_cluster_sample, inv = voxel_mean_pool(pos = pos, batch = batch, start = min_pos, end = max_pos, size = 1 / 16)
        feature = feature[inv]

        # # ! Debug:
        # num_points = scatter_sum(torch.ones(out_index.shape, device=in_index.device), out_index, dim = 0)
        # assert torch.min(num_points) > 0
        # # ! Debug:

        feature = F.celu(self.conv2(x_in = feature, pos_in = pos, batch_in = voxel_cluster, pos_out = pos_sample, batch_out = voxel_cluster_sample, in_index = in_index, out_index = out_index))

        ################################################################################################

        pos = pos_sample
        batch = batch_sample

        edge_index = radius_graph(pos, 2 / 16, batch, loop=True)

        if self.ar:
            feature, _, _ = self.ar2(feature, pos, edge_index)

        if self.layer_norm:
            feature = self.norm2(feature, batch)


        (out_index, in_index), pos, batch, pos_sample, batch_sample, voxel_cluster, voxel_cluster_sample, inv = voxel_mean_pool(pos = pos, batch = batch, start = min_pos, end = max_pos, size = 2 / 16)
        feature = feature[inv]

        # # ! Debug:
        # num_points = scatter_sum(torch.ones(out_index.shape, device=in_index.device), out_index, dim = 0)
        # assert torch.min(num_points) > 0
        # # ! Debug:

        feature = F.celu(self.conv3(x_in = feature, pos_in = pos, batch_in = voxel_cluster, pos_out = pos_sample, batch_out = voxel_cluster_sample, in_index = in_index, out_index = out_index))

        ################################################################################################

        pos = pos_sample
        batch = batch_sample

        edge_index = radius_graph(pos, 4 / 16, batch, loop=True)

        if self.ar:
            feature, _, _ = self.ar3(feature, pos, edge_index)
            feature, _, _ = self.ar4(feature, pos, edge_index)
            feature, _, _ = self.ar5(feature, pos, edge_index)

        if self.layer_norm:
            feature = self.norm3(feature, batch)

        # find voxel center of the corresponding pos
        voxel_center = find_voxel_center(pos, start = min_pos, size = 2 / 16)

        # move the glimpse center offset origin to voxel center
        center_feature = self.conv4(feature, pos, voxel_center)

        out = self.linear(center_feature) # B * N, 9

        if self.glimpse_type == "ball":
            mu_pos, sigma_pos, mu_size_ratio, sigma_size_ratio, glimpse__logit_pres = torch.split(out, [3, 3, 1, 1, 1], dim = 1)
        else: # self.glimpse_type == "box":
            mu_pos, sigma_pos, mu_size_ratio, sigma_size_ratio, glimpse__logit_pres = torch.split(out, [3, 3, 3, 3, 1], dim = 1) #

        pos_post = Normal(mu_pos, to_sigma(sigma_pos))
        # print(to_sigma(sigma_pos))
        size_ratio_post = Normal(mu_size_ratio, to_sigma(sigma_size_ratio))

        z_pos = pos_post.rsample()
        # z_pos = mu_pos

        z_r = size_ratio_post.rsample()
        # z_r = mu_size_ratio

        pres_post = None
        log_z_pres = None
        glimpse__logit_pres = None

        glimpse__center_offset_ratio = torch.tanh(z_pos)
        glimpse__ball_radius_ratio = torch.sigmoid(z_r)

        return (glimpse__center_offset_ratio, glimpse__ball_radius_ratio, log_z_pres, glimpse__logit_pres), (pos_post, size_ratio_post, pres_post), voxel_center, feature, pos, batch_sample

class SPAIRGlimpseEncoder(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        """
        takes in glimpse and generate masks and predicted point coordinates
        current structure: one encoder and two decoders, one MLP decoder for point prediction and one point conv decoder for mask prediction
        """

        self.layer_norm = cfg.glimpse_encoder_ln
        self.ar = cfg.glimpse_encoder_ar

        # radius graph message passing
        if self.ar:
            self.ar1 = AutoRegistrationLayer(x_dim = 3, f_hidden = 8, f_out = 8, g_hidden = 8, g_out = 8)
            self.ar2 = AutoRegistrationLayer(x_dim = 32, f_hidden = 32, f_out = 32, g_hidden = 32, g_out = 32)
            self.ar3 = AutoRegistrationLayer(x_dim = 128, f_hidden = 128, f_out = 128, g_hidden = 128, g_out = 128)

        # grid information aggregation
        self.conv1 = PointConv(0.25, max_num_neighbors = 64, c_in = 8 if self.ar else 1, c_mid = 16, c_out = 32)
        self.conv2 = PointConv(0.5, max_num_neighbors = 64, c_in = 32, c_mid = 64, c_out = 128)
        self.conv3 = PointConv(1, max_num_neighbors = 64, c_in = 128, c_mid = 128, c_out = 256)

        # normalization
        if self.layer_norm:
            self.norm1 = LayerNorm(16)
            self.norm2 = LayerNorm(64)
            self.norm3 = LayerNorm(128)
            self.norm4 = LayerNorm(256)

        self.linear = torch.nn.Linear(in_features = 256, out_features = 256)

        # MLP decoder

    def forward(self, rgb, pos, glimpse_member__glimpse_index, glimpse__center, glimpse__batch): # pos == glimpse_member__local_pos, batch == glimpse_member__glimpse_index

        min_pos, _ = torch.min(pos, dim = 0)
        max_pos, _ = torch.max(pos, dim = 0)
        # add noise that ranges from 0 to voxel_gird_size
        noise = torch.rand_like(min_pos)
        min_pos -= noise


        pos_list = [pos]
        glimpse_index_list = [glimpse_member__glimpse_index]
        in_out_index_list = []

        edge_index = radius_graph(pos, 0.25, glimpse_member__glimpse_index, loop=True)

        if self.ar:
            feature, _, _ = self.ar1(pos, pos, edge_index)
            if self.layer_norm:
                feature = self.norm1(feature, glimpse_member__glimpse_index)
        else:
            feature = rgb
        # use voxel pooling to make sure that all points in one voxel are covered.
        (out_index, in_index), pos, glimpse_member__glimpse_index, pos_sample, glimpse_member_sample__glimpse_index, voxel_cluster, voxel_cluster_sample, inv = voxel_mean_pool(pos = pos, batch = glimpse_member__glimpse_index, start = min_pos, end = max_pos, size = 0.25)
        
        # # ! Debug: with voxel grid clustering, every point is assigned to one and only one cluster, torch.min(num_points) == 0 means that
        # num_points = scatter_sum(torch.ones(out_index.shape, device=in_index.device), out_index, dim = 0)
        # assert torch.min(num_points) > 0
        # # ! Debug:

        # voxel_cluster << batch
        
        feature = F.celu(self.conv1(x_in = feature, pos_in = pos, batch_in = voxel_cluster, pos_out = pos_sample, batch_out = voxel_cluster_sample, in_index = in_index, out_index = out_index))

        ################################################################################################

        pos = pos_sample
        glimpse_member__glimpse_index = glimpse_member_sample__glimpse_index

        pos_list.append(pos)
        glimpse_index_list.append(glimpse_member__glimpse_index)
        in_out_index_list.append((in_index, out_index))

        edge_index = radius_graph(pos, 0.5, glimpse_member__glimpse_index, loop=True)

        if self.ar:
            feature, _, _ = self.ar2(feature, pos, edge_index)

        if self.layer_norm:
            feature = self.norm2(feature, glimpse_member__glimpse_index)

        (out_index, in_index), pos, glimpse_member__glimpse_index, pos_sample, glimpse_member_sample__glimpse_index, voxel_cluster, voxel_cluster_sample, inv = voxel_mean_pool(pos = pos, batch = glimpse_member__glimpse_index, start = min_pos, end = max_pos, size = 0.5)
        feature = feature[inv]

        # # ! Debug:
        # num_points = scatter_sum(torch.ones(out_index.shape, device=in_index.device), out_index, dim = 0)
        # assert torch.min(num_points) > 0
        # # ! Debug:

        feature = F.celu(self.conv2(x_in = feature, pos_in = pos, batch_in = voxel_cluster, pos_out = pos_sample, batch_out = voxel_cluster_sample, in_index = in_index, out_index = out_index))

        ################################################################################################

        pos = pos_sample
        glimpse_member__glimpse_index = glimpse_member_sample__glimpse_index

        pos_list.append(pos)
        glimpse_index_list.append(glimpse_member__glimpse_index)
        in_out_index_list.append((in_index, out_index))

        edge_index = radius_graph(pos, 1.0, glimpse_member__glimpse_index, loop=True)

        if self.ar:
            feature, _, _ = self.ar3(feature, pos, edge_index)
        if self.layer_norm:
            feature = self.norm3(feature, glimpse_member__glimpse_index)

        # aggregate all points in one glimpse to the glimpse center with local coordinate (0,0,0)
        pos_sample = torch.zeros_like(glimpse__center)                                                  
        glimpse_member_sample__glimpse_index = torch.arange(glimpse__center.size(0), dtype = torch.long, device = pos.device)   # 
        in_index = torch.arange(pos.size(0), dtype = torch.long, device = pos.device)
        out_index = glimpse_member__glimpse_index

        # # ! Debug:
        # num_points = scatter_sum(torch.ones(out_index.shape, device=in_index.device), out_index, dim = 0)
        # assert torch.min(num_points) > 0
        # assert_sorted_consecutive(in_index)
        # assert_sorted_consecutive(out_index)
        # # ! Debug:

        feature = F.celu(self.conv3(x_in = feature, pos_in = pos, batch_in = glimpse_member__glimpse_index, pos_out = pos_sample, batch_out = glimpse_member_sample__glimpse_index, in_index = in_index, out_index = out_index))

        ################################################################################################

        pos_list.append(pos_sample)
        glimpse_index_list.append(glimpse_member_sample__glimpse_index)
        in_out_index_list.append((in_index, out_index))

        out = self.linear(feature)

        mu, sigma = torch.chunk(out, 2, dim = 1)

        what_mask_post = Normal(mu, to_sigma(sigma))
        z_what_mask = what_mask_post.rsample()
        # z_what_mask = reparameterization(mu, logvar)
        z_what, z_mask = torch.chunk(z_what_mask, 2, dim = 1)

        # return z_what, z_mask, log_z_pres, glimpse__logit_pres, what_mask_post, pres_post, pos_list, glimpse_index_list, in_out_index_list
        return z_what, z_mask, what_mask_post, pos_list, glimpse_index_list, in_out_index_list, feature

class SPAIRGlimpseZPresGenerator(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.radius_max = cfg.max_radius
        self.layer_norm = cfg.glimpse_encoder_ln

        self.z_pres_linear = torch.nn.Linear(in_features = 8, out_features = 1)

        self.ar1 = AutoRegistrationLayer(x_dim = 256, f_hidden = 128, f_out = 64, g_hidden = 64, g_out = 64)
        self.ar2 = AutoRegistrationLayer(x_dim = 64, f_hidden = 32, f_out = 32, g_hidden = 32, g_out = 32)
        self.ar3 = AutoRegistrationLayer(x_dim = 32, f_hidden = 16, f_out = 16, g_hidden = 16, g_out = 8)

    def forward(self, glimpse__feature, glimpse__center, glimpse__batch, glimpse_member__local_pos, glimpse_member__log_mask, glimpse_member__glimpse_index, temperature):

        glimpse_member__normalized_mask = torch.exp(scatter_log_softmax(glimpse_member__log_mask, index = glimpse_member__glimpse_index, dim = 0))
        glimpse_member__weighted_pos = glimpse_member__local_pos * glimpse_member__normalized_mask
        glimpse__member_center = scatter_sum(glimpse_member__weighted_pos, glimpse_member__glimpse_index, dim = 0)

        glimpse__center_local_scale = glimpse__center / self.radius_max

        edge_index = radius_graph(glimpse__center_local_scale, 1, glimpse__batch, loop=True)
        
        z_pres_feature, _, _ = self.ar1(glimpse__feature, glimpse__center_local_scale, edge_index)
        z_pres_feature, _, _ = self.ar2(z_pres_feature, glimpse__center_local_scale, edge_index)
        z_pres_feature, _, _ = self.ar3(z_pres_feature, glimpse__center_local_scale, edge_index)

        glimpse__logit_pres = self.z_pres_linear(z_pres_feature).squeeze(1)
        glimpse__logit_pres = 8.8 * torch.tanh(glimpse__logit_pres)
        pres_post = Bernoulli(logits = glimpse__logit_pres)
        log_z_pres = F.logsigmoid(LogitRelaxedBernoulli(logits=glimpse__logit_pres, temperature=temperature).rsample())

        return log_z_pres, glimpse__logit_pres, pres_post, glimpse__member_center

class SPAIRGlimpseZPresMLP(torch.nn.Module):
    
    def __init__(self, cfg) -> None:
        super().__init__()

        self.z_pres_linear = torch.nn.Linear(in_features = 256, out_features = 1)
    
    def forward(self, glimpse__feature, glimpse_member__local_pos, glimpse_member__log_mask, glimpse_member__glimpse_index, temperature):

        glimpse_member__normalized_mask = torch.exp(scatter_log_softmax(glimpse_member__log_mask, index = glimpse_member__glimpse_index, dim = 0))
        glimpse_member__weighted_pos = glimpse_member__local_pos * glimpse_member__normalized_mask
        glimpse__member_center = scatter_sum(glimpse_member__weighted_pos, glimpse_member__glimpse_index, dim = 0)
        
        glimpse__logit_pres = self.z_pres_linear(glimpse__feature).squeeze(1)
        glimpse__logit_pres = 8.8 * torch.tanh(glimpse__logit_pres)
        pres_post = Bernoulli(logits = glimpse__logit_pres)
        log_z_pres = F.logsigmoid(LogitRelaxedBernoulli(logits=glimpse__logit_pres, temperature=temperature).rsample())

        return log_z_pres, glimpse__logit_pres, pres_post, glimpse__member_center

class SPAIRGlimpseMaskDecoder(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.conv1 = PointConv(1, max_num_neighbors = 64, c_in = 64, c_mid = 64, c_out = 32)
        self.conv2 = PointConv(0.5, max_num_neighbors = 64, c_in = 32, c_mid = 16, c_out = 16)
        self.conv3 = PointConv(0.25, max_num_neighbors = 64, c_in = 16, c_mid = 8, c_out = 8)
            
        self.linear = torch.nn.Linear(in_features = 8, out_features = 1)

    def forward(self, z_mask, pos_list, glimpse_index_list, in_out_index_list):

        # TODO: get in_index and out_index from encoder for decoding.

        (in_index, out_index) = in_out_index_list[-1]

        out = F.celu(self.conv1(x_in = z_mask, pos_in = pos_list[-1], batch_in = glimpse_index_list[-1], pos_out = pos_list[-2], batch_out = glimpse_index_list[-2], in_index = out_index, out_index = in_index))

        (in_index, out_index) = in_out_index_list[-2]

        out = F.celu(self.conv2(out, pos_list[-2], glimpse_index_list[-2], pos_list[-3], glimpse_index_list[-3], in_index = out_index, out_index = in_index))

        (in_index, out_index) = in_out_index_list[-3]

        out = F.celu(self.conv3(out, pos_list[-3], glimpse_index_list[-3], pos_list[-4], glimpse_index_list[-4], in_index = out_index, out_index = in_index))

        out = self.linear(out)

        out = F.logsigmoid(out)

        return out

class SPAIRGlimpseRGBDecoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = PointConv(1, max_num_neighbors = 64, c_in = 128, c_mid = 128, c_out = 64)
        self.conv2 = PointConv(0.5, max_num_neighbors = 64, c_in = 64, c_mid = 32, c_out = 32)
        self.conv3 = PointConv(0.25, max_num_neighbors = 64, c_in = 32, c_mid = 16, c_out = 16)

        self.linear = torch.nn.Linear(in_features = 16, out_features = 3)

    def forward(self, z_what, pos_list, glimpse_index_list):

        out = F.celu(self.conv1(z_what, pos_list[-1], glimpse_index_list[-2]))

        out = F.celu(self.conv2(out, pos_list[-2], glimpse_index_list[-3]))

        out = F.celu(self.conv3(out, pos_list[-3], glimpse_index_list[-4]))

        out = self.linear(out)

        return out

class SPAIRPointPosDecoder(torch.nn.Module):
    def __init__(self, latent_size = 128, num_points = 1024):
        super().__init__()

        # ! for clouds with the same number of points only

        self.num_points = num_points

        self.fc1 = torch.nn.Linear(in_features = latent_size, out_features = 256)
        self.fc2 = torch.nn.Linear(in_features = 256, out_features = 512)
        self.fc3 = torch.nn.Linear(in_features = 512, out_features = 1024)
        self.fc4 = torch.nn.Linear(in_features = 1024, out_features = num_points * 3) # xyz mask

    def forward(self, z, glimpse_index, center_flag = True):  # n_glimpse, latent_size    n_glimpse

        x = F.celu(self.fc1(z))

        x = F.celu(self.fc2(x))

        x = F.celu(self.fc3(x))

        x = self.fc4(x)

        x = x.view(x.shape[0], -1, 3) # n_glimpse, num_points, 3

        # center generated points
        if center_flag:
            x_center = torch.mean(x, dim=1, keepdim=True)
            x = x - x_center

        glimpse_index = glimpse_index.unsqueeze(1).repeat(1, self.num_points)

        # x = x.view(-1, 3)
        # pos_glimpse_index = pos_glimpse_index.view(-1)

        x = torch.cat(list(x), dim = 0)
        pos_predict_glimpse_index = torch.cat(list(glimpse_index), dim = 0)

        return x, pos_predict_glimpse_index

class SPAIRPointPosFlow(torch.nn.Module):
    def __init__(self, latent_size = 128, layer_norm = False):
        super().__init__()
        # * when the number of points in one glimpse is low, radius graph cannot guarantee that all nodes are connected.

        self.ar1 = AutoRegistrationLayer(x_dim = 3 + latent_size, f_hidden = 128, f_out = 128, g_hidden = 128, g_out = 64 + 3, end_relu=False)
        self.ar2 = AutoRegistrationLayer(x_dim = 64, f_hidden = 64, f_out = 64, g_hidden = 64, g_out = 32 + 3, end_relu=False)
        self.ar3 = AutoRegistrationLayer(x_dim = 32, f_hidden = 16, f_out = 16, g_hidden = 16, g_out = 3, end_relu=False)

        self.layer_norm = layer_norm

        if self.layer_norm:
            self.norm1 = LayerNorm(64)
            self.norm2 = LayerNorm(32)

        self.noise = None

    def forward(self, z, batch, center_flag = True, extra_predict_ratio = 0.25):
        # expand z to parallel following operation

        if self.noise is None:
            # all glimpse points are inside the ball with radius 1
            self.noise = Normal(torch.tensor(0.0, device=z.device), torch.tensor(0.3, device = z.device))

        if extra_predict_ratio > 0:
            
            prob = torch.ones(batch.size(0), device = batch.device)
            sample = torch.multinomial(prob, int(batch.size(0) * extra_predict_ratio))
            batch = torch.cat((batch, batch[sample]), dim = 0)

        z = z[batch]

        population = self.noise.sample((batch.size(0), 3))

        edge_index = radius_graph(population, 0.2, batch)

        feature = torch.cat((z, population), dim = 1)

        feature, _, _ = self.ar1(feature, population, edge_index)

        (feature, population) = torch.split(feature, (64, 3), dim = 1)

        if self.layer_norm:
            feature = self.norm1(feature)

        feature = F.celu(feature)

        edge_index = radius_graph(population, 0.1, batch)
        
        feature, _, _ = self.ar2(feature, population, edge_index)

        (feature, population) = torch.split(feature, (32, 3), dim = 1)

        if self.layer_norm:
            feature = self.norm2(feature)

        feature = F.celu(feature)

        edge_index = radius_graph(population, 0.05, batch)

        population, _, _ = self.ar3(feature, population, edge_index)

        # zero center
        if center_flag:
            population_center = scatter_mean(population, batch, dim=0)
            population = population - population_center[batch]
        
        return population, batch

class SPAIRGlimpseVAE(torch.nn.Module):
    def __init__(self, cfg):

        super().__init__()

        self.no_ZPres_generator = False
        
        self.encoder = SPAIRGlimpseEncoder(cfg)
        if self.no_ZPres_generator:
            self.z_pres_mlp = SPAIRGlimpseZPresMLP(cfg)
        else:
            self.z_pres_generator = SPAIRGlimpseZPresGenerator(cfg)
        self.mask_decoder = SPAIRGlimpseMaskDecoder(cfg)

        self.pos_decoder = SPAIRPointPosFlow(latent_size=64)

        self.extra_predict_ratio = cfg.extra_predict_ratio
        self.no_ZPres_generator = cfg.no_ZPres_generator

    def forward(self, rgb, glimpse_member__local_pos, glimpse_member__glimpse_index, glimpse__center, glimpse__batch, temperature):

        (glimpse__z_what,
        glimpse__z_mask, 
        glimpse__what_mask_post,
        pos_list, 
        glimpse_index_list, 
        in_out_index_list,
        glimpse__feature) = self.encoder(rgb, glimpse_member__local_pos, glimpse_member__glimpse_index, glimpse__center, glimpse__batch)

        glimpse_member__log_mask = self.mask_decoder(glimpse__z_mask, pos_list, glimpse_index_list, in_out_index_list)

        if self.no_ZPres_generator:
            (glimpse__log_z_pres, 
            glimpse__logit_pres,
            glimpse__pres_post, 
            glimpse__member_center) = self.z_pres_mlp(glimpse__feature, 
                                                        glimpse_member__local_pos,
                                                        glimpse_member__log_mask, 
                                                        glimpse_member__glimpse_index,
                                                        temperature)
        else:
            (glimpse__log_z_pres, 
            glimpse__logit_pres,
            glimpse__pres_post, 
            glimpse__member_center) = self.z_pres_generator(glimpse__feature, 
                                                            glimpse__center,
                                                            glimpse__batch,
                                                            glimpse_member__local_pos,
                                                            glimpse_member__log_mask, 
                                                            glimpse_member__glimpse_index,
                                                            temperature)

        glimpse__center_diff = torch.norm(glimpse__member_center - glimpse__center, 2, dim = 1)
        
        # if rgb is not None:
        #     rgb_predict = self.rgb_decoder(z_what, pos_list, glimpse_index_list)
        # else:
        #     rgb_predict = None

        # glimpse_predict__pos, glimpse_predict__glimpse_index = self.pos_decoder(z_what, glimpse_index_list[-1])
        glimpse_predict__pos, glimpse_predict__glimpse_index = self.pos_decoder(glimpse__z_what, glimpse_member__glimpse_index, True, extra_predict_ratio = self.extra_predict_ratio)

        # glimpse_predict__pos = torch.tanh(glimpse_predict__pos) # generated points must live in unit ball/cube

        return (glimpse__z_what,
                glimpse__z_mask,
                glimpse__log_z_pres, 
                glimpse__logit_pres, 
                glimpse_member__log_mask, 
                None, 
                glimpse_predict__pos, 
                glimpse_predict__glimpse_index, 
                glimpse__what_mask_post, 
                glimpse__pres_post, 
                glimpse__center_diff)

class CoarseEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = PointConv(10, max_num_neighbors = 64, c_in = 256, c_mid = 256, c_out = 512)

    def forward(self, pos, feature, batch):

        pos_center = scatter_mean(pos, batch, dim = 0)
        pos_center_batch = torch.arange(pos_center.size(0), dtype=torch.long, device=pos.device)

        _index = torch.arange(pos.size(0), dtype=torch.long, device=pos.device) # !

        assert torch.max(_index) == (feature.size(0) - 1), "CoarseEncoder assertion triggered"

        out = self.conv1(x_in = feature, pos_in = pos, batch_in = batch, pos_out = pos_center, batch_out = pos_center_batch, in_index = _index, out_index = batch)

        mu, sigma = torch.chunk(out, 2, dim = 1) # B * N, 256

        what_coarse_post = Normal(mu, to_sigma(sigma))

        z_what_coarse = what_coarse_post.rsample()

        return z_what_coarse, what_coarse_post, pos_center_batch

class CoarseVAE(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = CoarseEncoder()
        self.decoder = SPAIRPointPosFlow(latent_size = 256)

    def forward(self, voxel__pos, voxel__feature, voxel__batch, batch):

        # voxel__feature = voxel__feature.detach()

        z_what_coarse, what_coarse_post, pos_center_batch = self.encoder(voxel__pos, voxel__feature, voxel__batch)

        coarse_point_predict, pos_predict_batch = self.decoder(z_what_coarse, batch, False)

        return coarse_point_predict, pos_predict_batch, what_coarse_post
