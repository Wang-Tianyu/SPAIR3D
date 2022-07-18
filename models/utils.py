import os
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score

import torch
import torch.nn.functional as F

from torch_geometric.data import Batch
from torch_geometric.nn import voxel_grid, knn, knn_graph
from torch_geometric.nn.pool.consecutive import consecutive_cluster

from torch_scatter import scatter_mean, scatter_std, scatter_sum, scatter_max, scatter_logsumexp


def to_sigma(x):
    # log(1 + exp(beta * x))/beta
    return F.softplus(x + 0.5)

def to_var(x):
    return to_sigma(x)**2

def reparameterization(mu, logvar):

    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    z = mu + eps*std

    return z

def normal_kl_divergence(mu, logvar):
    """
    return glimpse_wise KL divergence
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = -1)

def categorical_kl_divergence(log_beta, log_beta_prior): # B * G, C    C
    """
    return glimpse_wise KL divergence
    """

    return torch.sum(log_beta * (log_beta - log_beta_prior), dim = -1)

def bernoulli_kl_divergence(beta_prob, beta_prior, eps = 1e-12):

    kl = beta_prob * (torch.log(beta_prob + eps) - torch.log(beta_prior + eps)) + (1 - beta_prob) * (torch.log(1 - beta_prob + eps) - torch.log(1 - beta_prior + eps))

    kl = kl.squeeze()

    return kl

def find_voxel_center(pos, start, size):

    pos = pos - start
    pos = pos/size
    pos = torch.floor(pos)
    pos = pos * size 
    pos = pos + start
    pos = pos + size/2

    return pos

def diagonal_distance(size):
    # ! non pytorch operation
    return math.sqrt(size**2 * 3)

def point_layer_norm(x, batch, eps = 1e-5):
    """
    Layer norm for point cloud data structure.
    Since point clouds in one batch usually have different number of points, there is not way to learn elementwise affine transformation
    x: the data vector K*B*N, C
    """

    assert x.dim() == 2, "input x for point layer norm should have 2 dims"

    C = x.size(1)

    if C == 1:
        Ex = scatter_mean(x, batch, dim = 0)
        Stdx = scatter_std(x, batch, unbiased=False, dim = 0)
    else:
        _x = torch.split(x, 1, dim = 1)
        _x = torch.cat(_x, dim = 0)
        _batch = batch.expand(C, -1)
        Ex = scatter_mean(_x, _batch, dim = 0)
        Stdx = scatter_std(_x, _batch, dim = 0)

    Ex = Ex[batch]
    Stdx = Stdx[batch]

    y = (x - Ex)/(Stdx + eps)

    return y

def repeat_latent(latent, pos, batch):
    """
    latent: B, K, D
    pos: B*N, 3
    batch: B*N, 1
    flatten the latent, repeat the pos K times to concatenate with latent in the same group (same batch and same component).
    batch is repeated.
    K is generated.
    group is generated.
    """

    B, K, _ = latent.shape

    BN = batch.size(0)  # batch * num_points

    latent = torch.cat(torch.split(latent, 1, dim = 1), dim = 0).squeeze() # K * B, D

    S = torch.arange(BN, dtype = torch.long, device = pos.device)
    S = S.repeat(K)

    pos = pos.repeat(K, 1)  # K * B * N, 3

    batch = batch.repeat(K) # K * B * N, 1

    K_index = torch.ones((K, BN), dtype=torch.long, device=pos.device ) 
    K_index = torch.cumsum(K_index, dim = 0) - 1
    K_index = torch.cat(list(K_index), dim = 0)                                 # K * BN      {00}{1111}

    group_index = batch + K_index * B

    # latent = latent[group_index]

    return pos, latent, batch, K_index, group_index, S

def repeat_batch_group(data, K):
    """
    repeat a batch of sparse points K times to create K components.
    when repeat alone first dim (creating more points), the batch index will be repeated.
    a new index called K is added to mark the component number.
    a new index called group is added to combine both component information and batch information (two points will only have the same group index if they are both in the same component and the same batch). 
    """
    # * support variant number of points per cloud

    x, pos, batch_index = data.x, data.pos, data.batch     # batch_size * num_points  |00|1111|2|333|

    if x is not None:
        x = x.repeat(K, 1)
    pos = pos.repeat(K, 1)  # K * batch_size * num_points  {|00|1111|2|333|}{|00|1111|2|333|}

    B = torch.max(batch_index) + 1
    N = batch_index.size(0)  # batch * num_points

    K_index = torch.ones((K, N), dtype=torch.long, device=pos.device ) 
    K_index = torch.cumsum(K_index, dim = 0) - 1
    K_index = torch.cat(list(K_index), dim = 0)                                 # K * num_points      {00}{1111}

    batch_index = batch_index.repeat(K)

    group_index = batch_index + K_index * B

    batch = Batch()

    batch.x = x
    batch.pos = pos
    batch.batch = batch_index
    batch.K = K_index
    batch.group = group_index
    batch.NBpK = N

    return batch

def repeat_batch(data, dim1, dim2 = 1):
    """
    repeat a batch of sparse points (dim1, dim2) times
    when repeat alone first dim (creating more points), the batch index will also be increased. 
    """
    x, pos, batch_index = data.x, data.pos, data.batch

    if x is not None:
        x = x.repeat(dim1, dim2)
    pos = pos.repeat(dim1, dim2)

    batch_index_offset = (torch.cumsum(torch.ones(dim1, dtype=torch.long, device=pos.device), dim = 0) - 1) * (torch.max(batch_index) + 1)
    batch_index_offset = batch_index_offset.repeat_interleave(data.num_nodes)
    batch_index = batch_index.repeat(dim1) + batch_index_offset

    batch = Batch() 

    batch.x = x
    batch.pos = pos
    batch.batch = batch_index

    return batch

def set_number_points(data, num, batch_size):

    index_list = []

    index = torch.arange(start = 0, end = data.num_nodes, dtype=torch.long, device=data.x.device)

    for i in range(batch_size):

        mask = (data.batch == i)
        
        index_list.append(index[mask][:num])

    index = torch.cat(index_list)

    data.apply(lambda x: x[index])

    return data

def camera_space_project_xyz(from_space, to_space, pts, input_radians = False):
    """
    from_space: the world position and rotation of the camera to be projected
    to_space: the world position and rotation of the camera to project to.
    pts: the point cloud in the "from" camera space.

    return: the point clouf in the "to" camera space.
    The world coordinate system should be a left hand coordinate system.
    """

    d_posrot = from_space - to_space

    R = Y_Rotation_Matrix(d_posrot[4], pts.device, input_radians)
    Rinv = Y_Rotation_Matrix(-to_space[4], pts.device, input_radians)

    dxyz = d_posrot[:4].view(1,4,1)
    dxyz = Rinv.bmm(dxyz)

    T = T_Matrix(*dxyz[0,:3,0], pts.device)

    pts = torch.cat((pts, torch.ones((pts.size(0), 1), device = pts.device)), dim = 1)
    pts = pts.unsqueeze(-1)

    R = R.expand(pts.size(0), -1, -1)
    T = T.expand(pts.size(0), -1, -1)

    pts = R.bmm(pts)
    pts = T.bmm(pts)

    return pts[:,:3, 0]

class Depth2PointCloud(torch.nn.Module):

    def __init__(self, W = 256, H = 256, f = 10, sensor_h = 16, sensor_w = 16, Normalize = False):
        super().__init__()
        """
        W: the width of the image in pixel.
        H: the hight of the image in pixel.
        f, sensor_h, sensor_w: in mm.
        """

        fx = sensor_w / f
        fy = sensor_h / f

        xs = torch.linspace(0, W - 1, W) / float(W - 1) * fx - fx / 2
        ys = torch.linspace(0, H - 1, H) / float(H - 1) * fy - fy / 2

        xs = xs.view(1, 1, 1, W).repeat(1, 1, W, 1)
        ys = ys.view(1, 1, H, 1).repeat(1, 1, 1, H)

        xyzs = torch.cat((xs, -ys, torch.ones(xs.size()), torch.ones(xs.size())), 1).view(1, 4, -1)

        self.register_buffer('xyzs' ,xyzs)

    def forward(self, depth):
        """
        depth: the depth in the format of (batch_size, 1, W, H) or (batch_size, 1, W * H)

        return: xyz of point cloud of (batch_size, 4, num_point)
        """

        if len(depth.shape) == 2:
            depth = depth.view(1, 1, *depth.shape)

        if len(depth.shape) == 3:
            depth = depth.unsqueeze(1)

        if len(depth.shape) == 4:
            depth = depth.view(depth.shape[0], 1, -1)

        pts = depth * self.xyzs
        pts[:, -1, :] = 1

        return pts

def Y_Rotation_Matrix(r, device, input_radians = False):

    if not input_radians:
        r = math.radians(r)
        
    R = torch.tensor(np.array(
    [[[ math.cos(r), 0, math.sin(r), 0],
      [ 0,           1, 0,           0],
      [-math.sin(r), 0, math.cos(r), 0],
      [ 0,           0, 0,           1]]], dtype=np.float32), device = device)

    return R

def T_Matrix(x, y, z, device):
    T = torch.tensor(np.array(
    [[[ 1, 0, 0, x],
      [ 0, 1, 0, y],
      [ 0, 0, 1, z],
      [ 0, 0, 0, 1]]], dtype=np.float32), device = device)

    return T

def scatter_sample(batch, num_sample = 1):
    """
    sample num_sample samples from each batch
    if there is not enough samples in one batch, error will be arised
    return the sampled index

    if want to take average of element in each batch, use avg_pool
    """

    if batch.dim() == 2:
        batch = batch.squeeze(1)

    assert batch.dim() == 1, "batch size assertion triggered in scatter_scample"

    unique = torch.unique(batch.squeeze())
    
    weight = batch.unsqueeze(0) == unique.unsqueeze(1)

    return torch.multinomial(weight.float(), num_sample=num_sample)

def avg_pool(cluster, data):
    """
    avg_pool modified from GridSampling
    the pooling results of labels identified with name y or Id or id will be decided with majority votes
    other features will be averaged
    """

    cluster, perm = consecutive_cluster(cluster)

    num_nodes = data.num_nodes

    for key, item in data:

        if torch.is_tensor(item) and item.size(0) == num_nodes:
            if key == 'y' or key == 'Id' or key == 'id' or key == 'layer' or key == 'Layer':
                # majority vote
                item = F.one_hot(item)
                item = scatter_sum(item, cluster, dim=0)
                data[key] = item.argmax(dim=-1)
            elif key == 'batch':
                data[key] = item[perm]
            else:
                assert item.dtype != torch.long, "assertion error triggered in avg_pool"
                data[key] = scatter_mean(item, cluster, dim=0)

    return data, cluster, perm

def scatter_mean_with_batch(pos, batch):

    pos_center = scatter_mean(pos, batch)
    pos_center_batch = torch.arange(pos_center.size(0), dtype=torch.long, device=pos.device)

    return pos_center, pos_center_batch

def voxel_mean_pool(pos, batch, start, end, size): # * sorted

    # return results sorted according to voxel_index
    
    voxel_cluster = voxel_grid(pos, batch, size = size, start = start, end = end)
    voxel_cluster, perm, inv, sorted_index = consecutive_cluster_sorted(voxel_cluster)

    batch = batch[inv]
    pos = pos[inv]

    # pos_sample is listed in voxel cluster index increaing order
    pos_sample = scatter_mean(pos, voxel_cluster, dim = 0)
    batch_sample = batch[perm]
    voxel_cluster_sample = torch.arange(pos_sample.size(0), dtype=torch.long, device=pos.device)
    
    # each pos is assigned to one and only one cluster
    out_index = voxel_cluster
    in_index = torch.arange(pos.size(0), dtype=torch.long, device=pos.device)

    return (out_index, in_index), pos, batch, pos_sample, batch_sample, voxel_cluster, voxel_cluster_sample, inv

def consecutive_cluster_sorted(index):
    """
    make index consecutive and return them in non-decreasing order
    if returned index is not requred to be sorted, use consecutive cluster instead
    """

    sorted_index, inv = torch.sort(index) # get sorted index

    unique, cluster = torch.unique_consecutive(sorted_index, return_inverse = True)

    perm = torch.arange(cluster.size(0), dtype=cluster.dtype, device=cluster.device)

    perm = cluster.new_empty(unique.size(0)).scatter_(0, cluster, perm)

    # inv[perm] = unique ?
    return cluster, perm, inv, sorted_index

def assert_sorted_consecutive(index):

    diff = index[1:] - index[:-1]

    assert ((diff == 0) | (diff == 1)).all()

def assert_arange(index):

    _index = torch.arange(index.size(0), dtype=torch.long, device=index.device)

    assert torch.equal(index, _index)

def linear_anneal(m, n, p, q, i):
    """
    linearly anneal parameter from m to n, starting from iteration p until step q
    return True if annealing is active, False otherwise
    """

    if q - p == 0:
        return False, n

    d = (n - m)/(q - p)

    if i >= p and i <= q:

        return True, m + d * (i - p)

    else:

        return False, m if i < p else n

def compute_performance(path, iter_idx, max_radius,
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
                bg_chamfer_predict__pos,
                majority_vote_flag = False,
                compute_MMD_CD = False):

    # get all glimpse prediction from the first batch
    glimpse_predict__batch = glimpse__batch[glimpse_predict__glimpse_index]
    index = (glimpse_predict__batch == 0)

    # get all ground truth from the first batch
    index = (batch == 0)
    full_point_index = torch.arange(pos.size(0), dtype=torch.long, device=pos.device)
    pos = pos[index]
    point_index = full_point_index[index]
    bg_log_alpha = bg_log_alpha[index]
    alpha = torch.exp(bg_log_alpha)
    bg_chamfer_predict__pos = bg_chamfer_predict__pos[index]

    # get all glimpse member ground truth from the first batch
    index = (glimpse_member__batch == 0)
    glimpse_member__glimpse_index = glimpse_member__glimpse_index[index]
    glimpse_member__normalized_log_alpha = glimpse_member__normalized_log_alpha[index]
    glimpse_member__alpha = torch.exp(glimpse_member__normalized_log_alpha)
    glimpse_member__center = glimpse__center[glimpse_member__glimpse_index]
    glimpse_member__point_index = glimpse_member__point_index[index]
    glimpse_chamfer_predict__global_pos = glimpse_chamfer_predict__local_pos[index] * max_radius + glimpse_member__center

    # compute the segmentation result
    _, segmentation = scatter_max(torch.cat((glimpse_member__alpha, alpha), dim = 0), torch.cat((glimpse_member__point_index, point_index), dim = 0), dim = 0)
    segmentation = torch.cat((glimpse_member__glimpse_index, point_index.new_full(point_index.size(), torch.max(glimpse_member__glimpse_index) + 1)), dim = 0)[segmentation]

    if majority_vote_flag:
        segmentation = majority_vote(segmentation, pos)
    else:
        segmentation, _ = consecutive_cluster(segmentation)
    
    # compute fg_msc and fg_sc
    if Id is not None:
        sc, msc = average_segcover(Id.to(segmentation.device), segmentation)
    else:
        sc = 0
        msc = 0

    # compute final reconstruction result
    all_predict = torch.cat((glimpse_chamfer_predict__global_pos, bg_chamfer_predict__pos), dim = 0)
    all_log_alpha = torch.cat((glimpse_member__normalized_log_alpha, bg_log_alpha))
    all_point_index = torch.cat((glimpse_member__point_index, point_index), dim = 0)

    # for numerical stability, compute in log space
    pos_reconstruct = torch.exp(scatter_logsumexp(torch.log(2 + all_predict + 1e-12) + all_log_alpha[:, None], all_point_index, dim = 0)) - 2

    MMD_CD_F = None
    MMD_CD_B = None

    # pos_reconstruct, pos, Id, segmentation

    if compute_MMD_CD:
        y, x = knn(pos_reconstruct, pos, 1)

        MMD_CD_F = torch.norm(pos - pos_reconstruct[x], p=None, dim = 1)
        MMD_CD_F = torch.mean(MMD_CD_F).item()

        x, y = knn(pos, pos_reconstruct, 1)

        MMD_CD_B = torch.norm(pos_reconstruct - pos[y], p=None, dim = 1)
        MMD_CD_B = torch.mean(MMD_CD_B).item()

    # ! only work when batch_size = 1
    if Id is not None:
        ARI = adjusted_rand_score(labels_true=Id, labels_pred=segmentation)
    else:
        ARI = 0

    return segmentation, ARI, sc, msc, MMD_CD_F, MMD_CD_B

def batch_statistic(data, batch, opt = 'mean'):

    if opt == 'mean':
        data = scatter_mean(data, batch, dim = 0)
    elif opt == 'sum':
        data = scatter_sum(data, batch, dim = 0)

    data = torch.mean(data, dim = 0)
    return data

def majority_vote(segmentation, pos, k = 5):

    segmentation, _ = consecutive_cluster(segmentation)

    x, y = knn_graph(pos, k)

    knn_seg = segmentation[x]

    knn_seg = F.one_hot(knn_seg)

    knn_seg = scatter_sum(knn_seg, y, dim = 0)

    segmentation = knn_seg.argmax(dim = -1)

    return segmentation

def iou_binary(mask_A, mask_B, debug=False):
    if debug:
        assert mask_A.shape == mask_B.shape
        assert mask_A.dtype == torch.bool
        assert mask_B.dtype == torch.bool
    intersection = (mask_A * mask_B).sum()
    union = (mask_A + mask_B).sum()
    # Return -100 if union is zero, else return IOU
    return torch.tensor(-100.0) if union == 0 else intersection.float() / union.float()

def average_segcover(Id, Seg):
    """
    Modified from Genesis
    Covering of Id by Seg
    Id.shape = [n_points]
    Seg.shape = [n_points]
    scale: If true, take weighted mean over IOU values proportional to the the number of pixels of the mask being covered.
    Assumes labels in Id and Seg are non-negative integers.
    Negative labels will be ignored.
    """
    # remove background, background will have the most number of points
    unique, counts = torch.unique(Id, return_counts = True)
    bg_index = unique[torch.argmax(counts)]
    fg_mask = Id != bg_index

    if not torch.any(fg_mask):
        return 1, 1

    Id = Id[fg_mask]
    Seg = Seg[fg_mask]

    assert Id.shape == Seg.shape

    bsz = Id.shape[0]
    N = 0.0
    sc_scores = 0 # scale == True
    msc_scores = 0
    # Loop over Id
    for i in range(Id.max().item() + 1):
        binaryA = Id == i
        N = N + 1 if binaryA.sum() > 0 else N
        if not binaryA.any():
            continue
        max_iou = torch.tensor(0.0)
        # Loop over Seg to find max IOU
        for j in range(Seg.max().item() + 1):
            binaryB = Seg == j
            if not binaryB.any():
                continue
            iou = iou_binary(binaryA, binaryB)
            max_iou = iou if iou > max_iou else max_iou
        # Scale
        sc_scores += binaryA.sum().float() * max_iou
        msc_scores += max_iou

    nonignore = Id >= 0
    sc_coverage = sc_scores / nonignore.sum().float()
    msc_coverage = msc_scores / N
    # Sanity check
    assert (0.0 <= sc_coverage).all() and (sc_coverage <= 1.0).all() and (0.0 <= msc_coverage).all() and (msc_coverage <= 1.0).all()
    # Take average over batch dimension
    sc_coverage = sc_coverage.item()
    msc_coverage = msc_coverage.item()
    return sc_coverage, msc_coverage

class UnitVariance(object):
    r"""Normalize point cloud to get unit-variance globally and centers node positions around the origin."""

    def __call__(self, data):
        data.pos = data.pos - data.pos.mean(dim=-2, keepdim=True)
        std = torch.std(data.pos)
        data.pos /= std
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def mIOU(Id, Seg, Layer, cls = 1):

    # unique, counts = torch.unique(Seg, return_counts = True)
    # bg_seg = unique[torch.argmax(counts)]
    bg_seg = 0

    # unique, counts = torch.unique(Id, return_counts = True)
    bg_index = 0
    fg_mask = Id != bg_index
    Id = Id[fg_mask]
    Seg = Seg[fg_mask]
    Layer = Layer[fg_mask]

    # for i in range(1, Id.max()):

    #     mask = (Id == i)

    #     flag = torch.all(Layer[mask] == Layer[mask][0])

    #     # print(np.all(label[mask] == label[mask][0]))
    #     if not flag:
    #         print(flag)

    assert Id.shape == Seg.shape

    # bsz = Id.shape[0]
    N = 0.0
    # sc_scores = 0 # scale == True
    msc_scores = 0
    # Loop over Id
    for i in range(Id.max().item() + 1):
        binaryA = (Id == i)
        # remov empty class
        if not binaryA.any():
            continue
        # select object type
        if not torch.all(Layer[binaryA] == cls):
            # print('not')
            continue
        N = N + 1 if binaryA.sum() > 0 else N
        max_iou = torch.tensor(0.0)
        max_index = -1
        # Loop over Seg to find max IOU
        for j in range(Seg.max().item() + 1):
            binaryB = (Seg == j)
            if not binaryB.any():
                continue
            iou = iou_binary(binaryA, binaryB)
            # print(iou)
            if iou > max_iou:
                max_iou = iou
                max_index = j

        # print(bg_seg, max_index)
        if max_index == bg_seg or max_index == -1:
            N = N - 1
            max_iou = 0

        msc_scores += max_iou

    # nonignore = Id >= 0
    msc_coverage = msc_scores
    if msc_scores == 0:
        return 0, N
    
    msc_coverage = msc_coverage.item()
    return msc_coverage, N
