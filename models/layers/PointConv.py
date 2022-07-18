import torch
import torch.nn.functional as F

from torch_geometric.nn import radius, knn_interpolate
from torch_geometric.utils import to_dense_batch

from torch_scatter import scatter_sum

class PointConv(torch.nn.Module):

    def __init__(self, radius = 0.25/16, max_num_neighbors = 64, c_in = 1, c_mid = 64, c_out = 64, pos_dim = 3):
        super().__init__()
        # save \frac{C_mid}{K x C_out}

        self.radius = radius
        self.max_num_neighbors = max_num_neighbors

        self.c_mid = c_mid
        self.c_in = c_in

        self.mlp1 = torch.nn.Linear(in_features = pos_dim, out_features = 16, bias = False)
        self.mlp2 = torch.nn.Linear(in_features = 16, out_features = c_mid, bias = False)
        self.mlp3 = torch.nn.Linear(in_features = c_mid * c_in, out_features = c_out)

    def forward(self, x_in, pos_in, batch_in, pos_out = None, batch_out = None, in_index = None, out_index = None):

        if pos_out is None:
            pos_out = pos_in
            batch_out = batch_in

        if in_index is None or out_index is None: # ! if all y has at least one x in radius r, then out_index is sorted and consecutive.
            out_index, in_index = radius(pos_in, pos_out, self.radius, batch_in, batch_out, max_num_neighbors=self.max_num_neighbors)

        # # ! Debug:
        # assert_sorted_consecutive(out_index)
        # # ! Debug:

        num_points = scatter_sum(torch.ones(out_index.shape, device=out_index.device), out_index, dim = 0)

        pos_i = pos_out[out_index]
        pos_j = pos_in[in_index]
        pos_local = pos_j - pos_i
        
        pos_local_dense, mask = to_dense_batch(pos_local, out_index) # ! out_index should be sorted

        if x_in is None:
            x_in_dense = torch.ones((pos_local_dense.size(0), pos_local_dense.size(1), 1), device=pos_in.device)
            x_in_dense[torch.logical_not(mask)] = 0
        else:
            # if assertion is triggered, some points are not covered by any points.
            # assert torch.max(in_index) == (x_in.size(0) - 1)
            x_in = x_in[in_index]/num_points[out_index].unsqueeze(1)  # torch.max(in_index) > x_in.size(0) - 1
            x_in_dense, _ = to_dense_batch(x_in, out_index) # grounp x_in with out_index

        # # ! Debug:
        # assert x_in_dense.size(0) == torch.unique(out_index, return_inverse = False, return_counts=False).size(0)
        # assert pos_local_dense.size(0) == torch.unique(out_index, return_inverse = False, return_counts=False).size(0)
        # # ! Debug:

        # ! bias term will still be added to padded element.
        # batch_dense, _ = to_dense_batch(batch_out[out_index], out_index, fill_value = -1)

        M = F.celu(self.mlp1(pos_local_dense))
        M = F.celu(self.mlp2(M))

        product = torch.bmm(x_in_dense.permute(0, 2, 1), M)

        product = torch.flatten(product, start_dim=1)

        out = self.mlp3(product)

        # # ! Debug:
        # assert out.size(0) == pos_out.size(0)
        # # ! Debug:
        
        return out

class PointDeconv(torch.nn.Module):

    def __init__(self, radius, max_num_neighbors, c_in = 1, c_mid = 64, c_out = 64, pos_dim = 3, k = 3):
        super().__init__()

        self.conv = PointConv(radius, max_num_neighbors, c_in, c_mid, c_out, pos_dim)
        self.k = k

    def forward(self, x_in, pos_in, batch_in, pos_out, batch_out):

        x_out = knn_interpolate(x_in, pos_in, pos_out, batch_in, batch_out, k = self.k)

        out = self.conv(x_out, pos_out, batch_out, pos_out, batch_out)

        return out

class CenterShift(torch.nn.Module):

    def __init__(self, c_in = 1, c_mid = 64, c_out = 64, pos_dim = 3):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.mlp11 = torch.nn.Linear(in_features = pos_dim, out_features = 16)
        self.mlp12 = torch.nn.Linear(in_features = 16, out_features = c_mid)
        self.mlp13 = torch.nn.Linear(in_features = c_mid, out_features = c_out * c_in)

    def forward(self, x, pos_i, pos_j):
        """
        move points from pos_i to pos_j
        x: N x c_in
        """
        
        # ! Debugging:
        assert pos_i.size(0) == pos_j.size(0)
        # ! Debugging:

        pos_local = pos_j - pos_i

        W = F.celu(self.mlp11(pos_local))
        W = F.celu(self.mlp12(W))
        W = self.mlp13(W)
        W = W.view(-1, self.c_in, self.c_out)

        out = torch.bmm(x.unsqueeze(1), W).squeeze() # N, c_out

        return out