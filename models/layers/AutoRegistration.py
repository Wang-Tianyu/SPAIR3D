import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset

def MLP(in_features, hidden_features, out_features, batch_norm = False):

    if batch_norm:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            torch.nn.BatchNorm1d(hidden_features),
            torch.nn.CELU(),        
            torch.nn.Linear(hidden_features, out_features)
        )
    else:
        return torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden_features),
        torch.nn.CELU(),        
        torch.nn.Linear(hidden_features, out_features)
    )

class AutoRegistrationLayer(torch.nn.Module):
    """
    h: delta pos MLP
    f: nodes MLP
    g: aggregate MLP
    """
    def __init__(self, x_dim = 3, pos_dim = 3, h_hidden = 32, f_hidden = 64, f_out = 64, g_hidden = 64, g_out = 64, end_relu = True):
        super().__init__()
        # h: x_dim
        # f: h_out_dim + x_dim
        # g: f_out_dim

        self.ar = AutoRegistration(g = MLP(f_out + x_dim, g_hidden, g_out), f = MLP(x_dim + pos_dim, f_hidden, f_out), h = MLP(x_dim, h_hidden, pos_dim), end_relu = end_relu)

    def forward(self, x, pos, edge_index):

        return self.ar(x, pos, edge_index), pos, edge_index

class AutoRegistration(MessagePassing):
    """
    h: delta pos MLP
    f: nodes MLP
    g: aggregate MLP
    """
    
    def __init__(self, g, f, h, end_relu):
        super().__init__(aggr='max')

        self.g = g
        self.f = f
        self.h = h
        self.end_relu = end_relu

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.g)
        reset(self.f)
        reset(self.h)

    def forward(self, x, pos, edge_index):
        
        out = self.propagate(edge_index, x=x, pos=pos)

        if self.end_relu:
            return F.celu(out)

        return out

    def message(self, x_i, x_j, pos_i, pos_j):

        delta_pos_i = F.relu(self.h(x_i))
        delta_pos = pos_j - pos_i + delta_pos_i

        delta_pos = pos_j - pos_i

        return F.relu(self.f(torch.cat([x_j, delta_pos], dim = 1)))
    

    def update(self, aggr_out, x, pos):

        if self.g is not None:
            aggr_out = self.g(torch.cat([aggr_out, x], dim = -1))

        return aggr_out

    def __repr__(self):
        return '{}(g={}, f={}, h={})'.format(self.__class__.__name__, self.g, self.f, self.h)
