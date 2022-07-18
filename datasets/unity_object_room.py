import os
import sys
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import random
import numpy as np

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    pass

import torch
from torch.utils.data import Dataset

import tensorflow as tf

from datasets.utils import (convert_to_pytorch, decode_fn_multiview)


class UORMVTFDataset():

    def __init__(self, path, mode = "train", num_workers = 4, batch_size = 12, shuffle_buffer_size = 2000, prefetch_buffer_size = 100, drop_remainder = True, scale = 8):
        super().__init__()

        path = os.path.join(path, mode)
        tfrs = os.listdir(path)
        self.scale = 8

        self._dataset = tf.data.TFRecordDataset([os.path.join(path, i) for i in tfrs])
        if mode == "test":
            self._dataset = self._dataset.repeat(1)
        else:
            self._dataset = self._dataset.repeat()
            self._dataset = self._dataset.shuffle(buffer_size = shuffle_buffer_size, reshuffle_each_iteration = True)
        self._dataset = self._dataset.map(decode_fn_multiview, num_parallel_calls = num_workers)
        self._dataset = self._dataset.batch(batch_size, drop_remainder = drop_remainder)
        self._dataset = self._dataset.map(convert_to_pytorch, num_parallel_calls = num_workers)
        self._dataset = self._dataset.prefetch(buffer_size = prefetch_buffer_size)
        self.iterator = iter(self._dataset)


    def __next__(self):

        _xyz, _rgb, _batch, _Id, _layer = self.iterator.get_next()
        xyz = torch.from_numpy(_xyz.numpy())
        xyz = xyz.pin_memory()
        rgb = torch.from_numpy(_rgb.numpy())
        rgb = rgb.pin_memory()
        # nxyz = torch.from_numpy(_nxyz.numpy())
        # nxyz = nxyz.pin_memory()
        batch = torch.from_numpy(_batch.numpy())
        batch = batch.pin_memory()
        Id = torch.from_numpy(_Id.numpy())
        Id = Id.pin_memory()
        layer = torch.from_numpy(_layer.numpy())
        layer = layer.pin_memory()

        xyz[:, 2] -= 7/16
        xyz = xyz * 16/8

        return (xyz, rgb, batch, Id, layer)

    def __iter__(self):
        return self         

class S3DISDataset(Dataset):
    def __init__(self, path, mode = "train"):

        self.path = os.path.join(path, mode)
        self.data = os.listdir(self.path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        d = np.load(os.path.join(self.path, self.data[idx]))

        # print(list(d.keys()))

        pos_ = d['pos']
        pos = np.zeros_like(pos_)
        # apply marrior along x and z
        x_flag = random.randint(0, 1)
        z_flag = random.randint(0, 1)
        pos[:,0] = pos_[:,0] if x_flag == 0 else -pos_[:,0]
        pos[:,1] = pos_[:,2]
        pos[:,2] = pos_[:,1] if z_flag == 0 else -pos_[:,1]

        Id = d['Id']
        Layer = d['Layer']

        pos = torch.tensor(pos, dtype=torch.float32)
        Id = torch.tensor(Id)
        Layer = torch.tensor(Layer)

        # rad = random.uniform(-math.pi, math.pi)
        # cos = math.cos(rad)
        # sin = math.sin(rad)
        # y_m = torch.tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
        # pos = torch.einsum('nd, dk->nk', pos, y_m)

        batch = torch.zeros_like(Id)

        # marker_data0 = go.Scatter3d(
        #     x = pos[:,0],
        #     y = pos[:,1],
        #     z = pos[:,2],
        #     marker=go.scatter3d.Marker(size = 5, color=Layer),
        #     opacity=0.8,
        #     mode = 'markers'
        # )

        # fig = go.Figure(data = marker_data0, layout = dict(scene = dict(
        #                                                                 zaxis = dict(title = 'z', range = [-1,1]),
        #                                                                 yaxis = dict(title = 'y', range = [-1,1]),
        #                                                                 xaxis = dict(range = [-1, 1]),
        #                                                                 aspectratio = dict(x=3,y=3,z=3)
        #                                                             )))
        # fig.show()

        # raise Exception

        return pos, Id, batch, Layer
        # return pos, Id, batch
