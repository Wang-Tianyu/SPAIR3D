import os
import random
import numpy as np
from tqdm import tqdm
import math

import tensorflow as tf
import torch
from torch_geometric.data import Batch
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import scatter_sum

from models.utils import Depth2PointCloud, camera_space_project_xyz, avg_pool

def convert_to_pyg(rgb, pos, Id, norm, layer, rm_sky, normalize = True, ratio = 16):
    """
    rgb: batch_size, num_points, 3    Can be None
    pos: batch_size, 3, num_points
    Id: batch_size, num_points
    """

    bs, _, n = pos.shape

    batch_index = torch.ones((bs, n), dtype=torch.long, device=pos.device)   # batch_size, num_points
    batch_index = torch.cumsum(batch_index, dim = 0) - 1                   # ! need to start at zero and no skip index
    batch_index = torch.cat(list(batch_index), dim = 0)                    # batch_size * num_points

    # batch = torch.arange(batch_size).view(-1, 1).repeat(1, num_nodes).view(-1)

    if rgb is not None:                         # batch_size, num_points, channels
        rgb_list = list(rgb)
        rgb = torch.cat(rgb_list, dim = 0)      # batch_size * num_points, channels

    if Id is not None:
        Id_list = list(Id)
        Id = torch.cat(Id_list, dim = 0)
        
    if layer is not None:
        layer_list = list(layer)
        layer = torch.cat(layer_list, dim = 0)

    if norm is not None:
        norm_list = list(norm)
        norm = torch.cat(norm_list, dim = 0)

    pos = pos.permute(0, 2, 1)          # batch_size, num_points, 3
    pos_list = list(pos)
    pos = torch.cat(pos_list, dim = 0)  # batch_size * num_points, 3

    if rm_sky:
        mask = (pos[:, -1] < 19)
        pos = pos[mask]
        batch_index = batch_index[mask]
        rgb = None if rgb is None else rgb[mask]
        Id = None if Id is None else Id[mask]
        norm = None if norm is None else norm[mask]
        layer = None if layer is None else layer[mask]
        
    if pos is not None and normalize:
        # pos[:, -1] -= 10
        pos = pos/ratio     # x y between [-0.5, 0.5] z between [0, 20/16], 1 unit in unity equal to 1/16 unit in point cloud
        
    pts = Batch() 

    pts.rgb = rgb
    pts.pos = pos
    pts.Id = Id
    pts.norm = norm
    pts.batch = batch_index # * Name it batch for now to suit pyg
    pts.layer = layer
    # pts = batch.contiguous()

    return pts

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_example(rgb, pos, Id, norm, view, n):

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    nx = norm[:,0]
    ny = norm[:,1]
    nz = norm[:,2]

    feature = {
        'x': _float_feature(x),
        'y': _float_feature(y),
        'z': _float_feature(z),
        'r': _float_feature(r),
        'g': _float_feature(g),
        'b': _float_feature(b),    
        'nx': _float_feature(nx),
        'ny': _float_feature(ny),
        'nz': _float_feature(nz),
        'view': _int64_feature(view),
        'Id': _int64_feature(Id),
        'n': _int64_feature(n)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()

def serialize_example_multiview(rgb, pos, Id, layer, norm, n0, n1):

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    # nx = norm[:,0]
    # ny = norm[:,1]
    # nz = norm[:,2]

    feature = {
        'x': _float_feature(x),
        'y': _float_feature(y),
        'z': _float_feature(z),
        'r': _float_feature(r),
        'g': _float_feature(g),
        'b': _float_feature(b),    
        # 'nx': _float_feature(nx),
        # 'ny': _float_feature(ny),
        # 'nz': _float_feature(nz),
        'Id': _int64_feature(Id),
        'layer': _int64_feature(layer),
        'n0': _float_feature([n0]),
        'n1': _int64_feature([n1]),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()

def convert_to_tfrecord(data_path, tfrecord_path, num_pre_tfrecord = 500, start = -1): 
    """
    convert numpy data into tfrecord
    """

    assert os.path.isdir(tfrecord_path), "need a folder path"
    
    if not os.path.exists(tfrecord_path):
        os.mkdir(tfrecord_path)

    if start < 0:
        start = len(os.listdir(tfrecord_path))
    
    print("read from " + data_path)
    print("write to " + tfrecord_path)

    file_list = os.listdir(data_path)

    d2p = Depth2PointCloud().cuda()

    ll = [file_list[i:i+num_pre_tfrecord] for i in range(0, len(file_list), num_pre_tfrecord)]

    for i, l in enumerate(tqdm(ll)):
    # for i, l in enumerate(ll):
        with tf.io.TFRecordWriter(os.path.join(tfrecord_path, str(i + start) + '.tfrecords')) as tf_writter:
            for f in tqdm(l):
            # for f in l:
                try:
                    d = np.load(os.path.join(data_path, f))
                    posrot = d['posrot']                    # 10, 6
                    n_view = posrot.shape[0]
                    # posrot = torch.from_numpy(posrot)
                    # posrot = torch.cat(list(posrot))  # 10 * 6

                    DM = d['DM']                            # 10, 256, 256
                    DM = DM.astype(np.float32)/65535 * 20
                    DM = torch.from_numpy(DM).cuda()
                    DM = DM.unsqueeze(1)                    # channel first 10, 1, 256, 256

                    RGB = d['RGB']       # 10, 256, 256, 3 
                    RGB = RGB/255
                    RGB = torch.from_numpy(RGB).cuda()
                    RGB = RGB.view(n_view, -1, 3)

                    Id = d['Id']         # 10, 256, 256   
                    Id = Id.astype(np.int64)             
                    Id = torch.from_numpy(Id).cuda()
                    Id = Id.view(n_view, -1)

                    Norm = d['Normal']
                    Norm = torch.from_numpy(Norm).cuda()
                    Norm = Norm.view(n_view, -1, 3)

                    xyz = d2p(DM)[:, :3, :] # 10, 3, 256*256
                    pts = convert_to_pyg(rgb = RGB, pos = xyz, Id = Id, layer = None, norm = None, rm_sky = True)

                    # ! scale sensitive, 
                    # ! modified avg_pool is used
                    cluster = voxel_grid(pts.pos, pts.batch, size = 0.15/16) # every unit cube in unity is divided into 10^3 voxel
                    pts = avg_pool(cluster, pts)   # Id will be majority voted, other features will be averaged.

                    n = torch.ones_like(pts.Id)
                    n = scatter_sum(n, pts.batch)

                    if n.size(0) != 10:
                        # check there are in total 10 views
                        continue

                    if not (n != 0).all():
                        # check that every view has at least one point
                        continue

                except EOFError as e:
                    print(e)
                    print(f)
                    print(f)
                    print(f)
                    print(f)
                    print(f)
                    continue

                serialized_example = serialize_example(
                    rgb = pts.rgb.cpu().numpy(), 
                    pos = pts.pos.cpu().numpy(), 
                    Id = pts.Id.cpu().numpy(), 
                    norm = pts.norm.cpu().numpy(),
                    view = pts.batch.cpu().numpy(),
                    n = n.cpu().numpy())

                tf_writter.write(serialized_example)

def convert_to_tfrecord_multiview(data_path, tfrecord_path, num_pre_tfrecord = 500, start = -1, voxel_size = 0.15, with_source_view = False):
    """
    merge multiple view into one and convert numpy data into tfrecord
    """

    assert os.path.isdir(tfrecord_path), "need a folder path"
    
    if not os.path.exists(tfrecord_path):
        os.mkdir(tfrecord_path)

    if start < 0:
        start = len(os.listdir(tfrecord_path))
    
    print("read from " + data_path)
    print("write to " + tfrecord_path)

    file_list = os.listdir(data_path)

    d2p = Depth2PointCloud().cuda()

    ll = [file_list[i:i+num_pre_tfrecord] for i in range(0, len(file_list), num_pre_tfrecord)]

    for i, l in enumerate(tqdm(ll)):

        with tf.io.TFRecordWriter(os.path.join(tfrecord_path, str(i + start) + '.tfrecords')) as tf_writter:
            for f in tqdm(l):
            # for f in l:
                d = np.load(os.path.join(data_path, f))
                number = float(f.split('.n')[0])
                number = math.modf(number)
                number0 = np.float32(number[0])
                number1 = np.int32(number[1])
                posrot = d['posrot']                    # 10, 6
                n_view = posrot.shape[0]
                posrot = torch.from_numpy(posrot).cuda()
                # posrot = torch.cat(list(posrot))  # 10 * 6

                DM = d['DM']                            # 10, 256, 256
                DM = DM.astype(np.float32)/65535 * 20
                DM = torch.from_numpy(DM).cuda()
                DM = DM.unsqueeze(1)                    # channel first 10, 1, 256, 256

                RGB = d['RGB']       # 10, 256, 256, 3 
                RGB = RGB/255
                RGB = torch.from_numpy(RGB).cuda()
                RGB = RGB.view(n_view, -1, 3)

                Id = d['Id']         # 10, 256, 256   
                Id = Id.astype(np.int64)   
                Id = torch.from_numpy(Id).cuda()
                Id = Id.view(n_view, -1)

                Layer = d['SId']
                Layer = Layer.astype(np.int64)
                Layer = torch.from_numpy(Layer).cuda()
                Layer = Layer.view(n_view, -1)

                # if "Normal" in d.keys():
                #     Norm = d['Normal']
                #     Norm = torch.from_numpy(Norm).cuda()
                #     Norm = Norm.view(n_view, -1, 3)
                # else:
                #     Norm = None

                # ? How to merge normal from multiple views?

                xyz = d2p(DM)[:, :3, :] # 10, 3, 256*256

                pts = convert_to_pyg(rgb = RGB, pos = xyz, Id = Id, norm = None, layer = Layer, rm_sky = True, normalize = False)

                if len(pts.pos) == 0:
                    continue

                # ! scale sensitive, 
                if voxel_size > 0:
                    cluster = voxel_grid(pts.pos, pts.batch, size = voxel_size)
                    pts, _, _ = avg_pool(cluster, pts)

                n = torch.ones_like(pts.Id)
                n = scatter_sum(n, pts.batch)

                # if n.size(0) != 10:
                #     # check there are in total 10 views
                #     continue

                if not (n != 0).all():
                    # check that every view has at least one point
                    continue

                # merge different view together

                to_space_index = random.randint(0, 9)

                # if with_source_view:
                #     # save source Id and depth into tfr
                #     source_pts = xyz[to_space_index, :, :] # 3, 356*256
                #     source_Id = Id[to_space_index]         # 256*256

                to_space = posrot[to_space_index]
                for i in range(n_view):

                    if to_space_index == i:
                        continue

                    pts.pos[pts.batch == i] = camera_space_project_xyz(from_space=posrot[i], to_space=to_space, pts=pts.pos[pts.batch == i])

                    # pts.norm[pts.batch == i] = camera_space_project_norm(from_space=posrot[i], to_space=to_space, norm=pts.norm[pts.batch == i])
                    # TODO: figure out normal rotation equation
                
                # normalization
                pts.pos = pts.pos/16

                if voxel_size > 0:
                    cluster = voxel_grid(pts.pos, torch.zeros(pts.pos.size(0), dtype = torch.long, device = pts.pos.device), size = voxel_size/16)
                    pts, _, _ = avg_pool(cluster, pts)

                # fig = plt.figure()
                # ax = fig.add_subplot(1, 1, 1, projection = '3d')
                # downsample = 2
                # ax.scatter(pts.pos[:,2].cpu()[::downsample], pts.pos[:,1].cpu()[::downsample], pts.pos[:,0].cpu()[::downsample], s=1, facecolors = pts.norm.cpu()[::downsample, :])
                # plt.show()

                Id = pts.Id.cpu().numpy()

                serialized_example = serialize_example_multiview(
                    rgb = pts.rgb.cpu().numpy(),
                    pos = pts.pos.cpu().numpy(), 
                    Id = pts.Id.cpu().numpy(), 
                    layer = pts.layer.cpu().numpy(),
                    # norm = pts.pos.cpu().numpy() if pts.norm is None else pts.norm.cpu().numpy(),
                    norm = None,
                    n0=number0,
                    n1=number1)

                tf_writter.write(serialized_example)

def decode_fn(serialized_example):

    features = {
            'x':tf.io.VarLenFeature(dtype=tf.float32),
            'y':tf.io.VarLenFeature(dtype=tf.float32),
            'z':tf.io.VarLenFeature(dtype=tf.float32),
            'r':tf.io.VarLenFeature(dtype=tf.float32),
            'g':tf.io.VarLenFeature(dtype=tf.float32),
            'b':tf.io.VarLenFeature(dtype=tf.float32),
            'nx':tf.io.VarLenFeature(dtype=tf.float32),
            'ny':tf.io.VarLenFeature(dtype=tf.float32),
            'nz':tf.io.VarLenFeature(dtype=tf.float32),
            'view':tf.io.VarLenFeature(dtype=tf.int64),
            'Id':tf.io.VarLenFeature(dtype=tf.int64),
            'n':tf.io.FixedLenFeature([10], dtype=tf.int64)
            }

    example = tf.io.parse_single_example(serialized_example, features)

    example = sample_view(example)

    return example

def decode_fn_multiview(serialized_example):

    features = {
            'x':tf.io.VarLenFeature(dtype=tf.float32),
            'y':tf.io.VarLenFeature(dtype=tf.float32),
            'z':tf.io.VarLenFeature(dtype=tf.float32),
            'r':tf.io.VarLenFeature(dtype=tf.float32),
            'g':tf.io.VarLenFeature(dtype=tf.float32),
            'b':tf.io.VarLenFeature(dtype=tf.float32),
            # 'nx':tf.io.VarLenFeature(dtype=tf.float32),
            # 'ny':tf.io.VarLenFeature(dtype=tf.float32),
            # 'nz':tf.io.VarLenFeature(dtype=tf.float32),
            'Id':tf.io.VarLenFeature(dtype=tf.int64),
            'layer':tf.io.VarLenFeature(dtype=tf.int64),
            }

    example = tf.io.parse_single_example(serialized_example, features)

    return example

# In case want to check the raw data, data trace save the raw data file name as well. 
def decode_fn_multiview_data_trace(serialized_example):

    features = {
            'x':tf.io.VarLenFeature(dtype=tf.float32),
            'y':tf.io.VarLenFeature(dtype=tf.float32),
            'z':tf.io.VarLenFeature(dtype=tf.float32),
            'r':tf.io.VarLenFeature(dtype=tf.float32),
            'g':tf.io.VarLenFeature(dtype=tf.float32),
            'b':tf.io.VarLenFeature(dtype=tf.float32),
            # 'nx':tf.io.VarLenFeature(dtype=tf.float32),
            # 'ny':tf.io.VarLenFeature(dtype=tf.float32),
            # 'nz':tf.io.VarLenFeature(dtype=tf.float32),
            'Id':tf.io.VarLenFeature(dtype=tf.int64),
            'layer':tf.io.VarLenFeature(dtype=tf.int64),
            'n0':tf.io.FixedLenFeature([1], dtype=tf.float32),
            'n1':tf.io.FixedLenFeature([1], dtype=tf.int64)
            }

    example = tf.io.parse_single_example(serialized_example, features)

    return example

def sample_view(example):
    """
    sample one of ten views
    """
    
    view = example['view'].values

    idx = random.randint(0, 9)

    view_index = (view == idx)

    x_coord = tf.boolean_mask(example['x'].values, view_index)

    y_coord = tf.boolean_mask(example['y'].values, view_index)

    z_coord = tf.boolean_mask(example['z'].values, view_index)

    r = tf.boolean_mask(example['r'].values, view_index)

    g = tf.boolean_mask(example['g'].values, view_index)

    b = tf.boolean_mask(example['b'].values, view_index)

    nx_coord = tf.boolean_mask(example['nx'].values, view_index)

    ny_coord = tf.boolean_mask(example['ny'].values, view_index)

    nz_coord = tf.boolean_mask(example['nz'].values, view_index)

    Id = tf.boolean_mask(example['Id'].values, view_index)

    n_points = example['n']
    n_points = tf.gather(n_points, idx)

    indices = tf.expand_dims(tf.range(n_points, dtype = tf.int64), 1)

    x = tf.sparse.SparseTensor(indices = indices, values = x_coord, dense_shape = [n_points])
    y = tf.sparse.SparseTensor(indices = indices, values = y_coord, dense_shape = [n_points] )
    z = tf.sparse.SparseTensor(indices = indices, values = z_coord, dense_shape = [n_points] )
    r = tf.sparse.SparseTensor(indices = indices, values = r, dense_shape = [n_points] )
    g = tf.sparse.SparseTensor(indices = indices, values = g, dense_shape = [n_points] )
    b = tf.sparse.SparseTensor(indices = indices, values = b, dense_shape = [n_points] )
    nx = tf.sparse.SparseTensor(indices = indices, values = nx_coord, dense_shape = [n_points])
    ny = tf.sparse.SparseTensor(indices = indices, values = ny_coord, dense_shape = [n_points] )
    nz = tf.sparse.SparseTensor(indices = indices, values = nz_coord, dense_shape = [n_points] )
    Id = tf.sparse.SparseTensor(indices = indices, values = Id, dense_shape = [n_points] )

    return {'x':x, 'y':y, 'z':z, 'r':r, 'g':g, 'b':b, 'nx':nx, 'ny':ny, 'nz':nz, 'Id':Id}


    # batch =  example['x'].indices.numpy()[:, 0])[view_index]
    # xyz = torch.stack((x_coord, y_coord, z_coord), dim = -1)
    # rgb = torch.stack((r, g, b), dim = -1)
    # xyz.data.pin_memory()
    # rgb.data.pin_memory()
    # batch.data.pin_memory()
    # Id.data.pin_memory()

def convert_to_pytorch_data_trace(example):

    x = tf.expand_dims(example['x'].values, 1)
    y = tf.expand_dims(example['y'].values, 1)
    z = tf.expand_dims(example['z'].values, 1)
    r = tf.expand_dims(example['r'].values, 1)
    g = tf.expand_dims(example['g'].values, 1)
    b = tf.expand_dims(example['b'].values, 1)
    # nx = tf.expand_dims(example['nx'].values, 1)
    # ny = tf.expand_dims(example['ny'].values, 1)
    # nz = tf.expand_dims(example['nz'].values, 1)

    n0 = example['n0']
    n1 = example['n1']

    Id = example['Id'].values
    layer = example['layer'].values
    batch = example['Id'].indices[:, 0]

    xyz = tf.concat([x, y, z], axis = 1)
    rgb = tf.concat([r, g, b], axis = 1)
    # nxyz = tf.concat([nx, ny, nz], axis = 1)

    # return {'xyz':xyz, 'rgb':rgb, 'batch':batch, 'Id':Id}
    return (xyz, rgb, batch, Id, layer, n0, n1)  

def convert_to_pytorch(example):

    x = tf.expand_dims(example['x'].values, 1)
    y = tf.expand_dims(example['y'].values, 1)
    z = tf.expand_dims(example['z'].values, 1)
    r = tf.expand_dims(example['r'].values, 1)
    g = tf.expand_dims(example['g'].values, 1)
    b = tf.expand_dims(example['b'].values, 1)
    # nx = tf.expand_dims(example['nx'].values, 1)
    # ny = tf.expand_dims(example['ny'].values, 1)
    # nz = tf.expand_dims(example['nz'].values, 1)

    Id = example['Id'].values
    layer = example['layer'].values
    batch = example['Id'].indices[:, 0]

    xyz = tf.concat([x, y, z], axis = 1)
    rgb = tf.concat([r, g, b], axis = 1)
    # nxyz = tf.concat([nx, ny, nz], axis = 1)
 
    return (xyz, rgb, batch, Id, layer)


    x = tf.expand_dims(example['x'].values, 1)
    y = tf.expand_dims(example['y'].values, 1)
    z = tf.expand_dims(example['z'].values, 1)

    Id = example['Id'].values
    layer = example['layer'].values
    batch = example['Id'].indices[:, 0]

    xyz = tf.concat([x, y, z], axis = 1)

    return (xyz, batch, Id, layer)