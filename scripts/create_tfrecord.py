import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets.utils import (convert_to_tfrecord, 
                            decode_fn, 
                            sample_view, 
                            convert_to_pytorch,
                            convert_to_tfrecord_multiview,
                            decode_fn_multiview)

data_dir = None
output_dir = None

convert_to_tfrecord_multiview(data_dir, output_dir, voxel_size = 0.2)
