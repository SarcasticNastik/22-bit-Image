import torch as T
from torch.utils.data import Dataset
import numpy as np
import os
import pickle as pkl


def init_pointcloud_loader(num_points):
    Z = np.random.rand(num_points) + 1.
    h, w = np.random.uniform(10., 214., size=(num_points,)), np.random.uniform(10., 214., size=(num_points,))
    X, Y = (w - 111.5) / 248. * -Z, (h - 111.5) / 248. * Z
    X, Y, Z = np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1)),  np.reshape(Z, (-1, 1))
    XYZ = np.concatenate((X, Y, Z), 1)
    return XYZ.astype('float32')


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def collate(batch):
    data = [b for b in zip(*batch)]
    if len(data) == 3:
        init_pc, imgs, gt_pc = data
    elif len(data) == 4:
        init_pc, imgs, gt_pc, metadata = data
    else:
        raise ValueError('Unknown data values')

    init_pc, imgs, gt_pc = T.from_numpy(np.array(init_pc)).requires_grad_(False), T.from_numpy(np.array(imgs)).requires_grad_(False), [T.from_numpy(pc).requires_grad_(False) for pc in gt_pc]
    if len(data) == 3:
        return (init_pc, imgs, gt_pc)
    else:
        return (init_pc, imgs, gt_pc, metadata)


class ShapeNet(Dataset):
    def __init__(self, path, grayscale=None, type='train', n_points=2000, **kwargs):
        assert type in ('train', 'valid', 'test')
        self.n_points, self.grayscale, self.file_list, self.path,  = n_points,  grayscale,  os.listdir(path), path
        if type in ('train', 'test'):
            self.type = type
        elif type == "valid":
            self.type = 'test'
        else:
            raise ValueError('Unknown data values')
        self.num_vals = kwargs.pop('num_vals', 30)
        self.pkl_list, self.sample_weights = [], []
        for folder in self.file_list:
            file_path = path+folder+self.type
            if type == 'valid':
                file_path = [file_path[i] for i in np.random.randint(len(file_path), size=self.num_vals // len(self.file_list))]

            file_path = [self.path+folder+self.type+f for f in file_path]
            self.pkl_list.extend(file_path)
            self.sample_weights.extend([1/len(file_path)] * len(file_path))

    def __len__(self):
        return len(self.pkl_list)

    def __getitem__(self, idx):
        pkl_path = self.pkl_list[idx]
        contents = pkl.load(open(pkl_path, 'rb'), encoding='latin1')
        img = rgb2gray(contents[0])[..., None] if self.grayscale else contents[0]
        img = (np.transpose(img / 255.0, (2, 0, 1)) - .5) * 2
        pc = np.array(contents[1], 'float32')[:, :3]
        pc -= np.mean(pc, 0, keepdims=True)
        return init_pointcloud_loader(self.n_points), np.array(img, 'float32'), pc
