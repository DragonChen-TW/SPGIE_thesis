import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import numpy as np
from skimage.segmentation import slic
import networkx as nx
# 
import sys
sys.path.append('..')
from spdataset import (
    MNISTSuperPixelDataset, MNISTMSuperPixelDataset,
    FASHIONMNISTSuperPixelDataset,
    CIFAR10SuperPixelDataset,
    SVHNSuperPixelDataset,
)
from models.jit_drn_model import DynamicReductionNetworkJit

def load_dataset(
    dataset_name: str, train: bool = True,
    num_superpixel: int = 75,
) -> torch.utils.data.Dataset:
    path = f'~/data/{dataset_name.upper()}_SUPERPIXEL'
    if num_superpixel != 75:
        path += f'_{num_superpixel}'
    
    transform = None
    if dataset_name == 'mnist':
        dataset_cls = MNISTSuperPixelDataset
    elif dataset_name == 'mnist_m':
        dataset_cls = MNISTMSuperPixelDataset
    elif dataset_name == 'fashion_mnist':
        dataset_cls = FASHIONMNISTSuperPixelDataset
    elif dataset_name == 'cifar10':
        dataset_cls = CIFAR10SuperPixelDataset
    elif dataset_name == 'svhn':
        dataset_cls = SVHNSuperPixelDataset
    
    dataset = dataset_cls(
        path, train=train,
        transform=transform,
    )
    return dataset

class Net(torch.nn.Module):
    def __init__(self,
        input_dim: int, hidden_dim: int, output_dim: int,
        drn_k: int, aggr: str, pool: str,
        graph_features: int = 3,
    ):
        super(Net, self).__init__()
        self.drn = DynamicReductionNetworkJit(
            input_dim=input_dim, hidden_dim=hidden_dim,
            output_dim=output_dim,
            k=drn_k, aggr=aggr,
            pool=pool,
            agg_layers=2, mp_layers=2, in_layers=3, out_layers=3,
            graph_features=graph_features,
        )

    def forward(self, data):
        logits = self.drn(data.x, data.batch, None) # TO CHECK
        return F.log_softmax(logits, dim=1)

def load_model_with_ckpt(
    input_dim: int = 5, hidden_dim: int = 20, output_dim: int = 10,
    drn_k: int= 4, aggr: str = 'add', pool: str = 'max',
    graph_features: int = 3,
    device: torch.device = torch.device('cpu'),
    ckpt_name: str = '',
) -> torch.nn.Module:
    model = Net(
        input_dim=input_dim, hidden_dim=hidden_dim,
        output_dim=output_dim,
        drn_k=drn_k, aggr=aggr, pool=pool,
    )
    model.to(device)
    if ckpt_name != '':
        ckpt = torch.load(ckpt_name, map_location=device)
        model.load_state_dict(ckpt)
    return model

if __name__ == '__main__':
    dataset = load_dataset('mnist', train=True)

    hidden_dim = 20
    ckpt_name = '~/SPGIE_thesis/mlruns/0/1c9779a75ea94423be73437ac34200df/artifacts/best.pt'

    aggr = 'add'
    pool = 'max'
    model = load_model_with_ckpt(
        input_dim=dataset.num_features,
        hidden_dim=hidden_dim,
        output_dim=dataset.num_classes,
        drn_k=4, aggr=aggr, pool=pool,
    )
def make_one_graph(img, channel_axis=None, num_superpixel=75):
    """Inputed images should be reshaped to (height, width, channel) first."""
    img = img.numpy()
    segments = slic(
        img, n_segments=num_superpixel,
        compactness=0.1,
        slic_zero=True, start_label=0, channel_axis=channel_axis,
    ).squeeze()
    num_nodes = np.max(segments) + 1
    node2map = {n: (segments == n) for n in range(num_nodes)}

    height = img.shape[0]
    width = img.shape[1]

    xy_pos = np.mgrid[0:1:1/height, 0:1:1/width]
    xy_pos = np.transpose(xy_pos, (1, 2, 0))

    # two liner
    pos_rgbs = [(
        xy_pos[segments == node, :],
        img[segments == node, :],
    ) for node in range(num_nodes)]
    node_features = np.array([np.concatenate(
        (
            np.mean(pos, axis=0),
            np.mean(rgb, axis=0),
        )
    ) for pos, rgb in pos_rgbs])

    bneighbors = np.unique(np.block([
        [segments[:, :-1].ravel(), segments[:-1, :].ravel(),], # vs_right
        [segments[:, 1:].ravel(), segments[1:, :].ravel()], # vs_below
    ]), axis=1)

    G = nx.Graph() # TO DEL
    for i in range(bneighbors.shape[1]): # TO DEL
        if bneighbors[0, i] != bneighbors[1, i]:
            G.add_edge(bneighbors[0, i], bneighbors[1, i])
    for node in range(num_nodes): # TO DEL
        G.add_edge(node, node)

    # TODO replace aboce section using np.unique

    edge_index = np.swapaxes(
        np.array([e for e in G.edges])
    , 0, 1)

    graph = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        pos=torch.tensor(node_features[:, :2], dtype=torch.float32),
        edge_index=torch.tensor(edge_index),
    )

    return graph, node2map

if __name__ == '__main__':
    gray_img = torch.rand((1, 28, 28))
    color_img = torch.rand((3, 28, 28))
    
    _ = make_one_graph(torch.permute(gray_img, (1, 2, 0)), channel_axis=None)
    _ = make_one_graph(torch.permute(color_img, (1, 2, 0)), channel_axis=2)