import numpy as np
import torch
from torchvision import transforms
# from torch_geometric.data import Data
import torch.multiprocessing as mp
from skimage.segmentation import slic

from torchvision.datasets import SVHN, CIFAR10, FashionMNIST
from ori_mnistm import MNISTM
# 

import matplotlib as plt
from matplotlib.lines import Line2D
from PIL import Image
import networkx as nx

import os
from tqdm import tqdm
import time
import argparse

DATA_DIR = '~/data'

def make_one_graph(img, channel_axis=None, num_superpixel=75):
    """Inputed images should be reshaped to (height, width, channel) first."""
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    segments = slic(
        img, n_segments=num_superpixel,
        compactness=1,
        channel_axis=channel_axis,
        slic_zero=True, start_label=0,
    ).squeeze()
    num_nodes = np.max(segments) + 1

    height = img.shape[0]
    width = img.shape[1]

    xy_pos = np.mgrid[0:1:1/height, 0:1:1/width]
    xy_pos = np.transpose(xy_pos, (1, 2, 0))

    # print('xy', xy_pos.shape)
    # print('seg', segments.shape)

    # two liner
    pos_rgbs = [(
        xy_pos[segments == node, :],
        img[segments == node, :],
    ) for node in range(num_nodes)]
    node_features = np.array([np.concatenate(
        (
            np.mean(pos, axis=0),
            np.mean(rgb, axis=0),
            np.std(rgb, axis=0),
            np.std(pos, axis=0),
        )
    ) for pos, rgb in pos_rgbs])

    # ===== make edge =====
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
    
    # TODO replace above section using np.unique
    
    edge_index = np.swapaxes(
        np.array([e for e in G.edges])
    , 0, 1)

    return node_features, edge_index

def save_data(graphs, fname):
    path = os.path.expanduser(
        os.path.join(DATA_DIR, OUTPUT_DIR)
    )
    os.makedirs(path, exist_ok=True)
    print(f'Save to {path}{fname}.pt')
    torch.save(graphs, f'{path}{fname}.pt')

def make_mnistm(is_train=True, multi_processing=True, num_superpixel=75):
    global OUTPUT_DIR
    DATASET_DIR = f'{DATA_DIR}/mnist_m/'
    OUTPUT_DIR = 'MNIST_M_SUPERPIXEL_WSTD/'

    phase = 'train' if is_train else 'test'
    dataset = MNISTM(DATASET_DIR, mode=phase)
    print(f'Making {DATASET_DIR.replace(DATA_DIR, "")} [{phase}] datatset......')
    images = dataset.get_all_imgs()
    # batch, channel, h, w
    images = np.transpose(images, (0, 2, 3, 1))

    t = time.time()
    if multi_processing:
        print('mp')
        num_threads = 4
        with mp.Pool(num_threads) as p:
            graph_datas = p.starmap(
                make_one_graph,
                [(img, 2, num_superpixel) for img in images],
            )
        # check is ordered
        # assert [q[0] for q in graph_datas] == list(range(len(graph_datas)))
    else:
        print('no mp')
        loader = tqdm(images, total=len(dataset))
        graph_datas = [
            make_one_graph(img, channel_axis=2, num_superpixel=num_superpixel)
            for img in loader
        ]
    
    t = time.time() - t
    print('fininsh', 't', t)

    label = np.array(dataset.labels, dtype=np.uint8)

    save_data([g[0] for g in graph_datas], f'{phase}_node')
    save_data([g[1] for g in graph_datas], f'{phase}_edge')
    save_data(label, f'{phase}_label')

# def make_fashion_mnist(is_train=True, multi_processing=True, num_superpixel=75):
#     global OUTPUT_DIR
#     DATASET_DIR = f'{DATA_DIR}/SVHN/'
#     OUTPUT_DIR = 'SVHN_SUPERPIXEL/'
#     if num_superpixel != 75:
#         OUTPUT_DIR.replace('SUPERPIXEL', f'{num_superpixel}_SUPERPIXEL')
#     is_download = False

#     phase = 'train' if is_train else 'test'
#     print(f'Making [{phase}] datatset......')
#     dataset = SVHN(DATASET_DIR, download=is_download, split=phase)
#     images = dataset.data.astype(float)
#     # batch, channel, h, w
#     images = np.transpose(images, (0, 2, 3, 1))

#     t = time.time()
#     if multi_processing:
#         num_threads = 8
#         with mp.Pool(num_threads) as p:
#             print(images.shape)
#             graph_datas = p.starmap(
#                 make_one_graph,
#                 [(img, 2, num_superpixel,) for img in images],
#             )
#         # check is ordered
#         # assert [q[0] for q in graph_datas] == list(range(len(graph_datas)))
#     else:
#         loader = tqdm(images, total=len(dataset))
#         graph_datas = [
#             make_one_graph(img, channel_axis=2, num_superpixel=num_superpixel)
#             for img in loader
#         ]
    
#     t = time.time() - t
#     print('fininsh', 't', t)

#     label = np.array(dataset.labels, dtype=np.uint8)

#     save_data([g[0] for g in graph_datas], f'{phase}_node')
#     save_data([g[1] for g in graph_datas], f'{phase}_edge')
#     save_data(label, f'{phase}_label')

def make_SVHN(is_train=True, multi_processing=True, num_superpixel=75):
    global OUTPUT_DIR
    DATASET_DIR = f'{DATA_DIR}/SVHN/'
    OUTPUT_DIR = 'SVHN_SUPERPIXEL/'
    if num_superpixel != 75:
        OUTPUT_DIR = OUTPUT_DIR.replace('SUPERPIXEL', f'{num_superpixel}_SUPERPIXEL')
    is_download = False

    phase = 'train' if is_train else 'test'
    print(f'Making [{phase}] datatset......')
    dataset = SVHN(DATASET_DIR, download=is_download, split=phase)
    images = dataset.data.astype(float)
    # batch, channel, h, w
    images = np.transpose(images, (0, 2, 3, 1))

    t = time.time()
    if multi_processing:
        num_threads = 8
        with mp.Pool(num_threads) as p:
            print(images.shape)
            graph_datas = p.starmap(
                make_one_graph,
                [(img, 2, num_superpixel,) for img in images],
            )
        # check is ordered
        # assert [q[0] for q in graph_datas] == list(range(len(graph_datas)))
    else:
        loader = tqdm(images, total=len(dataset))
        graph_datas = [
            make_one_graph(img, channel_axis=2, num_superpixel=num_superpixel)
            for img in loader
        ]
    
    t = time.time() - t
    print('fininsh', 't', t)

    label = np.array(dataset.labels, dtype=np.uint8)

    save_data([g[0] for g in graph_datas], f'{phase}_node')
    save_data([g[1] for g in graph_datas], f'{phase}_edge')
    save_data(label, f'{phase}_label')

def make_cifar10(is_train=True, multi_processing=True, num_superpixel=75):
    global OUTPUT_DIR
    OUTPUT_DIR = 'CIFAR10_SUPERPIXEL_200'
    if num_superpixel != 75:
        OUTPUT_DIR = OUTPUT_DIR.replace('SUPERPIXEL', f'{num_superpixel}_SUPERPIXEL')
    is_download = False

    phase = 'train' if is_train else 'test'
    print(f'Making [{phase}] datatset......')
    dataset = CIFAR10(DATA_DIR, download=is_download, train=is_train)
    images = dataset.data.astype(float)
    # batch, h, w, channel

    t = time.time()
    if multi_processing:
        num_threads = 8
        with mp.Pool(num_threads) as p:
            print(images.shape)
            graph_datas = p.starmap(
                make_one_graph,
                [(img, 2, num_superpixel,) for img in images],
            )
        # check is ordered
        # assert [q[0] for q in graph_datas] == list(range(len(graph_datas)))
    else:
        loader = tqdm(images, total=len(dataset))
        graph_datas = [
            make_one_graph(img, channel_axis=2, num_superpixel=num_superpixel)
            for img in loader
        ]
    
    t = time.time() - t
    print('fininsh', 't', t)

    label = np.array(dataset.targets, dtype=np.uint8)

    save_data([g[0] for g in graph_datas], f'{phase}_node')
    save_data([g[1] for g in graph_datas], f'{phase}_edge')
    save_data(label, f'{phase}_label')

def main():
#     make_SVHN(is_train=True)
#     make_SVHN(is_train=False)

#     make_SVHN(is_train=True, num_superpixel=200)
#     make_SVHN(is_train=False, num_superpixel=200)

    make_mnistm(is_train=True)
    make_mnistm(is_train=False)

    # make_cifar10(is_train=True)
    # make_cifar10(is_train=False)

#     make_cifar10(is_train=True, num_superpixel=150)
#     make_cifar10(is_train=False, num_superpixel=150)

if __name__ == '__main__':
    main()
