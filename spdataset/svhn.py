import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T

DATA_DIR = '~/data/SVHN_SUPERPIXEL'

class SVHNSuperPixelDataset(Dataset):
    num_features = 5
    num_classes = 10

    def __init__(self, data_dir=DATA_DIR, transform=None, train=True, num_superpixel=50):
        phase = 'train' if train else 'test'

        print(f'----- Loading {phase} dataset -----')
        data_dir = os.path.expanduser(data_dir)
        if num_superpixel != 75:
            data_dir += f'_{num_superpixel}'
        print(f'---------- dataset path: {data_dir} ----------')
        def get_path(f):
            return os.path.abspath(os.path.join(data_dir, f))
        
        nodes = torch.load(get_path(f'{phase}_node.pt'))
        edges = torch.load(get_path(f'{phase}_edge.pt'))
        labels = torch.load(get_path(f'{phase}_label.pt'))

        self.nodes = [torch.from_numpy(n).float() for n in nodes]
        self.pos = [n[:, :2] for n in self.nodes]
        self.edges = [torch.from_numpy(e) for e in edges]
        self.labels = torch.LongTensor(labels)

        self.transform = transform

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        n = self.nodes[idx]
        p = self.pos[idx]
        e = self.edges[idx]
        l = self.labels[idx]

        data = Data(
            x=n,
            pos=p,
            edge_index=e,
            y=l
        )
        if self.transform:
            data = self.transform(data)
        
        return data

# from make_dataset import draw_superpixel_from_graph
import matplotlib.pyplot as plt
def draw_superpixel_from_graph(node, edge):
    plt.figure(figsize=(5, 5))
    
    for i in range(edge.shape[1]):
        e = edge[:, i].tolist()
        d1 = node[e[0]][:2]
        d2 = node[e[1]][:2]
        dd1 = d1[1], d2[1]
        dd2 = d1[0], d2[0]
        plt.plot(dd1, dd2, color='#44AA44', linestyle='dashed')
        
    # for n in node:
    for i, n in enumerate(node):
        n = n.tolist()
        x, y = n[1], n[0]
        feature = n[2:]
        fs = [int(f * 255) for f in feature]
        if len(fs) == 1: # single channel
            fs *= 3
        fs = ['{:02x}'.format(f) for f in fs]
        c = '#{:s}'.format(''.join(fs))
        plt.plot(x, y, marker='o', ms=15, color=c)
    
    # plt.title('y = {}'.format(img.y.item()))
    plt.axis('off')
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.show()
    plt.savefig('temp.svg')

if __name__ == '__main__':
    trans = T.Cartesian(cat=False)
    dataset = SVHNSuperPixelDataset(train=False, transform=trans)
    data = dataset[0]

    print('x', data.x.shape)
    print('edge', data.edge_index.shape)

    draw_superpixel_from_graph(data.x, data.edge_index)
