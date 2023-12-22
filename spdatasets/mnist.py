import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T

DATA_DIR = '~/data/MNIST_SUPERPIXEL'

class MNISTSuperPixelDataset(Dataset):
    num_features = 3
    num_classes = 10

    def __init__(self, data_dir=DATA_DIR, transform=None, train=True):
        phase = 'train' if train else 'test'

        print(f'----- Loading {phase} dataset -----')
        data_dir = os.path.expanduser(data_dir)
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

if __name__ == '__main__':
    trans = T.Cartesian(cat=False)
    dataset = MNISTSuperPixelDataset(train=False, transform=trans)
    data = dataset[0]

    # draw_superpixel_from_graph(data.x, data.edge_index)