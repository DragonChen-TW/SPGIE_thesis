import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

class MNISTMData(Dataset):
    def __init__(self, root, mode='train', to_gray=False):
        '''
        Get Image Path.
        '''
        self.mode = mode # 'train' or 'test'

        root = os.path.expanduser(root)
        if mode == 'train':
            folder = os.path.join(root, 'mnist_m_train')
        else:
            folder = os.path.join(root, 'mnist_m_test')

        # generate imgs path
        imgs = [os.path.join(folder, img).replace('\\', '/') \
                for img in os.listdir(folder)]
        imgs = sorted(imgs, key=lambda img: int(img.split('/')[-1].split('.')[0]))
        imgs_num = len(imgs)
        self.imgs = imgs

        # generate labels
        label_file = 'mnist_m_{}_labels.txt'.format(mode)
        label_file = os.path.join(root, label_file)
        with open(label_file) as f:
            self.labels = [l[:-1].split(' ')[1] for l in f]

        # shape is 32x32
        # transforms
        trans = []
        if mode == 'train':
            trans += [
                transforms.RandomCrop(28),
            ]
        else:
            trans += [
                transforms.CenterCrop(28),
            ]
        trans += [transforms.ToTensor()]
        
        self.transforms = transforms.Compose(trans)

    def __getitem__(self, index):
        '''
        return one image's data
        if in test dataset, return image's id
        '''
        img_path = self.imgs[index]
        label = self.labels[index]
        label = int(label)

        data = Image.open(img_path)
        data = self.transforms(data)

        return data, label
    
    def get_all_imgs(self):
        length = len(self.imgs)
        loader = tqdm(range(length), total=length)
        imgs = [self[i][0].numpy() for i in loader]
        imgs = np.stack(imgs, 0).astype(float)
        return imgs

    def __len__(self):
        return len(self.imgs)