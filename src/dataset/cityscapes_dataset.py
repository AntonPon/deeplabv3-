from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np


class CityscapesDataSet(Dataset):

    def get_imgs_list(self, path_to_imgs, mask, img_type='.png'):
        return list(path_to_imgs.glob('**/*{}{}'.format(mask, img_type)))

    def __init__(self, path_to_data, split='train', num_classes=19, transforms=None, ignore_index=255):
        super(CityscapesDataSet, self).__init__()
        self.num_classes = num_classes
        self.transforms = transforms
        self.split = split
        self.path_to_masks = path_to_data / 'gtFine' / self.split
        path_to_imgs = path_to_data / 'leftImg8bit' / self.split

        if not path_to_data.exists():
            raise ValueError('the path doesn\'t exist')
        elif not path_to_data.is_dir():
            raise ValueError('the path is not dir')

        self.imgs_list = self.get_imgs_list(path_to_imgs, '_leftImg8bit')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        self.ignore_index = ignore_index
        self.class_map = dict(zip(self.valid_classes, range(self.num_classes)))

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, item):
        img_path = self.imgs_list[item]
        parts = img_path.parts[-2:]
        mask_path =self.path_to_masks / parts[0] / parts[1].replace('leftImg8bit', 'gtFine_labelIds')

        return self.prepare_data(str(img_path), str(mask_path), self.transforms)

    def prepare_data(self, img_path, mask_path, transforms):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError('cannot open the image: {}'.format(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise ValueError('cannot open the mask: {}'.format(mask_path))
        mask = self.mask_encode(mask)

        if transforms is not None:
            augmented = transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        img = np.transpose(img, (-1, 0, 1))
        return {'img': img, 'mask': mask}

    def mask_encode(self, mask):
        indices = np.zeros(mask.shape, dtype=bool)
        for void_index in self.void_classes:
            indices += (mask == void_index)
        mask[indices] = self.ignore_index
        for index in self.valid_classes:
            mask[mask == index] = self.class_map[index]
        return mask

from tqdm import tqdm
if __name__ == '__main__':
    a = CityscapesDataSet(Path('/home/user/Documents/datasets/cityscapes'))

    total_sum = 0
    for i in tqdm(range(4)):
        img = a[i]['mask']

    print(total_sum == len(a))

