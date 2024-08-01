import os
import torch
import torch.utils.data as data
import numpy as np
import cv2

from torchvision.datasets import ImageNet

from PIL import Image, ImageFilter
import h5py
from glob import glob
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

# from folder2lmdb import ImageFolderLMDB



class ImageNet_blur(ImageNet):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        gauss_blur = ImageFilter.GaussianBlur(11)
        median_blur = ImageFilter.MedianFilter(11)

        blurred_img1 = sample.filter(gauss_blur)
        blurred_img2 = sample.filter(median_blur)
        blurred_img = Image.blend(blurred_img1, blurred_img2, 0.5)

        if self.transform is not None:
            sample = self.transform(sample)
            blurred_img = self.transform(blurred_img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, blurred_img), target


class Imagenet_Segmentation(data.Dataset):
    CLASSES = 2

    def __init__(self,
                 path,
                 transform=None,
                 target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        # self.h5py = h5py.File(path, 'r+')
        self.h5py = None
        tmp = h5py.File(path, 'r')
        self.data_length = len(tmp['/value/img'])
        tmp.close()
        del tmp

    def __getitem__(self, index):

        if self.h5py is None:
            self.h5py = h5py.File(self.path, 'r')

        img = np.array(self.h5py[self.h5py['/value/img'][index, 0]]).transpose((2, 1, 0))
        target = np.array(self.h5py[self.h5py[self.h5py['/value/gt'][index, 0]][0, 0]]).transpose((1, 0))

        img = Image.fromarray(img).convert('RGB')
        target = Image.fromarray(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')
            target = torch.from_numpy(target).long()

        return img, target

    def __len__(self):
        # return len(self.h5py['/value/img'])
        return self.data_length


class Imagenet_Segmentation_Blur(data.Dataset):
    CLASSES = 2

    def __init__(self,
                 path,
                 transform=None,
                 target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        # self.h5py = h5py.File(path, 'r+')
        self.h5py = None
        tmp = h5py.File(path, 'r')
        self.data_length = len(tmp['/value/img'])
        tmp.close()
        del tmp

    def __getitem__(self, index):

        if self.h5py is None:
            self.h5py = h5py.File(self.path, 'r')

        img = np.array(self.h5py[self.h5py['/value/img'][index, 0]]).transpose((2, 1, 0))
        target = np.array(self.h5py[self.h5py[self.h5py['/value/gt'][index, 0]][0, 0]]).transpose((1, 0))

        img = Image.fromarray(img).convert('RGB')
        target = Image.fromarray(target)

        gauss_blur = ImageFilter.GaussianBlur(11)
        median_blur = ImageFilter.MedianFilter(11)

        blurred_img1 = img.filter(gauss_blur)
        blurred_img2 = img.filter(median_blur)
        blurred_img = Image.blend(blurred_img1, blurred_img2, 0.5)

        # blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
        # blurred_img2 = np.float32(cv2.medianBlur(img, 11))
        # blurred_img = (blurred_img1 + blurred_img2) / 2

        if self.transform is not None:
            img = self.transform(img)
            blurred_img = self.transform(blurred_img)

        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')
            target = torch.from_numpy(target).long()

        return (img, blurred_img), target

    def __len__(self):
        # return len(self.h5py['/value/img'])
        return self.data_length


class Imagenet_Segmentation_eval_dir(data.Dataset):
    CLASSES = 2

    def __init__(self,
                 path,
                 eval_path,
                 transform=None,
                 target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.h5py = h5py.File(path, 'r+')

        # 500 each file
        self.results = glob(os.path.join(eval_path, '*.npy'))

    def __getitem__(self, index):

        img = np.array(self.h5py[self.h5py['/value/img'][index, 0]]).transpose((2, 1, 0))
        target = np.array(self.h5py[self.h5py[self.h5py['/value/gt'][index, 0]][0, 0]]).transpose((1, 0))
        res = np.load(self.results[index])

        img = Image.fromarray(img).convert('RGB')
        target = Image.fromarray(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')
            target = torch.from_numpy(target).long()

        return img, target

    def __len__(self):
        return len(self.h5py['/value/img'])
    
# class INatDataset(ImageFolder):
#     def __init__(self, root, train=True, year=2012, transform=None, target_transform=None,
#                  category='name', loader=default_loader):
#         self.transform = transform
#         self.loader = loader
#         self.target_transform = target_transform
#         self.year = year
#         # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
#         path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
#         with open(path_json) as json_file:
#             data = json.load(json_file)

#         with open(os.path.join(root, 'categories.json')) as json_file:
#             data_catg = json.load(json_file)

#         path_json_for_targeter = os.path.join(root, f"train{year}.json")

#         with open(path_json_for_targeter) as json_file:
#             data_for_targeter = json.load(json_file)

#         targeter = {}
#         indexer = 0
#         for elem in data_for_targeter['annotations']:
#             king = []
#             king.append(data_catg[int(elem['category_id'])][category])
#             if king[0] not in targeter.keys():
#                 targeter[king[0]] = indexer
#                 indexer += 1
#         self.nb_classes = len(targeter)

#         self.samples = []
#         for elem in data['images']:
#             cut = elem['file_name'].split('/')
#             target_current = int(cut[2])
#             path_current = os.path.join(root, cut[0], cut[2], cut[3])

#             categors = data_catg[target_current]
#             target_current_true = targeter[categors[category]]
#             self.samples.append((path_current, target_current_true))

#     # __getitem__ and __len__ inherited from ImageFolder


# def build_dataset(is_train, args):
#     transform = build_transform(is_train, args)

#     if args.data_set == 'CIFAR':
#         dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
#         nb_classes = 100
#     elif args.data_set == 'IMNET':
#         if args.use_lmdb:
#             root = os.path.join(args.data_path, 'train.lmdb' if is_train else 'val.lmdb')
#             if not os.path.isfile(root):
#                 raise FileNotFoundError(f"LMDB dataset '{root}' is not found. "
#                         "Pleaes first build it by running 'folder2lmdb.py'.")
#             dataset = ImageFolderLMDB(root, transform=transform)
#         else:
#             root = os.path.join(args.data_path, 'train' if is_train else 'val')
#             dataset = datasets.ImageFolder(root, transform=transform)
#         nb_classes = 1000
#     elif args.data_set == 'INAT':
#         dataset = INatDataset(args.data_path, train=is_train, year=2018,
#                               category=args.inat_category, transform=transform)
#         nb_classes = dataset.nb_classes
#     elif args.data_set == 'INAT19':
#         dataset = INatDataset(args.data_path, train=is_train, year=2019,
#                               category=args.inat_category, transform=transform)
#         nb_classes = dataset.nb_classes

#     return dataset, nb_classes


# def build_transform(is_train, args):
#     resize_im = args.input_size > 32
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation=args.train_interpolation,
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#         )
#         if not resize_im:
#             # replace RandomResizedCropAndInterpolation with
#             # RandomCrop
#             transform.transforms[0] = transforms.RandomCrop(
#                 args.input_size, padding=4)
#         return transform

#     t = []
#     if resize_im:
#         size = int((256 / 224) * args.input_size)
#         t.append(
#             transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
#         )
#         t.append(transforms.CenterCrop(args.input_size))

#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
#     return transforms.Compose(t)



if __name__ == '__main__':
    import torchvision.transforms as transforms
    from tqdm import tqdm
    from imageio import imsave
    import scipy.io as sio

    # meta = sio.loadmat('/home/shirgur/ext/Data/Datasets/temp/ILSVRC2012_devkit_t12/data/meta.mat', squeeze_me=True)['synsets']

    # Data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_img_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    test_lbl_trans = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
    ])

    ds = Imagenet_Segmentation('/home/shirgur/ext/Data/Datasets/imagenet-seg/other/gtsegs_ijcv.mat',
                               transform=test_img_trans, target_transform=test_lbl_trans)

    for i, (img, tgt) in enumerate(tqdm(ds)):
        tgt = (tgt.numpy() * 255).astype(np.uint8)
        imsave('/home/shirgur/ext/Code/C2S/run/imagenet/gt/{}.png'.format(i), tgt)

    print('here')
