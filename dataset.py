import json
from abc import ABC

import torch.utils.data as tud
import torchvision.transforms as tvt
from PIL import Image

import config


class SplineDataset(tud.Dataset, ABC):
    """
    Custom dataset containing the spline images created
    """

    def __init__(self, mode: str, cl: str):
        """
        :param mode: Whether load the training dataset or the validation dataset
        :param cl: Whether to include closed curves or open curves only, or both curves
        """
        super(SplineDataset, self).__init__()
        # Initialize parameters
        self.N_list = range(config.N_min, config.N_max + 1)
        self.imgs = list()
        self.imgs_N = list()
        self.transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize(0.5, 0.5)])
        # Check the usage mode
        if mode == 'train':
            json_path = '.\\Json\\img_train.json'
            root_path = '.\\Splines\\Train\\'
        elif mode == 'val':
            json_path = '.\\Json\\img_val.json'
            root_path = '.\\Splines\\Val\\'
        else:
            raise ValueError('Wrong dataset type is provided')
        # Load the needed json file
        json_file = open(json_path)
        json_data = json.load(json_file)
        json_file.close()
        # Enumerate the collect all the available data
        if cl == 'all':
            for idx, key in enumerate(json_data):
                self.imgs.append(json_data[key]['path'])
                self.imgs_N.append(json_data[key]['n'])
        elif cl == 'closed':
            for idx, key in enumerate(json_data):
                if json_data[key]['cl']:
                    self.imgs.append(json_data[key]['path'])
                    self.imgs_N.append(json_data[key]['n'])
        elif cl == 'open':
            for idx, key in enumerate(json_data):
                if not json_data[key]['cl']:
                    self.imgs.append(json_data[key]['path'])
                    self.imgs_N.append(json_data[key]['n'])
        else:
            raise ValueError('Wrong curve type is provided')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        :param idx: Index of the image in the dataset
        :return: Loaded image and corresponding number of control points
        """
        # Open and load the image file
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        # Since it's not related to the colors, the image is converted to a grayscale image
        img = img.convert('L')
        # Resize to 256*256 for standardization
        img = img.resize((256, 256), Image.BOX)
        # Transform it into tensor and normalize
        img = self.transform(img)
        # Corresponding number of control points as label
        # Set the minimal label to be zero for one-hot encoding
        n = self.imgs_N[idx] - config.N_min
        return img, n
