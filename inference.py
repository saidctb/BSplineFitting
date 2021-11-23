import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as tvt
from PIL import Image

import config
from model import SplineModel


def spline_to_tensor(spl: np.ndarray):
    """
    :param spl: Curve to be transformed
    :return: Normalized tensor
    """
    assert spl.shape[1] == 2, 'Please input 2D curves in the form of [num, dim]'
    # Obtain image configs
    w = config.Width
    h = config.Height
    dpi = config.Dpi
    lw = config.Linewidth
    # Plot the spline curve and save as a colorful png
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax.plot(spl[:, 0], spl[:, 1], lw=lw)
    plt.axis('off')
    cache_path = '.\\Cache\\'
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    path = cache_path + 'cache.png'
    plt.savefig(path, bbox_inches=0)
    plt.close()
    # Import the image as a grayscale image
    img = Image.open('.\\Cache\\cache.png')
    img = img.convert('L')
    # Resize to 256*256 for standardization
    img = img.resize((256, 256), Image.BOX)
    # Transform into tensor and normalize
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize(0.5, 0.5)])
    img = transform(img)
    # Return the normalized tensor (1,1,256,256)
    img = img[None, :]
    return img


class SplineInference():
    """
    A class to load trained model to predict number of control points
    """

    def __init__(self, epoch: int, curve_type: str):
        """
        :param epoch: Determine which model (after how many epochs) to be used
        :param curve_type: Determine which model (for what kind of curves) to be used
        """
        print('Start loading the trained network for ' + curve_type + ' after ' + str(epoch) + ' epoch(s).')
        num_n = config.N_max - config.N_min + 1
        self.net = SplineModel(num_n)
        self.net.load_state_dict(torch.load('.\\Model\\' + curve_type + '\\SplineNet' + str(epoch) + '.pth'))
        print('Successfully loaded the trained network.')
        self.net.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)
        print('Using ' + torch.cuda.get_device_name(self.device) + ' for the trained model.')

    def __getitem__(self, spl: np.ndarray):
        """
        :param spl: Curve to be predicted
        :return: Prediction of number of control points
        """
        # First get the normalized tensor
        img = spline_to_tensor(spl)
        # Move to assigned device if necessary
        img = img.to(self.device)
        # Get the model output
        output = self.net(img)
        _, prediction = torch.max(output, 1)
        # Return the prediction
        prediction = torch.squeeze(prediction)
        prediction = prediction.cpu()
        return prediction.item()
