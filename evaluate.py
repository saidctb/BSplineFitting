import os

import matplotlib.pyplot as plt
import numpy
import torch.backends.cudnn
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import config


def evaluate(net, epoch_no, dataloader, curve_type, model_type):
    """
    :param net: Model to be evaluated
    :param epoch_no: Declare which epoch this evaluation result comes from
    :param dataloader: Dataloader of the validation dataset
    :param curve_type：Type of curves
    :param model_type: Which kind of model is being evaluated
    :return: Overall accuracy, average loss, predictions and labels
    """
    # Check the curve type
    if curve_type not in ['closed', 'open', 'all']:
        raise ValueError('Wrong curve type is provided')
    # Check the model type
    if model_type not in ['simple', 'complex']:
        raise ValueError('Wrong model type is provided')
    # Check the performance of trained model on validation dataset
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        predictions = torch.Tensor([]).to(device)
        labels = torch.Tensor([]).to(device)
        # The criterion is CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        # Get the predictions
        num_batches = 0
        running_loss = 0.0
        for idx, data in enumerate(dataloader):
            imgs, imgs_n = data
            imgs = imgs.to(device)
            imgs_n = imgs_n.to(device)
            output = net(imgs)
            loss = criterion(output, imgs_n)
            _, output = torch.max(output, 1)
            predictions = torch.cat((predictions, output))
            labels = torch.cat((labels, imgs_n))
            running_loss += loss.item()
            num_batches += 1
    # Get the confusion matrix
    predictions = predictions.cpu()
    labels = labels.cpu()
    cm = confusion_matrix(labels, predictions)
    # Compute the accuracies and average loss
    correct_num = numpy.sum(numpy.diag(cm))
    total_num = numpy.sum(cm)
    overall_accuracy = correct_num / total_num
    avg_loss = running_loss / float(num_batches)
    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(config.N_min, config.N_max + 1))
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    disp.plot(xticks_rotation=45, colorbar=False, cmap='Blues', ax=ax)
    str_xlabel = ax.get_xlabel() + '\n Overall Accuracy: ' + str(overall_accuracy)
    str_xlabel += '\n Average loss: ' + str(avg_loss)
    ax.set_xlabel(str_xlabel)
    # Save figure of confusion matrix
    figure_root_path = '.\\Figure\\' + model_type + '\\' + curve_type + '\\'
    if not os.path.exists(figure_root_path):
        os.makedirs(figure_root_path)
    plt.savefig(figure_root_path + 'confusion_matrix_' + str(epoch_no), bbox_inches=0)
    plt.close()
    # Return the predictions and labels
    return overall_accuracy, avg_loss, labels, predictions
