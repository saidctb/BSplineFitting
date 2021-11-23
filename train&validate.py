import os
import time

import matplotlib.pyplot as plt
import numpy
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.utils.data as tud
import torchvision.transforms as tvt

import config
from dataset import SplineDataset
from evaluate import evaluate
from model import SplineModel, SimpleSplineModel


def show_image(img):
    """
    :param img: Input image in the form of normalized tensor
    :return: Open a window showing the input image
    """
    transform = tvt.Normalize(-1, 2)
    img = tvt.ToPILImage()(transform(img))
    img.show()


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def train(net, num_epoch, train_dataloader, val_dataloader, curve_type, model_type):
    """
    :param net: Model to be trained
    :param num_epoch: Number of epochs
    :param train_dataloader: Dataloader of the training dataset
    :param val_dataloader: Dataloader of the validation dataset
    :param curve_type: Type of curves
    :param model_type: Which kind of model is being trained
    """
    start = time.time()
    # Check the curve type
    if curve_type not in ['open', 'closed', 'all']:
        raise ValueError('Wrong curve type is provided')
    # Check the model type
    if model_type not in ['simple', 'complex']:
        raise ValueError('Wrong model type is provided')
    print('Start training the ' + model_type + ' model for ' + curve_type + ' curves')
    # Initialize the parameters
    net.apply(init_weights)
    # Check available devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Select cross entropy loss as the criterion
    criterion = nn.CrossEntropyLoss()
    # Select SGD optimizer for better local stability
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    # Track the losses in the training process
    loss_list = []
    # Track the accuracy on the validation dataset
    acc_list = []
    # Track the average loss on the validation dataset
    val_loss_list = []
    # Train in epochs
    for epoch in range(num_epoch):
        running_loss = 0.0
        net.train()
        for idx, data in enumerate(train_dataloader):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (idx + 1) % 100 == 0:
                print("\n[epoch:%d, batch:%5d] Train Loss: %.5f" % (epoch + 1, idx + 1, running_loss / float(100)))
                loss_list.append(running_loss / float(100))
                running_loss = 0.0
        # Evaluate the trained model
        acc, val_loss, _, _ = evaluate(net, epoch + 1, val_dataloader, curve_type, model_type)
        print("\n[epoch:%d] Validation Loss: %.5f" % (epoch + 1, val_loss))
        val_loss_list.append(loss_list)
        print("\n[epoch:%d] Validation Accuracy: %.5f" % (epoch + 1, acc))
        acc_list.append(acc)
    # Save the trained model
    model_root_path = '.\\Model\\' + model_type + '\\' + curve_type + '\\'
    if not os.path.exists(model_root_path):
        os.makedirs(model_root_path)
    torch.save(Net.state_dict(), model_root_path + 'SplineNet' + str(num_epoch) + '.pth')
    # Plot the training loss
    figure_root_path = '.\\Figure\\' + model_type + '\\' + curve_type + '\\'
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(loss_list)
    ax.set_xlabel('100 batches')
    ax.set_ylabel('Average Cross Entropy Loss')
    ax.set_title('Training Loss')
    plt.grid()
    plt.savefig(figure_root_path + 'train_loss.png')
    plt.close()
    # Plot the accuracy
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(acc_list)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average accuracy')
    ax.set_title('Validation Accuracy')
    plt.grid()
    plt.savefig(figure_root_path + 'val_accuracy.png')
    plt.close()
    # Plot the accuracy
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(val_loss_list)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Cross Entropy Loss')
    ax.set_title('Validation Loss')
    plt.grid()
    plt.savefig(figure_root_path + 'val_loss.png')
    plt.close()
    # Send an end message and print the training time
    end = time.time()
    print('The model for ' + curve_type +
          ' curves has been trained for %d epochs in %.2f seconds' % (num_epoch, end-start))


if __name__ == '__main__':
    # Ensure reproducibility
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmarks = False
    # Set training parameters
    Num_epoch = config.Epoch
    Curve_types = config.Curve_types
    Model_types = config.Model_types
    Num_N = config.N_max - config.N_min + 1
    # Check available GPU
    Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using ' + torch.cuda.get_device_name(Device) + ' to train and evaluate the models.')
    # Loop over all the assigned training process
    for Model_type in Model_types:
        # Create the model to be trained
        if Model_type == 'simple':
            Net = SimpleSplineModel(Num_N)
        elif Model_type == 'complex':
            Net = SplineModel(Num_N)
        else:
            raise ValueError('Wrong model type is provided')
        # Move this model to GPU if possible
        Net = Net.to(Device)
        # Print out the information of the model for a sanity check
        Num_of_parameters = sum(p.numel() for p in Net.parameters() if p.requires_grad)
        print("The number of learnable parameters in the model: %d" % Num_of_parameters)
        Num_of_layers = len(list(Net.parameters()))
        print("The number of layers in the model: %d" % Num_of_layers)
        for Curve_type in Curve_types:
            # Create the dataset and dataloader for training
            Train_dataset = SplineDataset('train', Curve_type)
            Train_dataloader = tud.DataLoader(dataset=Train_dataset, batch_size=10, shuffle=True, num_workers=0)
            # Create the dataset and dataloader for validation
            Val_dataset = SplineDataset('val', Curve_type)
            Val_dataloader = tud.DataLoader(dataset=Val_dataset, batch_size=10, shuffle=False, num_workers=0)
            # Initialize, train and evaluate the model on given datasets
            train(Net, Num_epoch, Train_dataloader, Val_dataloader, Curve_type, Model_type)
