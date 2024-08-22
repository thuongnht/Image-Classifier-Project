# Imports here
import argparse
import copy
import os
import time

import numpy as np
# Check torch version and CUDA status if GPU is enabled.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
# Training data augmentation
# torchvision transforms are used to augment the training data
# with random scaling, rotations, mirroring, and/or cropping
import torchvision.transforms as transforms

print(f"Torch version {torch.__version__}")
print(f"Torch cuda {'available' if torch.cuda.is_available() else 'unavailable'}")


# STATIC Parameters
default_batch_size = 16
default_num_workers = 8
default_num_epochs = 12

default_learning_rate = 0.001
default_weight_decay = 0.005
default_momentum = 0.9

default_arch = 'vgg16'
default_save_dir = 'models'
default_gpu_mode = True


# STATIC Transforms

# Data normalization: Define your transforms for the training, validation, and testing sets
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize,
])

validate_data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

test_data_transforms = validate_data_transforms


class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise argparse.ArgumentError(f"ArgumentParser failed -> {message}")
        pass


def load_data_for_test(data_dir=None, batch_size=default_batch_size, num_workers=default_num_workers, **kwargs):
    if data_dir is None or data_dir == '':
        raise Exception("Directory of datas undefined!")

    test_dir = os.path.join(data_dir, 'test')
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_dataset, test_loader


def load_data_for_train(data_dir=None, batch_size=default_batch_size, num_workers=default_num_workers, **kwargs):
    if data_dir is None or data_dir == '':
        raise Exception("Directory of datas undefined!")

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    # Data loading: Load the datasets with ImageFolder
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_data_transforms)
    validate_dataset = torchvision.datasets.ImageFolder(root=valid_dir, transform=validate_data_transforms)
    # Data batching: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # Train Class names or target labels
    class_names = train_dataset.classes
    num_classes = len(train_dataset.classes)
    print(f"TRAIN - Nr. of Classes: {num_classes} - Classes: {class_names} ")
    # print(train_dataset.class_to_idx)
    print(f"VALIDATE - Nr. of Classes: {len(validate_dataset.classes)} - Classes: {validate_dataset.classes} ")
    return num_classes, train_dataset, train_loader, validate_dataset, validate_loader


def initial_model(arch=default_arch, train_on_gpu=False, num_classes=None,
                  learning_rate=default_learning_rate, weight_decay=default_weight_decay, momentum=default_momentum,
                  **kwargs):
    if num_classes is None:
        raise Exception("Number of labels undefined!")

    # Build and train network
    if arch == 'vgg16':
        model = models.vgg16(weights='VGG16_Weights.DEFAULT')
    elif arch == 'vgg19':
        model = models.vgg19(weights='VGG19_Weights.DEFAULT')
    elif arch == 'resnet152':
        model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
    else:
        raise Exception(f"Sorry, Model {arch} not supported!!!")

        # Freeze training for all layers
    for param in model.parameters():
        param.requires_grad = False

    # Newly created modules have 'require_grad=True' by default
    if 'VGG' in model.__class__.__name__:
        num_features = model.classifier[-1].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1))
    elif 'ResNet' in model.__class__.__name__:
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1))

    if train_on_gpu:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    print(f"Model {model.__class__.__name__} train on gpu {train_on_gpu}")
    return model, criterion, optimizer


def train_model(model=None, criterion=None, optimizer=None,
                train_loader=None, validate_loader=None,
                train_on_gpu=False, num_epochs=default_num_epochs,
                **kwargs):
    """
    Training the network: Build and train your network
    :param model:
    :param criterion:
    :param optimizer:
    :param train_loader:
    :param validate_loader:
    :param train_on_gpu:
    :param num_epochs:
    :return:
    """
    torch.multiprocessing.freeze_support()

    if model is None:
        raise Exception("Model undefined!")
    if criterion is None:
        raise Exception("Criterion undefined!")
    if optimizer is None:
        raise Exception("Optimizer undefined!")
    if train_loader is None:
        raise Exception("Train data undefined!")
    if validate_loader is None:
        raise Exception("Validate data undefined!")

    # keeping track of best weights
    best_model = copy.deepcopy(model.state_dict())
    # track change in validation loss
    validate_loss_min = np.inf

    for epoch in range(num_epochs):
        begin = time.time()

        # Train
        running_loss = 0.0
        model.train()  # Set model to training mode
        for i, data in enumerate(train_loader, 0):
            # get the inputs data is a list of [inputs, labels]
            inputs, labels = data
            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # forward + backward + optimize
                outputs = model(inputs)
                # calculate the batch loss
                loss = criterion(outputs, labels)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update average running loss
                running_loss += loss.item()*inputs.size(0)
        print (f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss}/{len(train_loader.dataset)}={running_loss/len(train_loader.dataset):.4f}')

        # Validate
        running_validate_loss = 0.0
        valid_acc = 0.0
        model.eval()   # Set model to evaluate mode
        for i, data in enumerate(validate_loader, 0):
            # get the inputs data is a list of [inputs, labels]
            inputs, labels = data
            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                # forward + backward + optimize
                outputs = model(inputs)
                # calculate the batch loss
                loss = criterion(outputs, labels)
                # convert output probabilities to predicted class
                _, preds = torch.max(outputs, 1)
                valid_acc += torch.sum(preds == labels.data)
                # update average running loss
                running_validate_loss += loss.item()*inputs.size(0)
        print (f'Epoch [{epoch + 1}/{num_epochs}], Validate Loss: {running_validate_loss}/{len(validate_loader.dataset)}={running_validate_loss/len(validate_loader.dataset):.4f}, Accuracy: {valid_acc}/{len(validate_loader.dataset)}={valid_acc/len(validate_loader.dataset):.4f}')

        # save model if validation loss has decreased
        if running_validate_loss <= validate_loss_min:
            print(f'Validation loss decreased ({validate_loss_min:.4f} --> {running_validate_loss:.4f}). Saving model ...')
            best_model = copy.deepcopy(model.state_dict())
            validate_loss_min = running_validate_loss

        duration = time.time() - begin
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {duration // 60:.0f}m {duration % 60:.0f}s")
        torch.cuda.empty_cache()
        print()

    model.load_state_dict(best_model)
    print('Finished Training')
    return model


def test_model(model=None, optimizer=None, test_loader=None,
               train_on_gpu=False,
               **kwargs):
    if model is None:
        raise Exception("Model undefined!")
    if optimizer is None:
        raise Exception("Optimizer undefined!")
    if test_loader is None:
        raise Exception("Test data undefined!")

    # Validation Loss and Accuracy: Do validation on the test set
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            inputs, labels = data
            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels.data)
    print(f'Accuracy of the network on the test images: {correct}/{total}={100 * correct // total}%')
    pass


def save_model(model=None, optimizer=None, path_model=None, class_to_idx=None,
               train_on_gpu=False,
               batch_size=default_batch_size, num_workers=default_num_workers, num_epochs=default_num_epochs,
               learning_rate=default_learning_rate, weight_decay=default_weight_decay, momentum=default_momentum,
               **kwargs):
    if model is None:
        raise Exception("Model undefined!")
    if optimizer is None:
        raise Exception("Optimizer undefined!")
    if path_model is None:
        raise Exception("Path to model undefined!")
    if class_to_idx is None:
        raise Exception("Label to index undefined!")

    model.class_to_idx = class_to_idx
    if 'VGG' in model.__class__.__name__:
        parameters = {
            'class_to_idx': model.class_to_idx,
            'classifier': model.classifier,
            'state_dict': model.state_dict(),
            'optimizer': optimizer,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_on_gpu': train_on_gpu,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'momentum': momentum
        }
    elif 'ResNet' in model.__class__.__name__:
        parameters = {
            'class_to_idx': model.class_to_idx,
            'classifier': model.fc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_on_gpu': train_on_gpu,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'momentum': momentum
        }
    else:
        raise Exception(f"Sorry, Model {model.__class__.__name__} not supported!!!")
    # print(model.class_to_idx)
    torch.save(parameters, path_model)
    print(f"Model saved to {path_model}")
    pass


def main():
    """
    Train a new network on a data set with train.py
    Prints out training loss, validation loss, and validation accuracy as the network trains

    Basic usage:
        python train.py data_directory
    Options:
    * Set directory to save checkpoints:
        python train.py data_dir --save_dir save_directory
    * Choose architecture:
        python train.py data_dir --arch "vgg13"
    * Set hyperparameters:
        python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    * Use GPU for training:
        python train.py data_dir --gpu

    :return:
    """
    parser = CustomArgumentParser(
        prog='train.py',
        usage='python %(prog)s data_dir [options]',
        description='Image Classifier Project'
    )
    parser.add_argument('data_dir')                                                             # positional argument
    parser.add_argument('-sd', '--save_dir', help='where the trained model is saved',
                        default=default_save_dir)
    parser.add_argument('--arch', help='Architecture, e.g. vgg16/ vgg19/ resnet152',
                        choices=['vgg16', 'vgg19', 'resnet152'], default=default_arch)
    parser.add_argument('--gpu', help='GPU mode', action='store_true',
                        default=default_gpu_mode)
    parser.add_argument('--batch_size', help='Batch size',
                        default=default_batch_size, type=int)
    parser.add_argument('--num_workers', help='Number of workers',
                        default=default_num_workers, type=int)
    parser.add_argument('--num_epochs', help='Number of epochs',
                        default=default_num_epochs, type=int)
    parser.add_argument('--learning_rate', help='learning rate',
                        default=default_learning_rate, type=float)
    parser.add_argument('--weight_decay', help='weight decay',
                        default=default_weight_decay, type=float)
    parser.add_argument('--momentum', help='momentum',
                        default=default_momentum, type=float)
    try:
        args = vars(parser.parse_args())
        print(f"Arguments {args}")
        train_on_gpu = torch.cuda.is_available() and args.get('gpu')
        print(f"Train on GPU {train_on_gpu}")

        # load train, validate data
        num_classes, train_dataset, train_loader, validate_dataset, validate_loader = load_data_for_train(**args)
        # initiate model
        model, criterion, optimizer = initial_model(
            train_on_gpu=train_on_gpu,
            num_classes=num_classes,
            **args
        )
        # do train & validate
        model = train_model(
            model=model, criterion=criterion, optimizer=optimizer,
            train_on_gpu=train_on_gpu,
            train_loader=train_loader,
            validate_loader=validate_loader,
            **args
        )
        # load test data
        test_dataset, test_loader = load_data_for_test(**args)
        # do test
        test_model(model=model, optimizer=optimizer, test_loader=test_loader, train_on_gpu=train_on_gpu, **args)
        # save model
        path_model = os.path.join(args.get('save_dir'), f"image_classifier_{args.get('arch')}.pth")
        save_model(model=model, optimizer=optimizer, path_model=path_model,
                   class_to_idx=train_dataset.class_to_idx,
                   train_on_gpu=train_on_gpu,
                   **args)
    except (argparse.ArgumentError, argparse.ArgumentTypeError) as ex:
        print(ex)
        parser.print_help()
    except Exception as ex:
        print(ex)
    pass


if __name__ == '__main__':
    main()


