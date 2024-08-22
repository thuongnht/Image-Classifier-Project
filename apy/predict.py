# Imports here
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Check torch version and CUDA status if GPU is enabled.
import torch
import torch.nn as nn
import torchvision.models as models
# Training data augmentation
# torchvision transforms are used to augment the training data
# with random scaling, rotations, mirroring, and/or cropping
import torchvision.transforms as transforms
from PIL import Image

print(f"Torch version {torch.__version__}")
print(f"Torch cuda {'available' if torch.cuda.is_available() else 'unavailable'}")


# STATIC Parameters
default_category_names = 'cat_to_name.json'
default_gpu_mode = True
default_topk = 5


# STATIC Transforms

# Data normalization: Define your transforms for the training, validation, and testing sets
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

test_data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise Exception(f"ArgumentParser failed -> {message}")
        pass


def load_categories(path_category=default_category_names):
    if path_category is None:
        raise Exception("Path to model undefined!")

    with open(path_category, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def load_model(path_model=None):
    if path_model is None:
        raise Exception("Path to model undefined!")

    checkpoint = torch.load(path_model)
    # print(checkpoint)
    if 'vgg19' in path_model.lower():
        model = models.vgg19(weights='VGG19_Weights.DEFAULT')
    elif 'vgg16' in path_model.lower():
        model = models.vgg16(weights='VGG16_Weights.DEFAULT')
    elif 'resnet152' in path_model.lower():
        model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
    else:
        raise Exception(f"Sorry, Model {path_model} not supported!!!")

        # freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    if 'VGG' in model.__class__.__name__:
        model.classifier = checkpoint['classifier']
    elif 'ResNet' in model.__class__.__name__:
        model.fc = checkpoint['classifier']

    model.load_state_dict(checkpoint['state_dict'])

    # model specifics
    model.class_to_idx = checkpoint['class_to_idx']
    # print(model.class_to_idx)

    if checkpoint['train_on_gpu']:
        model.cuda()

    # optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def process_image(image_path=None, transform=test_data_transforms):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    :param image_path:
    :param transform:
    :return:
    """
    if image_path is None:
        raise Exception("Path to image undefined!")
    loaded_image = Image.open(image_path)
    img_tensor = transform(loaded_image).float()
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def predict(image_path=None, model=None, topk=default_topk, train_on_gpu=False):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    :param image_path:
    :param model:
    :param topk:
    :param train_on_gpu:
    :return:
    """
    if model is None:
        raise Exception("Model undefined!")
    if image_path is None:
        raise Exception("Path to image undefined!")

    # Class Prediction: Implement the code to predict the class from an image file
    model.eval()
    # Implement the code to predict the class from an image file
    image_tensor = process_image(image_path)
    # Enable GPU functioning for prediction, when available
    if train_on_gpu:
        image_tensor = image_tensor.cuda()
        model.cuda()

    with torch.no_grad():
        # softmax layer to convert the output to probabilities
        # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        outputs = nn.Softmax(dim=1)(model(image_tensor))
        # https://pytorch.org/docs/stable/generated/torch.topk.html
        prediction = torch.topk(outputs, topk, dim=1)
        probabilities = prediction[0]
        labels = prediction[1]

    if train_on_gpu:
        probabilities = probabilities.cpu()
        labels = labels.cpu()

    idx_to_label = {idx: label for label, idx in model.class_to_idx.items()}
    #print(idx_to_label)

    return image_tensor, [p for p in probabilities.detach().numpy()[0]], [idx_to_label.get(idx) for idx in labels.detach().numpy()[0]]


def imshow(image, ax=None):
    """
    Imshow for Tensor.
    :param image:
    :param ax:
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def display_predict(image_path=None, model=None, map_categories=None, topk=default_topk, train_on_gpu=False):
    """
    Display image and preditions from model
    :param image_path:
    :param model:
    :param map_categories:
    :param topk:
    :param train_on_gpu:
    :return:
    """
    if model is None:
        raise Exception("Model undefined!")
    if image_path is None:
        raise Exception("Path to image undefined!")

    # Get predictions
    img, probs, preds = predict(image_path=image_path, model=model, topk=topk, train_on_gpu=train_on_gpu)

    # get appropriate labels
    idx_to_name = {category: map_categories[str(category)] for category in preds}
    classes = list(idx_to_name.values())
    print(f"Probabilities {probs} <-> Labels {classes}")
    print(f"The flower is most likely a '{classes[0]}' with about {100*probs[0]:.4f} probability!")

    if train_on_gpu:
        img = img.cpu().squeeze()

    # Show the image
    plt.figure(figsize=(12, 5))
    ax = plt.subplot(1, 2, 1)
    ax = imshow(img, ax=ax)

    # Set title to be the actual class
    ax.set_title(classes[0], size=20)

    # Plot a bar plot of predictions
    ax = plt.subplot(1, 2, 2)
    # Convert results to dataframe for plotting
    pd_probs = pd.DataFrame({'p': probs}, index=classes)
    pd_probs.sort_values('p')['p'].plot.barh(color='blue', edgecolor='k', ax=ax)
    plt.xlabel('Predicted Probability')
    plt.tight_layout()
    plt.show()
    pass


def predict(path_image=None, path_model=None, topk=default_topk,
            gpu=default_gpu_mode,
            path_category=default_category_names,
            **kwargs):
    """
    Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
    """
    try:
        print(f"Arguments {path_image}, {path_model}, {topk}")
        train_on_gpu = torch.cuda.is_available() and gpu
        print(f"Train on GPU {train_on_gpu}")

        # load categories
        map_categories = load_categories(path_category=path_category)
        # load model
        model, optimizer = load_model(path_model=path_model)
        # do prediction
        display_predict(image_path=path_image, model=model,
                        map_categories=map_categories,
                        topk=topk,
                        train_on_gpu=train_on_gpu)
    except Exception as ex:
        print(ex)

    pass




