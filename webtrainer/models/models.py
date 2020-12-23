import torch
from torchvision import models
from models.resnet import Resnet18, Resnet34
from models.mobilenet import MobileNetV2
from models.squeezenet import SqueezeNet
from models.inceptionv3 import Inception3
import enum

supported_models = ["Resnet18", "Resnet34", "Mobilenet", "Squeezenet", "Inception"]

def load_models(model_info, dataset):
    # Name of the model
    model_name = model_info['name']
    # Yet to be implemented
    model_pretrained = model_info['pretrained']
    # Task
    model_task = model_info['task']
    if 'classification' in model_task:
        model_task = ModelTasks.classification
    elif 'segmentation' in model_task:
        model_task = ModelTasks.segmentation
    else:
        print("Unknown task defaulting to classification")
        model_task = ModelTasks.classification
    # Channels
    in_channels = dataset.channels
    if model_name in supported_models:
        if model_name == "Resnet18":
            model = Resnet18(task=model_task, num_classes=model_info['num_classes'], in_channels=in_channels, out_channels=dataset.num_output)
            # TODO add in functionality that loads all pretrained parameters except last layer
        elif model_name == "Resnet34":
            model = Resnet34(task=model_task, num_classes=model_info['num_classes'], in_channels=in_channels, out_channels=dataset.num_output)
        elif model_name == "Mobilenet":
            model = MobileNetV2(num_classes=model_info['num_classes'], in_channels=in_channels)
        elif model_name == "Squeezenet":
            model = SqueezeNet(num_classes=model_info['num_classes'], in_channels=in_channels)
        elif model_name == "Inception":
            model = Inception3(num_classes=model_info['num_classes'], in_channels=in_channels)
        else:
            print("Unknown model... defaulting to resnet18")
            model = models.resnet18(pretrained=model_pretrained)
    else:
        print("Unknown model... defaulting to resnet18")
        model = Resnet18(num_classes=model_info['num_classes'], in_channels=in_channels)
    return model

class ModelTasks(enum.Enum):
    classification = 1
    segmentation = 2
