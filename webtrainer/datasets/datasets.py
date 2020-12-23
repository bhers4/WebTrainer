import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2 as opencv
import json
import enum
from torchvision import datasets, transforms
import torchvision
from matplotlib import pyplot as plt

class DatasetTypes(enum.Enum):
    supported = 1
    custom = 2

class TrainerDataset(object):

    def __init__(self, dataset_info, num_workers, transforms=None):
        self.dataset_info = dataset_info
        self.num_workers = num_workers
        # Supported Datasets
        self.supported_datasets = ['CIFAR10', 'MNIST']
        self.dataset_name = dataset_info['name']
        self.batch_size = dataset_info['batch_size']
        # Do check if dataset_name is in supported_datasets
        self.transforms = transforms
        # Channels
        self.channels = 3
        self.num_output = 1
        # Custom Dataset Parameters
        self.loaded_imgs = []
        self.img_paths = []
        self.json_paths = []
        self.imgs = {}
        self.min_size_x = 10000
        self.min_size_y = 10000
        #
        if self.dataset_name in self.supported_datasets:
            self.dataset_type = DatasetTypes.supported
        else:
            self.dataset_type = DatasetTypes.custom
        #
        if self.dataset_type == DatasetTypes.supported:
            self.load_supported_dataset()
        else:
            self.load_custom_dataset()

        return

    def load_supported_dataset(self):
        if self.transforms is None:
            self.transforms = torchvision.transforms.ToTensor()

        if self.dataset_name == "CIFAR10":
            self.train_dataset = datasets.CIFAR10('data/', train=True, download=True, transform=self.transforms)
            self.test_dataset = datasets.CIFAR10('data/', train=False, download=True, transform=self.transforms)
        elif self.dataset_name == "MNIST":
            self.channels = 1
            self.train_dataset = datasets.MNIST('data/', train=True, download=True, transform=self.transforms)
            self.test_dataset = datasets.MNIST('data/', train=False, download=True, transform=self.transforms)
        # Dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers)
        return


    # Loads custom dataset
    def load_custom_dataset(self):
        dataset_paths = self.dataset_info['paths']
        loaded_imgs = []
        loaded_jsons = []
        loaded_pairs = []
        for path in dataset_paths:
            if os.path.isdir(path):
                imgs = self.get_imgs(path)
                jsons = self.get_jsons(path)
                for item in imgs:
                    loaded_imgs.append(item)
                for item in jsons:
                    loaded_jsons.append(item)
        #
        pairs = [path for path in self.img_paths if path in self.json_paths]
        self.loaded_imgs = loaded_imgs
        # Create dict to pass to json mappings
        json_dict = {}
        for item in pairs:
            json_dict[item] = item + ".json"
        json_mapping, num_output = create_json_mapping(json_dict)
        self.num_output = num_output  # Number of Output classes
        self.json_mapping = json_mapping
        # Load images
        imgs = []
        masks = []
        for item in pairs:
            img = self.imgs[item]
            img = np.array(Image.open(img).convert(mode='RGB'))
            json_file = item+".json"
            if img.shape[1] < self.min_size_x:
                self.min_size_x = img.shape[1]
            if img.shape[0] < self.min_size_y:
                self.min_size_y = img.shape[0]
            imgs.append(img)
            with open(json_file, "r") as json_mask:
                json_mask = json.load(json_mask)
                img_height = json_mask['imageHeight']
                img_width = json_mask['imageWidth']
                img_shapes = json_mask['shapes']
                img_mask = np.zeros((img_height, img_width))
                for shape in img_shapes:
                    shape_label = shape['label']
                    polys = shape['points']
                    label_val = self.json_mapping[shape_label]
                    opencv.fillPoly(img_mask, pts=np.int32([polys]), color=label_val)
                masks.append(img_mask)
        # Do train test split
        train_imgs = []
        train_masks = []
        test_imgs = []
        test_masks = []
        split = self.dataset_info['split']
        num_train = int(split*len(imgs))
        num_test = len(imgs) - num_train
        rand_sample = random.sample(range(len(imgs)), num_train)
        for i in range(len(imgs)):
            if i in rand_sample:
                train_imgs.append(imgs[i])
                train_masks.append(masks[i])
            else:
                test_imgs.append(imgs[i])
                test_masks.append(masks[i])
        print("Train imgs: {}, Train masks: {}, Test imgs: {}, Test masks: {}".format(len(train_imgs), len(train_masks), len(test_imgs), len(test_masks)))
        self.train_dataset = CustomImageDataset(train_imgs, train_masks, json_mapping)
        self.test_dataset = CustomImageDataset(test_imgs, test_masks, json_mapping)
        # Dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers)
        return

    # Loads all the pngs and jpegs
    def get_imgs(self, path):
        loaded_images = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_type = file.split(".")[-1]
                file_key = file.split(".")[0]
                if file.endswith(".png"):
                    loaded_images.append(os.path.join(root, file))
                    self.img_paths.append(os.path.join(root, file_key))
                    self.imgs[os.path.join(root, file_key)] = os.path.join(root, file)
                elif file.endswith(".jpeg"):
                    loaded_images.append(os.path.join(root, file))
                    self.img_paths.append(os.path.join(root, file_key))
                    self.imgs[os.path.join(root, file_key)] = os.path.join(root, file)
                elif file.endswith(".jpg"):
                    loaded_images.append(os.path.join(root, file))
                    self.img_paths.append(os.path.join(root, file_key))
                    self.imgs[os.path.join(root, file_key)] = os.path.join(root, file)
        return loaded_images

    # Load jsons
    def get_jsons(self, path):
        loaded_jsons = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_type = file.split(".")[-1]
                file_key = file.split(".")[0]
                if file.endswith(".json"):
                    loaded_jsons.append(os.path.join(root, file))
                    self.json_paths.append(os.path.join(root, file_key))
        return loaded_jsons

    def get_batch_size(self):
        return self.batch_size

    def get_train_iter(self):
        # Look into setting epoch number
        return iter(self.train_loader)

    def get_test_iter(self):
        return iter(self.test_loader)


class ImageDataset(Dataset):
    def __init__(self, imgs, jsons, load, json_mapping):
        '''
            imgs: list of images
            jsons: list of jsons
        '''
        self.imgs = imgs
        self.jsons = jsons
        self.load = load
        self.xs = []
        self.ys = []
        self.loaded_imgs = {}
        self.loaded_jsons = {}
        # Dict -> Int mapping
        self.dict_int_mapping = {}
        self.json_mapping = json_mapping
        if load:
            for key in imgs.keys():
                # print("Img: ", imgs[key])
                try:
                    img = Image.open(imgs[key])
                    self.loaded_imgs[key] = img
                    self.ys.append(img.size[0])
                    self.xs.append(img.size[1])
                except:
                    print("Error loading Image at ", imgs[key])
            print("Y, min: ", np.min(self.ys), " Max: ", np.max(self.ys), " Mean: ", np.mean(self.ys))
            print("X, min: ", np.min(self.xs), " Max: ", np.max(self.xs), " Mean: ", np.mean(self.xs))
            for key in self.jsons.keys():
                json_mask = self.jsons[key]
                print("Json mask: ", json_mask)
                with open(json_mask, "r") as json_mask:
                    json_mask = json.load(json_mask)
                    img_height = json_mask['imageHeight']
                    img_width = json_mask['imageWidth']
                    img_shapes = json_mask['shapes']
                    img_mask = np.zeros((img_height, img_width))
                    for shape in img_shapes:
                        shape_label = shape['label']
                        polys = shape['points']
                        label_val = self.json_mapping[shape_label]
                        opencv.fillPoly(img_mask, pts=np.int32([polys]), color=label_val)
                    self.loaded_jsons[key] = img_mask
            self.create_dict_to_int_mapping(imgs)
        return

    def create_dict_to_int_mapping(self, data_dict):
        counter = 0
        for key in data_dict:
            self.dict_int_mapping[counter] = key
            self.dict_int_mapping[key] = counter
            counter += 1

    def __len__(self):
        return len(self.imgs.keys())

    def __getitem__(self, i):
        """
            IDEA: one way to load images just for batch is have Image.open( ) in torch dataset.__getitem__(self, i).
            This is a RAM efficient way
        """
        dict_key = self.dict_int_mapping[i]
        if self.load:
            img = self.loaded_imgs[dict_key]
            json_img = self.loaded_jsons[dict_key]
        # Do some normalizing/cropping
        return torch.tensor(img), torch.tensor(json_img)


class CustomImageDataset(Dataset):
    
    def __init__(self, imgs, masks, json_mapping):
        super(CustomImageDataset, self).__init__()
        self.imgs = imgs
        self.masks = masks
        self.json_mapping = json_mapping
        # Reshape images
        self.resize = transforms.Resize((512, 512))
        self.totensor = transforms.ToTensor()
        return
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]
        img = np.array(self.resize(Image.fromarray(img)))
        mask = np.array(self.resize(Image.fromarray(mask)))
        return self.totensor(img), self.totensor(mask)

def parse_dataset(dataset_info, num_workers):
    print("Dataset info: ", dataset_info)
    dataset = TrainerDataset(dataset_info, num_workers)
    return dataset

def create_json_mapping(jsons):
    label_mapping = {}
    counter = 1
    for key in jsons.keys():
        json_mask = jsons[key]
        with open(json_mask, "r") as json_mask:
            json_mask = json.load(json_mask)
            # img_height = json_mask['imageHeight']
            # img_width = json_mask['imageWidth']
            img_shapes = json_mask['shapes']
            # img_mask = np.zeros((img_height, img_width))
            for shape in img_shapes:
                shape_label = shape['label']
                polys = shape['points']
                if shape_label not in label_mapping.keys():
                    label_mapping[shape_label] = counter
                    counter += 1
                label_val = label_mapping[shape_label]
                # opencv.fillPoly(img_mask, pts=np.int32([polys]), color=label_val)
    return label_mapping, counter


def create_datasets(imgs, jsons, split=0.8, load=True):
    '''
        imgs: Dict
        jsons: Dict
        Split is 0-1 fraction of for example to do 80% training data and 20% testing data
        For very large datasets we might not want to keep all the images in RAM
        Will have to change this to be more generic for RNN/ANN based tasks
    '''
    total_num_imgs = len(imgs.keys())
    total_num_jsons = len(jsons.keys())
    num_train = int(total_num_imgs*split)
    num_test = total_num_imgs - num_train
    print("Total: {},{} Train: {}, Test: {}".format(total_num_imgs, total_num_jsons, num_train, num_test))
    train_keys = random.sample(imgs.keys(), num_train)
    train_imgs = {}
    train_jsons = {}
    test_imgs = {}
    test_jsons = {}
    for key in imgs.keys():
        if key not in train_keys:
            test_jsons[key] = jsons[key]
            test_imgs[key] = imgs[key]
        else:
            train_jsons[key] = jsons[key]
            train_imgs[key] = imgs[key]
    print("Train imgs: {}, Test imgs: {}".format(len(train_imgs), len(test_imgs)))
    # Create coherent json mapping
    json_mapping = create_json_mapping(jsons)
    train_dataset = ImageDataset(train_imgs, train_jsons, load, json_mapping)
    test_dataset = ImageDataset(test_imgs, test_jsons, load, json_mapping)

    return