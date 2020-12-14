import unittest
import os
import sys
import torchvision
from torch.utils.data import DataLoader


class DatasetUnitTesting(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(DatasetUnitTesting, self).__init__(*args, **kwargs)
        # This adds the dir up one directory to our sys path to allow imports
        self.get_main_dir()
        return

    def get_main_dir(self):
        curr_dir = os.getcwd()
        main_dir = curr_dir.split('/')
        del main_dir[-1]
        new_path = "/"
        for item in main_dir:
            new_path = os.path.join(new_path, item)
        new_path = os.path.join(new_path, "webtrainer/datasets/datasets.py")

        # importlib util way
        import importlib.util
        spec = importlib.util.spec_from_file_location('parse_dataset', new_path)
        parse_data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parse_data)
        self.parse_data = parse_data
        return

    def test_cifar10(self):
        cifar10_params = {'name': 'CIFAR10', 'batch_size': 100, 'shuffle': True, 'split': 0.8}

        num_workers = 4
        dataset = self.parse_data.parse_dataset(cifar10_params, num_workers)
        # Testing attributes
        self.assertTrue(dataset.dataset_info == cifar10_params, "Object gets modified")
        self.assertTrue(dataset.num_workers == num_workers, "Wrong Number of Workers")
        self.assertTrue(cifar10_params['batch_size']==dataset.batch_size, "Wrong batch size")
        self.assertTrue(dataset.channels == 3, "Wrong number of channels")
        # Test Transforms
        self.assertTrue(isinstance(dataset.transforms,torchvision.transforms.ToTensor), "Bad Torchvision Transforms")
        self.assertTrue(dataset.get_batch_size() == cifar10_params['batch_size'], "Get batch size is wrong")
        # Check DataLoaders
        self.assertTrue(isinstance(dataset.train_loader, DataLoader), "Train loader isnt Dataloader")
        self.assertTrue(isinstance(dataset.test_loader, DataLoader), "Test loader isnt Dataloader")
        return

    def test_mnist(self):
        params = {'name': 'MNIST', 'batch_size': 100, 'shuffle': True, 'split': 0.8}

        num_workers = 4
        dataset = self.parse_data.parse_dataset(params, num_workers)
        # Testing attributes
        self.assertTrue(dataset.dataset_info == params, "Object gets modified")
        self.assertTrue(dataset.num_workers == num_workers, "Wrong Number of Workers")
        self.assertTrue(params['batch_size']==dataset.batch_size, "Wrong batch size")
        self.assertTrue(dataset.channels == 1, "Wrong number of channels")
        # Test Transforms
        self.assertTrue(isinstance(dataset.transforms,torchvision.transforms.ToTensor), "Bad Torchvision Transforms")
        self.assertTrue(dataset.get_batch_size() == params['batch_size'], "Get batch size is wrong")
        # Check DataLoaders
        self.assertTrue(isinstance(dataset.train_loader, DataLoader), "Train loader isnt Dataloader")
        self.assertTrue(isinstance(dataset.test_loader, DataLoader), "Test loader isnt Dataloader")
        return

if __name__ == "__main__":
    unittest.main()