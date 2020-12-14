import unittest
import os
import torchvision
from torch.utils.data import DataLoader
import torch

#
# class ModelsUnitTesting(unittest.TestCase):
#
#     def __init__(self, *args, **kwargs):
#         super(ModelsUnitTesting, self).__init__(*args, **kwargs)
#         # This adds the dir up one directory to our sys path to allow imports
#         self.import_models_file()
#         self.import_dataset_loader()
#         self.import_resnet18()
#         return
#
#     def import_models_file(self):
#         curr_dir = os.getcwd()
#         main_dir = curr_dir.split('/')
#         del main_dir[-1]
#         new_path = "/"
#         for item in main_dir:
#             new_path = os.path.join(new_path, item)
#         new_path = os.path.join(new_path, "models/models.py")
#
#         # importlib util way
#         import importlib.util
#         spec = importlib.util.spec_from_file_location('load_models', new_path)
#         parse_model = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(parse_model)
#         self.parse_model = parse_model
#         return
#
#     def import_dataset_loader(self):
#         curr_dir = os.getcwd()
#         main_dir = curr_dir.split('/')
#         del main_dir[-1]
#         new_path = "/"
#         for item in main_dir:
#             new_path = os.path.join(new_path, item)
#         new_path = os.path.join(new_path, "datasets/datasets.py")
#
#         # importlib util way
#         import importlib.util
#         spec = importlib.util.spec_from_file_location('parse_dataset', new_path)
#         parse_data = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(parse_data)
#         self.parse_data = parse_data
#         return
#
#     def import_resnet18(self):
#         curr_dir = os.getcwd()
#         main_dir = curr_dir.split('/')
#         del main_dir[-1]
#         new_path = "/"
#         for item in main_dir:
#             new_path = os.path.join(new_path, item)
#         new_path = os.path.join(new_path, "models/resnet.py")
#
#         # importlib util way
#         import importlib.util
#         spec = importlib.util.spec_from_file_location('Resnet18', new_path)
#         parse_resnet18 = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(parse_resnet18)
#         self.parse_resnet18 = parse_resnet18
#         return
#
#     def test_resnet18(self):
#         model_info = {'name': 'Resnet18', 'pretrained': False}
#         dataset_info = {'name':'CIFAR10', 'batch_size':100}
#         dataset = self.parse_data.parse_dataset(dataset_info, 4)
#         model = self.parse_model.load_models(model_info, dataset)
#         self.assertTrue(isinstance(model, self.parse_resnet18.Resnet18))
#         return
#
# if __name__ == "__main__":
#     unittest.main()
