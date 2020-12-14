import unittest
import os
import torchvision
from torch.utils.data import DataLoader
import torch


class LossUnitTesting(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(LossUnitTesting, self).__init__(*args, **kwargs)
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
        new_path = os.path.join(new_path, "loss/loss.py")

        # importlib util way
        import importlib.util
        spec = importlib.util.spec_from_file_location('load_loss', new_path)
        parse_loss = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parse_loss)
        self.parse_loss = parse_loss
        return

    def test_BCELoss(self):
        loss_info = {'name':'BCELoss'}
        loss = self.parse_loss.load_loss(loss_info)
        self.assertTrue(isinstance(loss, torch.nn.BCELoss), "Wrong BCELoss")
        return

    def test_CELoss(self):
        loss_info = {'name':'CrossEntropyLoss'}
        loss = self.parse_loss.load_loss(loss_info)
        self.assertTrue(isinstance(loss, torch.nn.CrossEntropyLoss), "Wrong CrossEntropyLoss")
        return

    def test_CustomLoss(self):
        loss_info = {'name': 'Custom'}
        loss = self.parse_loss.load_loss(loss_info)
        self.assertFalse(loss, "Wrong Custom Loss")
        return

if __name__ == "__main__":
    unittest.main()