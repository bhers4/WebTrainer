import torch
import numpy as np
import enum
import datetime
import os
import json


class ModelTasks(enum.Enum):
    classification = 1
    segmentation = 2


# Main Trainer
class Trainer(object):

    def __init__(self):
        '''
            Set Default Parameters
        '''
        # Number of Epochs to Train
        self.num_epoch = 0
        self.curr_epoch = 0
        # Datasets
        self.dataset_trainer = None
        self.train_dataset = None
        self.test_dataset = None
        # Task
        self.task = None
        # Flag to say good to train
        self.valid_args = False
        # Model
        self.model = None
        # Loss Function for training
        self.loss = None
        # Optimizer
        self.optim = None
        # Epoch losses
        self.epoch_losses = []
        self.epoch_test_losses = []
        self.test_accs = []
        self.train_accs = []
        # Iter losses
        self.iter_losses = []
        # Active flag
        self.active = False
        # Save Config
        self.config = None
        # Run id
        self.run_id = None
        # Saved For some +1 and +5 epochs logic
        self.saved = False
        # Save imgs dict
        self.save_img_info = {}
        return

    def set_epochs(self, num_epochs):
        self.num_epoch = num_epochs
        return

    def get_epochs(self):
        return self.num_epoch

    def set_trainer_dataset(self, dataset):
        self.dataset_trainer = dataset
        return

    def set_task(self, task):
        self.task = task
        return

    # Model Getters/Setters
    def set_model(self, model):
        self.model = model
        return

    def get_model(self):
        return self.model
    # Loss Function Getter/Setter
    def set_loss(self, loss):
        self.loss = loss
        return

    def get_loss(self):
        return self.loss
    # Optimizer Getter/Setter
    def set_optim(self, optim):
        self.optim = optim
        return

    def get_optim(self):
        return self.optim

    def check_parameters(self):
        # First check, number of epochs is greater than 0
        if self.num_epoch > 0:
            self.valid_args = True
        return

    def test_network(self, test_iter):
        if not self.valid_args:
            print("Invalid args")
            return
            # Enumerate over dataset
        self.test_losses = []
        self.test_correct = 0
        self.test_total = 0
        for i, data in enumerate(test_iter, 0):
            self.optim.zero_grad()
            inputs = data[0].to(self.device)
            targets = data[1].to(self.device)
            output = self.model(inputs)
            if self.task.value == 1:
                prediction = torch.max(output, dim=1)[1]
                correct = (prediction == targets).sum()
                self.test_correct += correct.item()
                self.test_total += prediction.shape[0]
                # Dont convert output to long
                local_loss = self.loss(output, targets)
            elif self.task.value == 2:
                targets = targets.squeeze(1)
                local_loss = self.loss(output, targets.long())
            # local_loss = self.loss(output, targets)
            local_loss.backward()
            self.optim.step()
            self.test_losses.append(local_loss.item())
        return np.mean(self.test_losses)

    def run_n_epochs(self, train_info, add_n):
        if self.active:
            print("Network already actively training")
        else:
            self.check_parameters()
            if not self.valid_args:
                print("Havent set everything in JSON")
                return

            project_name, dataset_name, batch_size, shuffle, optim, lr = train_info
            # Dont allow changing name, models, or optim but only lr
            if self.config['name'] != project_name:
                print("Changed Project Name to ", project_name)
            if self.dataset_trainer.dataset_name != dataset_name:
                print("Dataset Name changed: ", dataset_name)
            if self.dataset_trainer.batch_size != batch_size:
                print("Batch size changed: ", batch_size)
            # Optim
            if float(self.config['optim']) != lr:
                print("Learning rate changed: ", lr)
            # Set Active Flag
            self.active = True
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            print("Running {} more epochs".format(add_n))
            for epoch in range(self.curr_epoch+1, self.curr_epoch+add_n+1,1):
                # Get iterators
                self.curr_epoch = epoch
                self.train_dataset = self.dataset_trainer.get_train_iter()
                self.test_dataset = self.dataset_trainer.get_test_iter()
                self.iter_losses = []
                self.train_correct = 0
                self.train_total = 0

                # Enumerate over dataset
                for i, data in enumerate(self.train_dataset, 0):
                    self.optim.zero_grad()
                    inputs = data[0].to(self.device)
                    targets = data[1].to(self.device)
                    # Get output
                    output = self.model(inputs)
                    if self.task.value == 1:
                        prediction = torch.max(output, dim=1)[1]
                        correct = (prediction == targets).sum()
                        self.train_correct += correct.item()
                        self.train_total += prediction.shape[0]
                        # Dont convert to long
                        local_loss = self.loss(output, targets)
                    elif self.task.value == 2:
                        targets = targets.squeeze(1)
                        local_loss = self.loss(output, targets.long())
                    local_loss.backward()
                    self.optim.step()
                    self.iter_losses.append(local_loss.item())
                self.epoch_losses.append(np.mean(self.iter_losses))
                test_loss = self.test_network(self.test_dataset)
                self.epoch_test_losses.append(test_loss)
                test_acc = self.test_correct / self.test_total
                train_acc = self.train_correct / self.train_total
                self.test_accs.append(test_acc)
                print("Epoch: %d, Loss: %.4f, Test Loss: %.4f, Train Acc: %.2f Test Acc: %.2f" % (epoch,
                                                                                                  self.epoch_losses[
                                                                                                      -1],
                                                                                                  test_loss,
                                                                                                  train_acc * 100,
                                                                                                  test_acc * 100))
                self.train_accs.append(self.train_correct / self.train_total)
            self.save()
            self.active = False


    def save_epoch_imgs(self):
        if self.task.value == 1:  # This makes sure we only do this for classification
            from matplotlib import pyplot as plt
            from PIL import Image
            data = self.train_dataset.next()
            # Get classes
            data_classes = self.dataset_trainer.train_dataset.classes
            # print("Data classes: ", data_classes)
            data_input = data[0]
            data_targets = data[1]
            # They will be 4d tensors
            if data_input.shape[0] >= 3:
                data_0 = data_input[0, :, :, :]
                target_0 = data_targets[0].item()
                target_0 = data_classes[target_0]
                data_1 = data_input[1, :, :, :]
                target_1 = data_targets[1].item()
                target_1 = data_classes[target_1]
                data_2 = data_input[2, :, :, :]
                target_2 = data_targets[2].item()
                target_2 = data_classes[target_2]
                self.save_img_info['target_0'] = target_0
                self.save_img_info['target_1'] = target_1
                self.save_img_info['target_2'] = target_2
                # Get predictions
                output = self.model(data_input.cuda())
                prediction = torch.max(output, dim=1)[1]
                pred_0 = data_classes[prediction[0]]
                pred_1 = data_classes[prediction[1]]
                pred_2 = data_classes[prediction[2]]
                self.save_img_info['pred_0'] = pred_0
                self.save_img_info['pred_1'] = pred_1
                self.save_img_info['pred_2'] = pred_2
                # Get cwd
                curr_dir = os.getcwd()
                # print("Curr dir: ", curr_dir)
                ui_static_dir = os.path.join(curr_dir, "webui/static/images")
                if not os.path.isdir(ui_static_dir):
                    os.mkdir(ui_static_dir)
                
                d_0 = np.zeros((data_0.shape[1], data_0.shape[2], data_0.shape[0]))
                for j in range(data_0.shape[0]):
                    d_0[:, :, j] = data_0[j, :, :]
                # Check if image is from 0->1
                if np.max(d_0) <= 1.0:
                    d_0 = d_0 * 255
                    d_0 = d_0.astype(np.uint8)
                d_1 = np.zeros((data_1.shape[1], data_1.shape[2], data_1.shape[0]))
                for j in range(data_1.shape[0]):
                    d_1[:, :, j] = data_1[j, :, :]
                # Check if image is from 0->1
                if np.max(d_1) <= 1.0:
                    d_1 = d_1 * 255
                    d_1 = d_1.astype(np.uint8)
                d_2 = np.zeros((data_2.shape[1], data_2.shape[2], data_2.shape[0]))
                for j in range(data_2.shape[0]):
                    d_2[:, :, j] = data_2[j, :, :]
                # Check if image is from 0->1
                if np.max(d_2) <= 1.0:
                    d_2 = d_2 * 255
                    d_2 = d_2.astype(np.uint8)
                if d_0.shape[2] == 1:
                    d_0 = np.array(Image.fromarray(d_0[:, :, 0]).convert('RGB'))
                if d_1.shape[2] == 1:
                    d_1 = np.array(Image.fromarray(d_1[:, :, 0]).convert('RGB'))
                if d_2.shape[2] == 1:
                    d_2 = np.array(Image.fromarray(d_2[:, :, 0]).convert('RGB'))
                plt.imsave(os.path.join(ui_static_dir, "data_0.png"), d_0)
                plt.imsave(os.path.join(ui_static_dir, "data_1.png"), d_1)
                plt.imsave(os.path.join(ui_static_dir, "data_2.png"), d_2)
        return

    def train_network(self, train_info):
        self.check_parameters()
        if not self.valid_args:
            print("Havent set everything in JSON")
            return
        # Do something about potentially changed parameters here from UI
        project_name, dataset_name, batch_size, shuffle, optim, lr = train_info
        if self.config['name'] != project_name:
            print("Changed Project Name to ", project_name)
        if self.dataset_trainer.dataset_name != dataset_name:
            print("Dataset Name changed: ", dataset_name)
        if self.dataset_trainer.batch_size != batch_size:
            print("Batch size changed: ", batch_size)
        # Optim
        if self.config['optim'] != lr:
            print("Learning rate changed: ", lr)
        # Run ID
        self.run_id = "".join(self.config['name'].split(" "))
        self.project_name = str(self.run_id)
        curr_date = datetime.datetime.now()
        self.run_id += "_"+str(curr_date.year)+str(curr_date.month)+str(curr_date.day)+"_"+str(curr_date.hour)+\
                       "_"+str(curr_date.minute)+"_"+str(curr_date.second)
        print("Self run id: ", self.run_id)
        # Set Active Flag
        self.active = True
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("Starting training...")
        for epoch in range(self.num_epoch):
            # Get iterators
            self.curr_epoch = epoch
            self.train_dataset = self.dataset_trainer.get_train_iter()
            self.test_dataset = self.dataset_trainer.get_test_iter()
            self.iter_losses = []
            self.train_correct = 0
            self.train_total = 0
            # Get first 3 images
            self.save_epoch_imgs()
            # Enumerate over dataset
            for i, data in enumerate(self.train_dataset, 0):
                self.optim.zero_grad()
                inputs = data[0].to(self.device)
                targets = data[1].to(self.device)
                # Get output
                output = self.model(inputs)
                if self.task.value == 1:
                    prediction = torch.max(output, dim=1)[1]
                    correct = (prediction == targets).sum()
                    self.train_correct += correct.item()
                    self.train_total += prediction.shape[0]
                    # Dont convert to long
                    local_loss = self.loss(output, targets)
                elif self.task.value == 2:
                    targets = targets.squeeze(1)
                    local_loss = self.loss(output, targets.long())
                local_loss.backward()
                self.optim.step()
                self.iter_losses.append(local_loss.item())
            # Append average loss of the epoch
            self.epoch_losses.append(np.mean(self.iter_losses))
            # Get Test loss
            test_loss = self.test_network(self.test_dataset)
            # Log Test Loss
            self.epoch_test_losses.append(test_loss)
            test_acc = self.test_correct / (self.test_total+1)
            train_acc = self.train_correct / (self.train_total+1)
            self.test_accs.append(test_acc)
            self.train_accs.append(self.train_correct/(self.train_total+1))
            # End of Epoch Print
            print("Epoch: %d, Loss: %.4f, Test Loss: %.4f, Train Acc: %.2f Test Acc: %.2f" % (epoch,
                                                                                                  self.epoch_losses[
                                                                                                      -1],
                                                                                                  test_loss,
                                                                                                  train_acc * 100,
                                                                                                  test_acc * 100))
        self.save()
        self.active = False

        # TEMPORARY thing
        if self.task.value == 2 and not self.config['server']['active']:
            from matplotlib import pyplot as plt
            print("Self dataset trainer mapping: ", self.dataset_trainer.json_mapping)
            for i, data in enumerate(self.dataset_trainer.get_test_iter(), 0):
                inputs = data[0].to(self.device)
                targets = data[1].to(self.device)
                output = self.model(inputs)
                prediction = torch.max(output, dim=1)[1]
                for j in range(inputs.shape[0]):
                    img = inputs[j].permute(1, 2, 0)
                    img = img.cpu().detach().numpy()
                    plt.imshow(img)
                    pred = prediction[j]
                    pred = pred.cpu().detach().numpy()
                    plt.imshow(pred, alpha=0.3)
                    plt.show()
        return

    def save(self):
        # Make sure history dir exists
        history_dir = os.path.join(os.getcwd(), "history")
        if not os.path.isdir(history_dir):
            os.mkdir(history_dir)
        run_dir = os.path.join(history_dir, self.run_id)
        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)
        json_file = os.path.join(run_dir, 'run.json')
        with open(json_file, "w+") as json_data:
            run_data = {}
            run_data['model_info'] = self.config['models']
            run_data['train_loss'] = self.epoch_losses
            run_data['test_loss'] = self.epoch_test_losses
            run_data['train_accs'] = self.train_accs
            run_data['test_accs'] = self.test_accs
            json.dump(run_data, json_data)
        # Save model
        model_file = os.path.join(run_dir, 'run.pt')
        torch.save(self.model.state_dict(), model_file)
        return