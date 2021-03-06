import argparse
import json
from datasets.datasets import parse_dataset
from trainer.trainer import Trainer
from trainer.vision_embedding import VisionEmbedding
from models.models import load_models
from loss.loss import load_loss
from optim.optim import load_optim
from matplotlib import pyplot as plt
from webui.ui import WebInterface
from models.models import ModelTasks

help='deep learning framework'

def add_arguments(parser):
    # Path to JSON file
    parser.add_argument('--configs', help='Path to json file')
    # Number of workers
    parser.add_argument('--j', type=int, default=4, help='Number of Threads/workers')
    # Once basic UI is done do "start server"
    return

def main(args):
    # Load config file
    configs_file = args.configs
    configs_file = json.load(open(configs_file, "r"))
    # Load Threads
    threads = args.j
    # Load Server options
    server_options = configs_file['server']
    print("Threads: ", threads)
    print("Configs: ", configs_file)
    print("Server Options: ", server_options)

    # Get Dataset Parameters
    dataset = parse_dataset(configs_file['dataset'], threads)
    run_args = configs_file['run']
    # Model
    model = load_models(configs_file['models'], dataset)
    if configs_file['models']['task'] == 'classification':
        task = ModelTasks.classification
    elif configs_file['models']['task'] == 'segmentation':
        task = ModelTasks.segmentation
    else:
        task = ModelTasks.classification
    
    # Trainer
    if configs_file['name'] == "vision_embedding":
        nn_trainer = VisionEmbedding()
    else:
        nn_trainer = Trainer()
    # Set number of epochs
    nn_trainer.set_epochs(run_args['num_epochs'])
    # Set Datasets
    nn_trainer.set_trainer_dataset(dataset)
    # Set Model
    nn_trainer.set_model(model)
    # Set Loss
    loss = load_loss(configs_file["loss"])
    nn_trainer.set_loss(loss)
    # Task
    nn_trainer.set_task(task)
    # Optim
    optim = load_optim(configs_file['optim'], model, epochs=run_args['num_epochs'])
    nn_trainer.set_optim(optim)
    # Pass in configs
    nn_trainer.config = configs_file
    # Train!
    server_status = server_options['active']
    if server_status:
        webui = WebInterface(configs_file['name'], ip_config=server_options['host'], port=server_options['port'])
        webui.set_trainer(nn_trainer)
        # If you run it then you gotta hand off control to clicking webserver
        webui.run()
    else:
        # project_name, dataset_name, batch_size, shuffle, optim, lr = train_info
        train_info = (configs_file['name'], configs_file['dataset']['name'], configs_file['dataset']['batch_size'],
                      configs_file['dataset']['shuffle'], configs_file['optim']['name'], configs_file['optim']['lr'])
        nn_trainer.train_network(train_info)
        epoch_losses = nn_trainer.epoch_losses
        epoch_test_losses = nn_trainer.epoch_test_losses
        test_accs = nn_trainer.test_accs
        plt.subplot(1, 2, 1)
        plt.plot(epoch_losses, c='b', label='train')
        plt.plot(epoch_test_losses, c='r', label='test')
        plt.title("Epoch losses")
        plt.subplot(1, 2, 2)
        plt.plot(nn_trainer.train_accs, c='b', label='Train Acc')
        plt.plot(test_accs, c='m', label='Test Acc')
        plt.title("Accuracies")
        plt.legend()
        plt.show()
    return

if __name__ == "__main__":
    # Arg parser
    parser = argparse.ArgumentParser(description=help)
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
