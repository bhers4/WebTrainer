from flask import Flask, render_template, request
import os
import json
import threading


class WebInterface:

    def __init__(self, name, ip_config='0.0.0.0', port=5000):
        # Trainer
        self.trainer = None
        # Ip
        self.ip_config = ip_config
        self.port = port
        # Name
        self.name = name
        # Flask App
        self.app = Flask(name, static_url_path='/webui/static', template_folder='webui/templates/')
        # Call Setup
        self.setup_routes()
        # Trainer Thread
        self.trainer_thread = None
        return

    def setup_routes(self):
        # Main route
        self.add_endpoint('/', endpoint_name='main_page', handler=self.render_mainpage, methods=['GET'])
        # TESTING
        self.add_endpoint('/react', endpoint_name='react_main', handler=self.render_main_react, methods=['GET'])
        # Basic js file route
        self.add_endpoint('/webui/static/<file>', endpoint_name='server static files', handler=self.get_static_file,
                          methods=['GET'])
        # Basic get images from /webui/static/images
        self.add_endpoint('/webui/static/images/<file>', endpoint_name='static_images', handler=self.get_static_images, 
                          methods=['GET'])
        # Training Endpoints
        self.add_endpoint('/train', endpoint_name='train_start', handler=self.train_page, methods=['POST'])
        self.add_endpoint('/train/data/', endpoint_name='train_data', handler=self.get_training_data, methods=['GET'])
        self.add_endpoint('/train/add', endpoint_name='add_more_epochs', handler=self.add_more_epochs, methods=['POST'])
        self.add_endpoint('/train/imgs', endpoint_name='show_train_imgs', handler=self.get_train_imgs, methods=['GET'])
        # Run info
        self.add_endpoint('/run/info', endpoint_name='run_info', handler=self.get_run_info, methods=['GET'])
        return

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, methods=None):
        self.app.add_url_rule(endpoint, endpoint_name, handler, methods=methods)
        return

    def run(self):
        if self.app:
            self.app.run(host=self.ip_config, port=self.port)
        else:
            print("No app setup which shouldn't be possible")
        return

    # Setter for Trainer
    def set_trainer(self, trainer):
        self.trainer = trainer
        return

    # Getter for Trainer
    def get_trainer(self):
        return self.trainer

    # Calls Trainer Main Training Loop
    def train(self, train_info):
        self.trainer.train_network(train_info)
        return

    # /train
    def train_page(self):
        # Get Form info
        project_name = request.form['project_name']
        dataset_name = request.form['dataset_name']
        batch_size = int(request.form['batch_size'])
        shuffle = request.form['shuffle']
        optim = request.form['optim']
        lr = float(request.form['lr'])

        train_info = (project_name, dataset_name, batch_size, shuffle, optim, lr)
        train_func = threading.Thread(target=self.train, args=(train_info,))
        train_func.start()
        self.trainer_thread = train_func
        # self.train(train_info)
        return json.dumps({'status':'OK'})

    # /webui/static/<file>
    def get_static_file(self, file):
        from flask import send_from_directory
        static_file = os.path.join(os.getcwd(), 'webui/static/')
        return send_from_directory(static_file, file)
    
    # /webui/static/images
    def get_static_images(self, file):
        from flask import send_from_directory
        static_file = os.path.join(os.getcwd(), 'webui/static/images/')
        return send_from_directory(static_file, file)

    # /train/data
    def get_training_data(self):
        epoch_loss = self.trainer.epoch_losses
        epoch_test_losses = self.trainer.epoch_test_losses
        test_accs = self.trainer.test_accs
        train_accs = self.trainer.train_accs
        curr_epoch = self.trainer.curr_epoch
        total_epochs = self.trainer.num_epoch
        curr_active = self.trainer.active
        return json.dumps({'status':'OK', 'epoch_loss':epoch_loss, 'epoch_test_losses':epoch_test_losses,
                           'test_accs':test_accs, 'curr_epoch':curr_epoch,'total_epochs':total_epochs,
                           'curr_active':curr_active, 'train_accs': train_accs, 
                           'trainer_task': self.trainer.task.value})

    def add_more_epochs(self):
        project_name = request.form['project_name']
        dataset_name = request.form['dataset_name']
        batch_size = int(request.form['batch_size'])
        shuffle = request.form['shuffle']
        optim = request.form['optim']
        lr = float(request.form['lr'])
        add_epochs = int(request.form['add_epochs'])
        train_info = (project_name, dataset_name, batch_size, shuffle, optim, lr)
        train_func = threading.Thread(target=self.trainer.run_n_epochs, args=(train_info,add_epochs))
        train_func.start()
        return json.dumps({'status':'OK'})

    # Get train imgs
    def get_train_imgs(self):
        # Start by getting /static/images/ dir
        imgs_dir = os.path.join(os.getcwd(), "webui/static/images/")
        imgs_found = {}
        for root, dirs, files in os.walk(imgs_dir):
            for file in files:
                if "data_" in file and file.endswith(".png"):
                    # imgs_found.append(file)
                    imgs_found[file] = file
        # Next get list of classes
        trainer_classes = self.trainer.save_img_info
        trainer_task = self.trainer.task.value
        return json.dumps({'status': 'OK', 'imgs_found': imgs_found, "trainer_classes": trainer_classes,
                           'trainer_task': trainer_task})

    # Rendering Functions
    # /
    def render_mainpage(self):
        return render_template('main.html', name=self.name, dataset_name=self.trainer.config['dataset']['name'],
                               batch_size=self.trainer.config['dataset']['batch_size'],
                               shuffle=self.trainer.config['dataset']['shuffle'],
                               split=self.trainer.config['dataset']['split'],
                               num_epochs=self.trainer.config['run']['num_epochs'],
                               model=self.trainer.config['models']['name'],
                               task=self.trainer.config['models']['task'],
                               num_classes=self.trainer.config['models']['num_classes'],
                               lr=self.trainer.config['optim']['lr'],
                               optim=self.trainer.config['optim']['name'])

    # /react --
    def render_main_react(self):
        return render_template("main_react.html")

    # /run/info
    def get_run_info(self):
        name = self.name
        dataset_name = self.trainer.config['dataset']['name']
        batch_size = self.trainer.config['dataset']['batch_size']
        split = self.trainer.config['dataset']['split']
        num_epochs = self.trainer.config['run']['num_epochs']
        model = self.trainer.config['models']['name']
        task = self.trainer.config['models']['task']
        num_classes = self.trainer.config['models']['num_classes']
        lr = self.trainer.config['optim']['lr']
        optim = self.trainer.config['optim']['name']
        return json.dumps({'status': 'OK', "name": name, "dataset_name": dataset_name, "batch_size": batch_size,
                           "split": split, "num_epochs": num_epochs, "model": model, "task": task,
                           "num_classes": num_classes, "lr": lr, "optim": optim})


