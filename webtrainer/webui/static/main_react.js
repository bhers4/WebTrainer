'use-strict';
// import React from 'react';
// import ReactDOM from 'react-dom';

const e = React.createElement;

// Navbar Component
class WebNavBar extends React.Component {
    constructor(props) {
        super(props);
    }
    render(){
        return(
          <div>
              <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
                  <a className="navbar-brand" href="/">Web Trainer</a>
                  <button className="navbar-toggler" type="button" data-toggle="collapse"
                          data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent"
                          aria-expanded="false" aria-label="Toggle navigation">
                      <span className="navbar-toggler-icon"></span>
                  </button>
              </nav>
          </div>
        );
    }
}

// Input parts
class RunInfo extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            value: "",
            project_name: "",
            dataset_name: "",
            batch_size: 0,
            shuffle: true,
            split: 0.8,
            num_epochs: 1.0,
            model: "Resnet18",
            num_classes: 10,
            optim: "Adam",
            lr: 0.01
        };
        // Refs
        this.train_btn_ref = React.createRef();
        // Graph Data
        this.graph_data = [];
        this.acc_data = [];
        // Interval id
        this.interval_id = "";
        // Placeholder for image div
        this.image_div = "";

        this.get_run_info = this.get_run_info.bind(this);
        this.update_project_info = this.update_project_info.bind(this);
        this.set_run_info = this.set_run_info.bind(this);
        this.update_batch_size = this.update_batch_size.bind(this);
        this.update_split = this.update_split.bind(this);
        this.update_optim = this.update_optim.bind(this);
        this.update_lr = this.update_lr.bind(this);
        this.update_model = this.update_model.bind(this);
        this.start_training = this.start_training.bind(this);
        this.get_graph_data = this.get_graph_data.bind(this);
        this.run_1_epoch = this.run_1_epoch.bind(this);
        this.run_5_epoch = this.run_5_epoch.bind(this);
        this.run_10_epoch = this.run_10_epoch.bind(this);

    }

    // Do this or else its infinite
    componentDidMount(){
        this.get_run_info();
        //
        this.graph_data = [{
            x: [1, 2, 3],
            y: [2, 4, 5],
            mode: 'lines+markers',
            marker: {color: 'blue'}
        }];
        //
        // Plotly.newPlot('plotly_div', this.graph_data);

    }

    set_run_info(project_name, dataset_name, batch_size, model, lr, split, num_epochs, num_classes, optim){
        this.setState({ project_name: project_name,
                        dataset_name: dataset_name,
                        batch_size: batch_size,
                        model: model,
                        lr: lr,
                        split: split,
                        optim: optim,
                        num_epochs: num_epochs,
                        num_classes: num_classes});
    }

    update_project_info(event){
        this.setState({project_name: event.target.value});
    }

    update_dataset_name(event){
        this.setState({dataset_name: event.target.value});
    }

    update_batch_size(event){
        this.setState({batch_size: event.target.value});
    }

    update_split(event){
        this.setState({split: event.target.value});
    }

    update_epochs(event){
        this.setState({num_epochs: event.target.value});
    }

    update_optim(event){
        this.setState({optim: event.target.value});
    }

    update_lr(event){
        this.setState({lr: event.target.value});
    }

    update_model(event){
        this.setState({model: event.target.value});
    }


    get_run_info(){
        // Could swap to fetch but I prefer ajax
        $.ajax({
           url: "/run/info",
            type: "GET",
            dataType: 'json',
            success: (response)=> {
                var name = response.name;
                var dataset_name = response.dataset_name;
                var batch_size = response.batch_size;
                var model = response.model;
                var split = response.split;
                var lr = response.lr;
                var num_epochs = response.num_epochs;
                var num_classes = response.num_classes;
                var optim = response.optim;
                this.set_run_info(name, dataset_name, batch_size, model, lr, split, num_epochs, num_classes, optim);
            },
            error: (err)=>{
               console.log("Err: ", err);
            }
        });
    }

    start_training(){
        // Create Form Data
        var training_form = new FormData();
        training_form.append('project_name', this.state.project_name);
        training_form.append('dataset_name', this.state.dataset_name);
        training_form.append('num_epochs', this.state.num_epochs);
        training_form.append('batch_size', this.state.batch_size);
        training_form.append('shuffle', this.state.shuffle);
        training_form.append('optim', this.state.optim);
        training_form.append('lr', this.state.lr);

        $.ajax({
            url: "/train",
            type: "POST",
            cache: false,
            processData: false,
            contentType: false,
            data: training_form,
            success: (response)=>{
                console.log("Response: ", response);
                // Disable button
                var btn_ref = this.train_btn_ref;
                console.log("Btn ref: ", btn_ref);
                //btn_ref.disabled = true;
                // Set Interval
                this.interval_id = setInterval(this.get_graph_data, 10000);
            },
            error: (err)=>{
                console.log("Err: ", err);
            }
        });
    }

    run_1_epoch(){
        // Create Form Data
        var training_form = new FormData();
        training_form.append('project_name', this.state.project_name);
        training_form.append('dataset_name', this.state.dataset_name);
        training_form.append('num_epochs', this.state.num_epochs);
        training_form.append('batch_size', this.state.batch_size);
        training_form.append('shuffle', this.state.shuffle);
        training_form.append('optim', this.state.optim);
        training_form.append('lr', this.state.lr);
        training_form.append('add_epochs', 1.0);

        $.ajax({
        url:'/train/add',
        type: 'post',
        cache: false,
        processData: false,
        contentType: false,
        data: training_form,
        success: (response)=>{
            console.log("Response: ", response);
        },
        error: (err)=>{
            console.log("Err: ", err);
        }
        });
        this.interval_id = setInterval(this.get_graph_data, 10000);
        // disable_boxes();
    }

    run_5_epoch(){
        // Create Form Data
        var training_form = new FormData();
        training_form.append('project_name', this.state.project_name);
        training_form.append('dataset_name', this.state.dataset_name);
        training_form.append('num_epochs', this.state.num_epochs);
        training_form.append('batch_size', this.state.batch_size);
        training_form.append('shuffle', this.state.shuffle);
        training_form.append('optim', this.state.optim);
        training_form.append('lr', this.state.lr);
        training_form.append('add_epochs', 5.0);

        $.ajax({
        url:'/train/add',
        type: 'post',
        cache: false,
        processData: false,
        contentType: false,
        data: training_form,
        success: (response)=>{
            console.log("Response: ", response);
        },
        error: (err)=>{
            console.log("Err: ", err);
        }
        });
        this.interval_id = setInterval(this.get_graph_data, 10000);
        // disable_boxes();
    }

    run_10_epoch(){
        // Create Form Data
        var training_form = new FormData();
        training_form.append('project_name', this.state.project_name);
        training_form.append('dataset_name', this.state.dataset_name);
        training_form.append('num_epochs', this.state.num_epochs);
        training_form.append('batch_size', this.state.batch_size);
        training_form.append('shuffle', this.state.shuffle);
        training_form.append('optim', this.state.optim);
        training_form.append('lr', this.state.lr);
        training_form.append('add_epochs', 10.0);

        $.ajax({
        url:'/train/add',
        type: 'post',
        cache: false,
        processData: false,
        contentType: false,
        data: training_form,
        success: (response)=>{
            console.log("Response: ", response);
        },
        error: (err)=>{
            console.log("Err: ", err);
        }
        });
        this.interval_id = setInterval(this.get_graph_data, 10000);
        // disable_boxes();
    }

    get_graph_data(){
        $.ajax({
            url:'/train/data/',
            type:'GET',
            dataType: 'json',
            success: (response)=>{
                var epoch_losses = response.epoch_loss;
                var epoch_test_losses = response.epoch_test_losses;
                var test_accs = response.test_accs;
                var train_accs = response.train_accs;
                var curr_active = response.curr_active;
                var trainer_task = response.trainer_task;
                if(curr_active==false){
                    clearInterval(this.interval_id);
                    // enable_boxes();
                }
                var x_axis = [];
                for(var i=0;i<epoch_losses.length;i++){
                    x_axis.push(i);
                }
                this.graph_data = {
                    x: x_axis,
                    y: epoch_losses,
                    mode: 'lines+markers',
                    name: 'Train',
                    marker: {color: 'blue'}
                };
                var test_loss = {
                   x: x_axis,
                   y: epoch_test_losses,
                   mode: 'lines+markers',
                   name: 'Test',
                   marker: {color: 'red'}
                };
                var layout = {
                    title: 'Loss'
                };
                //
                Plotly.newPlot('loss_graph', [this.graph_data, test_loss], layout);

                this.acc_data = {
                   x: x_axis,
                   y: train_accs,
                   mode: 'lines+markers',
                   name: 'Train',
                   marker: {color: 'blue'}
                };

                var test_acc = {
                   x: x_axis,
                   y: test_accs,
                   mode: 'lines+markers',
                   name: 'Test',
                   marker: {color: 'red'}
                };

                //
                layout = {
                    title: 'Accuracy'
                };
                //
                Plotly.newPlot('accuracy_graph', [this.acc_data, test_acc], layout);
            }
        });
        //
        // **** Rewrite this into a list of React items, it doesnt like when I do {this.image_div} cause its not
        // Typical var x = <div></div> its instead document.createElement shit
        // $.ajax({
        //     url:'/train/imgs/',
        //     type:'GET',
        //     dataType: 'json',
        //     success: (response)=> {
        //         var imgs_found = response.imgs_found;
        //         var trainer_classes = response.trainer_classes;
        //         var trainer_task = response.trainer_task;
        //         // console.log("Trainer task: ", trainer_task);
        //         if (trainer_task == 1) {
        //             var targets = [];
        //             targets.push(trainer_classes['target_0']);
        //             targets.push(trainer_classes['target_1']);
        //             targets.push(trainer_classes['target_2']);
        //             // Pred
        //             var predictions = [];
        //             predictions.push(trainer_classes['pred_0']);
        //             predictions.push(trainer_classes['pred_1']);
        //             predictions.push(trainer_classes['pred_2']);
        //             // Images
        //             var imgs = [];
        //             imgs.push(imgs_found['data_0.png']);
        //             imgs.push(imgs_found['data_1.png']);
        //             imgs.push(imgs_found['data_2.png']);
        //             // Get imgs row
        //             var imgs_row_div = document.createElement('div');
        //             imgs_row_div.className = "row";
        //             if (imgs_row_div.innerHTML != "") {
        //                 imgs_row_div.innerHTML = "";  // This clears the div
        //             }
        //             var i;
        //             for(i=0;i<(imgs.length);i++) {
        //                 // Create overall div (lg-4 md-6)
        //                 var img_var = document.createElement('div');
        //                 img_var.className = "col-lg-4 col-md-6";
        //                 // Create a div to put all this in
        //                 // Sub div for img
        //                 var img_holder = document.createElement('div');
        //                 img_holder.className = "col-lg-12 col-md-12";
        //                 var actual_img = document.createElement('img');
        //                 var seconds = new Date().getTime() / 1000;
        //                 seconds = parseInt(seconds);
        //
        //                 actual_img.src = "webui/static/images/" + imgs[i] + "?" + seconds;
        //
        //                 actual_img.style.width = "80%";
        //                 actual_img.style.margin = "1rem";
        //                 // Add img to col then add col to row
        //                 img_holder.appendChild(actual_img);
        //                 img_var.appendChild(img_holder);
        //                 // Prediction div
        //                 var tag_holder = document.createElement('div');
        //                 tag_holder.className = "col-lg-12 col-md-12";
        //                 // Create h3 with the class
        //                 var img_tag = document.createElement('h4');
        //
        //                 img_tag.innerText = "Target: " + targets[i];
        //                 var pred_tag = document.createElement('h4');
        //                 pred_tag.innerText = "Pred: " + predictions[i];
        //                 tag_holder.appendChild(img_tag);
        //                 tag_holder.appendChild(pred_tag);
        //                 img_var.appendChild(tag_holder);
        //                 imgs_row_div.appendChild(img_var);
        //             }
        //             this.image_div = imgs_row_div;
        //             this.setState({"image_div": this.image_div});
        //         }
        //     }
        // });
    }

    render(){
        var project = <div className="row">
                        <div className="col-lg-8">
                            <label htmlFor="project_name">Project Name</label>
                            <input type="text" className="form-control" id="project_name"
                                   value={this.state.project_name} onChange={this.update_project_info}></input>
                        </div>
                      </div>;
        var dataset_info = <div className="row">
            <h3>Dataset Info</h3>
        </div>;
        // Model and others, use react to make it a set of options
        var data_inputs = <div className="row">
            <div className="col-lg-2 col-md-4 col-sm-6">
                <label htmlFor="dataset_name">Dataset Name:</label>
                <input className="form-control" type="text" id="dataset_name" value={this.state.dataset_name}
                onChange={this.update_dataset_name}></input>
            </div>
            <div className="col-lg-2 col-md-4 col-sm-6">
                <label htmlFor="batch_size">Batch Size:</label>
                <input className="form-control" type="number" id="batch_size" value={this.state.batch_size}
                onChange={this.update_batch_size}></input>
            </div>
            <div className="col-lg-2 col-md-4 col-sm-6">
                <label htmlFor="shuffle">Shuffle</label>
                <select className="form-control" id="shuffle">
                    <option>True</option>
                    <option>False</option>
                </select>
            </div>
            <div className="col-lg-2 col-md-4 col-sm-6">
                <label htmlFor="train_split">Split:</label>
                <input className="form-control" type="number" id="train_split" value={this.state.split}
                       onChange={this.update_split}></input>
            </div>
            <div className="col-lg-2 col-md-4 col-sm-6">
                <label htmlFor="num_epochs">Number of Epochs</label>
                <input className="form-control" type='number' id="num_epochs" value={this.state.num_epochs}
                onChange={this.update_epochs}></input>
            </div>
            <div className="col-lg-2 col-md-4 col-sm-6">
                <label htmlFor="model">Model</label>
                <select className="form-control" id="model" onChange={this.update_model} defaultValue="Resnet18">
                    <option>Resnet18</option>
                    <option>Resnet34</option>
                    <option>Mobilenet</option>
                    <option>Squeezenet</option>
                    <option>Inceptionv3</option>
                </select>
            </div>
            <div className="col-lg-2 col-md-4 col-sm-6">
                <label htmlFor="num_classes">Number of Classes</label>
                <input className="form-control" type='number' id="num_classes" value={this.state.num_classes}
                       disabled></input>
            </div>
            <div className="col-lg-2 col-md-4 col-sm-6">
                <label htmlFor="optim">Optim</label>
                <input className="form-control" type='text' id='optim' value={this.state.optim}
                onChange={this.update_optim}/>
            </div>
            <div className="col-lg-2 col-md-4 col-sm-6">
                <label htmlFor="lr">Learning Rate</label>
                <input className="form-control" type='number' id='lr' value={this.state.lr} onChange={this.update_lr}/>
            </div>
        </div>;

        // Train Div
        var train_div = <div className="row">
            <div className="col-lg-12 col-md-12">
                <h3>Train</h3>
            </div>
            <div className="col-lg-12" style={{margin: "0.25rem"}}>
                <button className="btn btn-primary" ref={this.train_btn_ref} onClick={this.start_training}>
                    Start Training</button>
            </div>
        </div>

        var graph_divs = <div className="row">
            <div id="loss_graph" className="col-lg-6 col-md-12">

            </div>
            <div id="accuracy_graph" className="col-lg-6 col-md-12">

            </div>
        </div>
        //
        var run_more_epochs = <div className="row">
            <div className="col-lg-3 col-md-4 col-sm-6" style={{margin: "0.25rem"}}>
                <button className="btn btn-success" onClick={this.run_1_epoch}>Run 1 More Epoch</button>
            </div>
            <div className="col-lg-3 col-md-4 col-sm-6" style={{margin: "0.25rem"}}>
                <button className="btn btn-secondary" onClick={this.run_5_epoch}>Run 5 More Epochs</button>
            </div>
            <div className="col-lg-3 col-md-4 col-sm-6" style={{margin: "0.25rem"}}>
                <button className="btn btn-info" onClick={this.run_10_epoch}>Run 10 More Epochs</button>
            </div>
        </div>;

        return(
            <div>
                {project}
                {dataset_info}
                {data_inputs}
                {train_div}
                {graph_divs}
                {run_more_epochs}
                {this.image_div}
            </div>
        )
    }
}

// Classification Page
class ClassificationPage extends React.Component {
    constructor(props) {
        super(props);
        this.state = {value:""};
        // Init Functions
        this.get_run_info = this.get_run_info.bind(this);
    }

    get_run_info(){
        // Could swap to fetch but I prefer ajax
        $.ajax({
           url: "/run/info",
            type: "GET",
            success: (response)=> {
                console.log("Got run info");
                var name = response.name;
            },
            error: (err)=>{
               console.log("Err: ", err);
            }
        });
    }

    render(){
        // Call get run info
        this.get_run_info();
        var header = <WebNavBar></WebNavBar>
        var run_info = <RunInfo></RunInfo>
        return(
          <div>
              {header}
              <div className="container">
                  <div className="row d-flex justify-content-center">
                      <div className="col-lg-12 col-md-12">
                          <h3>WebTrainer</h3>
                      </div>
                  </div>
                  {run_info}
              </div>
          </div>
        );
    }

}

// END REACT

// Add in quick thing that checks the task

const dom_container = document.querySelector('#react_div');
ReactDOM.render(e(ClassificationPage), dom_container);