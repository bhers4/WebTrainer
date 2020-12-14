var train_graph;
var acc_graph;
var interval_id;

window.addEventListener('DOMContentLoaded', (event)=>{
   console.log("Event: ", event);

   // Add Button Event Listener
    train_button = document.getElementById('train_go');
    train_button.addEventListener('click', (event)=>{
        run_training(false,1);
    });

    // Setup basic graph
    training_graph = document.getElementById('training_graph');
    chart2d = training_graph.getContext('2d');
    train_graph = new Chart(chart2d, {
        type: 'line',
        data: {
            labels: [2, 3, 4, 5],
            datasets: [{
                label: 'Train Loss',
                yAxisID: 'Loss',
                fill: false,
                backgroundColor: "rgba(20, 120, 255, 1)",
                borderColor: "rgba(20, 120, 255, 1)",
            },
            {
                label: 'Test Loss',
                yAxisID: 'Test Loss',
                fill: false,
                backgroundColor: "rgba(255, 50, 100, 1)",
                borderColor: "rgba(255, 50, 100, 1)",
            }]
        },
        options:{
            scales:{
                yAxes: [{
                    name: 'Loss',
                    id: 'Loss',
                    position: 'left'
                },
                {
                    name: 'Test Loss',
                    id: 'Test Loss',
                    position: 'right'
                }]
            }
        }
    });
    // Set up accuracy graph
    acc_graphs = document.getElementById('acc_graphs');
    chart2d = acc_graphs.getContext('2d');
    acc_graph = new Chart(chart2d, {
        type: 'line',
        data: {
            labels: [2, 3, 4, 5],
            datasets: [{
                label: 'Train Acc',
                yAxisID: 'Acc',
                fill: false,
                backgroundColor: "rgba(20, 120, 255, 1)",
                borderColor: "rgba(20, 120, 255, 1)",
            },
            {
                label: 'Test Acc',
                yAxisID: 'Test Acc',
                fill: false,
                backgroundColor: "rgba(255, 50, 100, 1)",
                borderColor: "rgba(255, 50, 100, 1)",
            }]
        },
        options:{
            scales:{
                yAxes: [{
                    name: 'Acc',
                    id: 'Acc',
                    position: 'left'
                },
                {
                    name: 'Test Acc',
                    id: 'Test Acc',
                    position: 'right'
                }]
            }
        }
    });

    // Set up 1 more, 5 more and 10 more epochs
    btn_1_epoch = document.getElementById('add_1');
    btn_1_epoch.addEventListener('click', ()=>{
        run_training(true, 1);
    })
    btn_5_epoch = document.getElementById('add_5');
    btn_5_epoch.addEventListener('click', ()=>{
        run_training(true, 5);
    })
    btn_10_epoch = document.getElementById('add_10');
    btn_10_epoch.addEventListener('click', ()=>{
        run_training(true, 10);
    })
});

function run_training(add_epochs=false, add_num_epochs=1){

    // Get Form Info
    project_name = document.getElementById('project_name').value;
    dataset_name = document.getElementById('dataset_name').value;
    batch_size = document.getElementById('batch_size').value;
    shuffle = document.getElementById('shuffle').value;
    split = document.getElementById('train_split').value;
    num_epochs = document.getElementById('num_epochs').value;
    model = document.getElementById('model').value;
    optim = document.getElementById('optim').value;
    lr = document.getElementById('lr').value;

    // Form data
    form_data = new FormData();
    form_data.append('project_name', project_name);
    form_data.append('dataset_name', dataset_name);
    form_data.append('batch_size', batch_size);
    form_data.append('shuffle', shuffle);
    form_data.append('optim', optim);
    form_data.append('lr', lr);
    if(add_epochs == true){
        form_data.append('add_epochs',add_num_epochs);
    }

    // Change to a post request
    // Do ajax call
    if(add_epochs==true){
        $.ajax({
        url:'/train/add',
        type: 'post',
        cache: false,
        processData: false,
        contentType: false,
        data: form_data,
        success: (response)=>{
            console.log("Response: ", response);
        },
        error: (err)=>{
            console.log("Err: ", err);
        }
        });
        interval_id = setInterval(check_status, 10000);
        disable_boxes();
    }
    else {
        $.ajax({
        url:'/train',
        type: 'post',
        cache: false,
        processData: false,
        contentType: false,
        data: form_data,
        success: (response)=>{
            console.log("Response: ", response);
        },
        error: (err)=>{
            console.log("Err: ", err);
        }
        });
        interval_id = setInterval(check_status, 10000);
        disable_boxes();
    }
}

function disable_boxes(){
    var proj_name = document.getElementById('project_name');
    proj_name.disabled = true;
    // Dataset name
    var dataset_name = document.getElementById('dataset_name');
    dataset_name.disabled = true;
    // Batch size
    var batch_size = document.getElementById('batch_size');
    batch_size.disabled = true;
    // Train split
    var train_split = document.getElementById('train_split');
    train_split.disabled = true;
    // Learning Rate
    var lr = document.getElementById('lr');
    lr.disabled = true;
    // Number of Epochs
    var num_epochs = document.getElementById('num_epochs');
    num_epochs.disabled = true;
    // Train button
    var train_btn = document.getElementById('train_go');
    train_btn.disabled = true;
}

function enable_boxes(){
    var proj_name = document.getElementById('project_name');
    proj_name.disabled = false;
    var dataset_name = document.getElementById('dataset_name');
    dataset_name.disabled = false;
    var batch_size = document.getElementById('batch_size');
    batch_size.disabled = false;
    // Train split
    var train_split = document.getElementById('train_split');
    train_split.disabled = false;
    // Learning Rate
    var lr = document.getElementById('lr');
    lr.disabled = false;
    // Number of Epochs
    var num_epochs = document.getElementById('num_epochs');
    num_epochs.disabled = false;
    // Train button
    var train_btn = document.getElementById('train_go');
    train_btn.disabled = false;
}

function check_status(){
    // Runs every 5 seconds
    // Get Latest Data
    $.ajax({
        url:'/train/data/',
        type:'GET',
        dataType: 'json',
        success: (response)=>{
            epoch_losses = response.epoch_loss;
            epoch_test_losses = response.epoch_test_losses;
            test_accs = response.test_accs;
            train_accs = response.train_accs;
            curr_active = response.curr_active;
            if(curr_active==false){
                clearInterval(interval_id);
                enable_boxes();
            }
            training_graph = document.getElementById('training_graph');
            chart2d = training_graph.getContext('2d');
            x_axis = [];
            for(i=0;i<epoch_losses.length;i++){
                x_axis.push(i);
            }
            train_graph.data.labels = x_axis;
            train_graph.data.datasets[0].data = epoch_losses;
            train_graph.data.datasets[1].data = epoch_test_losses;
            train_graph.update();

            acc_graphs = document.getElementById('acc_graphs');
            acc_chart2d = acc_graphs.getContext('2d');
            x_axis = [];
            for(i=0;i<epoch_losses.length;i++){
                x_axis.push(i);
            }
            acc_graph.data.labels = x_axis;
            acc_graph.data.datasets[0].data = train_accs;
            acc_graph.data.datasets[1].data = test_accs;
            acc_graph.update();
        }
    })
}