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
        // this.get_run_info();
        var header = <WebNavBar></WebNavBar>;
        return(
          <div>
              {header}
          </div>
        );
    }

}

// END REACT

// Add in quick thing that checks the task

const dom_container = document.querySelector('#react_div');
ReactDOM.render(e(ClassificationPage), dom_container);