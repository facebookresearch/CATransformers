# carbon-nas

Repository Structure
* eval: contains scripts for pruning the model and evaluating model accuracy
* open_clip_custom: modified open_clip repository for training the pruned models
* optimization: contains scripts for running the multi-objective optimization with AX
* phaze: Contains the estimation toolchain for estimating latency and carbon
* Configuration.py contains configuration settings for the optimization run. 

#Setup
`conda env create -f environment.yml`
`conda activate env`
`./setup.sh`

#Quick start
`python main.py --metric=<metric> --name=<Run_name>`
