# Phaze

Phaze is a framework to perform the co-optimization between accelerator architecture search and model partitioning for distributed training. For more details, please refer to our ICML 2024 paper, [Integrated Hardware Architecture and Device Placement Search](https://openreview.net/pdf?id=ucl3B05EsX).

## Installation

To install the dependencies for Phaze, run:

```bash
./setup.sh
```

Add the following path variables in `~/.bashrc`:
```bash
export THIRD_PARTY_PATH=$(pwd)/Phaze/third_party_for_phaze
export WHAM_PATH=$THIRD_PARTY_PATH/wham/
export SUNSTONE_PATH=$THIRD_PARTY_PATH/sunstone/
export PYTHONPATH=$THIRD_PARTY_PATH:$WHAM_PATH:$SUNSTONE_PATH:$PYTHONPATH
```

- Phaze uses Gurobi 10.0.1 to solve the ILP formulations. To run the ILP solver, obtain a Gurobi license from the [The Gurobi Website](https://www.gurobi.com/).

## Quick Start

We provide scripts to run the experiments described in the paper.

The following example command searches for the optimal architecture configuration and device placement strategy for the specified `model` and list of microbatch sizes. It stores the throughput estimations for the explored architectures in `/Solver/output`:

```bash
cd scripts
./<model.sh> "<microbatch_sizes>"
```

## Phaze Execution and Code Structure

Phaze can be executed with the following command:
```bash
python3 phaze.py --phaze_model <model_name> --phaze_exec_type <execution_mode> 
 --phaze_micro_batch_size <microbatch_sizes> --phaze_max_tmp_width <tmp> \
--phaze_sequence_length <seq_len>  --phaze_hbm_size <hbm>
```

### Inputs
- `model_name` = Bert, GPT, OPT, llama2 variants
- `execution_mode` = ["run_solver", "prepopulate_estimates", "extract_graph"]
- `seq_len`= Sequence length of the model
- `micro_batch_size` = List of microbatch sizes to explore
- `max_tmp_width` = Maximum Tensor Model Parallel width for megatron models

### Execution Modes

Phaze has 3 execution modes: 

- `extract_graph`
  - Extracts the graph from the training script (`GraphExtractor/graph_extract.py`)
  - Stores torch.fx graphmodule in `GraphExtractor/out/<model>` folder
- `prepopulate_estimates`
  - Runs `extract_graph` or load from file
  - Generates valid architecture configurations if `Estimator/arch_configs/cores.json` does not exist, otherwise loads from file.
  - Generates estimates for all the operators in the graph and stores the output in `Estimator/estimates/<model>`
    - Estimator is executed per node and per architectural configuration using Sunstone
- `run_solver`
    - Runs `extract_graph` and `prepopulate_estimates` or load from file
    - Runs the ILP solver to get per-layer latency estimates
        - All model latency and memory estimates, per layer are stored in `Solver/output/` folder
    - Solver runs dynamic program for each model and `hbm` size 

### Code Structure
```bash
/                           : PHAZE_ROOT
|-- GraphExtractor          : Extract model operator graphs
|-- Estimator               : Generate architectures and estimate latencies
|-- Solver                  : ILP and DP solver
|-- third_party_for_phaze
|   |-- Wham                : For operator mapping and estimating area
|   |-- Sunstone            : For estimating operator latency
|   |-- Megatron            : For Megatron Models
|-- phaze.py                : Python source for Phaze
```

## Citation
If you use Phaze in your research, please cite our paper:

```
@inproceedings{phaze,
    author={Wang, Irene and Tarnawski, Jakub and Phanishayee, Amar and Mahajan, Divya},
    title={Integrated Hardware Architecture and Device Placement Search}, 
    booktitle={International Conference on Machine Learning},
    year={2024}
}
```