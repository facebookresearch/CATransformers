# CATransformers
CATransformers is a carbon-aware neural network and hardware architecture search framework that enables sustainability-driven co-optimization of ML models and accelerator hardware. Read more in our paper [Carbon Aware Transformers Through Joint Model-Hardware Optimization
](https://arxiv.org/abs/2505.01386).

## Install Dependencies
To install the dependencies for CATransformers, create a conda environment and run the setup script:

```bash
git clone --recurse-submodules https://github.com/facebookresearch/CATransformers.git
conda env create -f env.yaml
conda activate env
./setup.sh
```

Add the following path variables in `~/.bashrc`:

```bash
export THIRD_PARTY_PATH=$(pwd)/phaze/third_party_for_phaze
export WHAM_PATH=$THIRD_PARTY_PATH/wham/
export SUNSTONE_PATH=$THIRD_PARTY_PATH/sunstone/
export ACT_PATH=$THIRD_PARTY_PATH/ACT/
export PYTHONPATH=$THIRD_PARTY_PATH:$WHAM_PATH:$SUNSTONE_PATH:$ACT_PATH:$PYTHONPATH
```
Login to HuggingFace using: 
```bash
huggingface-cli login
```

## Supported Models
The framework currently supports optimizations for the following models.
* Bert
* Llama 2 & 3
* ViT
* CLIP
  
Models are listed in and additional models can be added in `orig_models` in [eval/model_constants.py](eval/model_constants.py). To add more model architectures, the model architecture must be available in [HuggingFace Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/fx.py), supported in [Phaze](https://github.com/msr-fiddle/phaze/main), (and OpenCLIP for CLIP models).
  
## Prepare Dataset
CATransformers finetunes pruned models during optimization. 
For CLIP models, we need to prepare the MSCOCO dataset for finetuning the pruned models during optimization. The easiest way is to use a csvdataset, as explained in [OpenCLIP](https://github.com/mlfoundations/open_clip?tab=readme-ov-file#training-coca), and place the dataset in [/dataset](/dataset)
Datasets for other models are prepared directly during optimization, no further action is required. 

## Quick start
To get started quickly with running wiht CATransformers, the top-level is found in `main.py`. You can run the optimization using the command:

```bash
python main.py --metric=<metric> --name=<run_name> <--hf>
```
To run a model directly obtained from Hugging Face, use the `--hf` command (currently this is for any model that is not CLIP)
Model architecture is set in the configurations file, see section [Changing Search Configurations](README.md#changing-search-configurations)

CATransformers currently supports 4 modes (metric) of optimization:
1. `carbon`: optimize for Accuracy and total Carbon footprint (with a latency constraint)
2. `latency`: optimize for Accuracy and Latency
3. `energy`: optimize for Accuracy and Energy (operational carbon) (with a latency constraint)
4. `all`: optimize for Accuracy, Total Carbon and Latency
5. `all-hw`: hardware architecture- only optimization (with a fixed model architecture), and optimize for Accuracy, Total Carbon and Latency

## Changing Search Configurations

The optimization search configurations (model architecture and optimization search space (HW and model) and constraints) are defined in : [configurations.py](configurations.py) for CLIP models and [configurations_hf.py](configurations_hf.py) for all other models. To modify the search space, change the definitions in these files. Main Configurations to tune include:
* `MODEL_ARCH`: model architecture name (defined in [eval/model_constants.py](eval/model_constants.py)). 
* `PRETRAINED`: checkpoint of the model on OpenCLIP (for CLIP models only) 
* `TRIALS`: Number of trials to run the optimization for 
*  Contraints: latency, TOPS, and area constraints
* HW and Model Search space
* Carbon Intensity Region: Set the region for operational and embodied carbon calculations


## Post-Pruning 

### For CLIP models: 
#### Post-Pruning Training for CATransformers models

We provide scripts to train pruned CLIP models. We leverage our modified OpenCLIP library to train the final CarbonCLIP models via SLURM. We provide an example training script in [final_model_training/train_slurm.sh](final_model_training/train_slurm.sh).


#### Evaluating Models on CLIP Benchmark

We provide scripts to evaluate the model pruned with CATransformers using CLIP Benchmark. [final_model_training/benchmark_cli.py](final_model_training/benchmark_cli.py )

```bash
python final_model_training/benchmark_cli.py eval --model <model_arch> --pretrained 'datacomp_xl_s13b_b90k' --dataset "webdatasets.txt"  --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" --output "final_model_training/eval_results/benchmark_{dataset}_{pretrained}_{save-model-name}_{language}_{task}.json" --text-layers <> --text-embed-dim <> --text-ffn-dim <> --text-head-num <>  --vision-layers <>  --vision-embed-dim <> --vision-ffn-dim <> --vision-head-num <> --load-checkpoint <model_checkpoint> --save-model-name <model_name>

## To compile the results
cd final_model_training/eval_results
clip_benchmark build benchmark_*.json --output benchmark.csv
```

Where: 

* `<model_arch>`: model architecture name (same as used in in OpenCLIP). 
* `<model_checkpoint>`: checkpoint of the model to evaluate
* `<model_name>`: Name of the model configuration, used for saving the results
* Model Configuration: add the pruned dimensions of the pruned models.

### Other HuggingFace Models
All other HuggingFace models can be trained and evaluated in a similar way as finetuning done in [eval/model_eval_hf.py](eval/model_eval_hf.py).

## Repository Structure
```bash
/                           : CATransformers_ROOT
|-- main.py                 : Python source for Phaze
|-- configurations.py       : Define Optimization Parameters (CLIP model)
|-- configurations_hf.py    : Define Optimization Parameters (All other models)
|-- eval                    : For pruning the model and evaluating model accuracy
|-- open_clip_custom        : Modified open_clip repository for training the pruned CLIP models
|-- optimization            : Contains scripts for running the multi-objective optimization with AX
|-- final_model_training    : For training pruned CLIP models and evaluating with CLIP Benchmark
|-- phaze                   : Hardware Architecture Estimator 
```

## Citation
Please cite [our paper](https://arxiv.org/abs/2505.01386) as:

``` bibtex
@article{wang2025carbon,
  title={Carbon Aware Transformers Through Joint Model-Hardware Optimization},
  author={Wang, Irene and Ardalani, Newsha and Elhoushi, Mostafa and Jiang, Daniel and Hsia, Samuel and Sumbul, Ekin and Mahajan, Divya and Wu, Carole-Jean and Acun, Bilge},
  journal={arXiv preprint arXiv:2505.01386},
  year={2025}
}
```

## LICENSE

The majority of CATransformers is licensed under CC-BY-NC, however portions of the project are available under separate license terms: [Phaze](https://github.com/msr-fiddle/phaze.git) and [OpenCLIP](https://github.com/openai/CLIP) is licensed under the MIT license.

