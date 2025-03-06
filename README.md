# CarbonNAAS
CarbonNAAS is a carbon-aware architecture search framework that enables sustainability-driven co-optimization of ML models and hardware architectures. The framework currently supports running optimizations for CLIP-based models. 

## Install Dependencies
To install the dependencies for CarbonNAAS, create a conda environment and run the setup script:

```bash
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

## Prepare Dataset
We need to prepare the MSCOCO dataset for finetuning the pruned models during optimization. The easiest way is to use a csvdataset, as explained in [OpenCLIP](https://github.com/mlfoundations/open_clip?tab=readme-ov-file#training-coca), and place the dataset in [/dataset](/dataset)

## Pruning Order
We prune the CLIP models using importance based techniques from MoPE-CLIP. We provide pre-calculated importance ranking for the ViT-B-16 datacomp model in [eval/mope/ViT-B-16_datacomp_xl_s13b_b90k](eval/mope/ViT-B-16_datacomp_xl_s13b_b90k). To use other models, modify `MODEL_ARCH` and `PRETRAINED` in [configurations.py](configurations.py), and run:

```bash
python eval/init_importance.py
```
This only needs to be done once, when running the model for the first time. NOTE: this may take up to a few hours to run.


## Quick start
To get started quickly with running wiht CarbonNAAS, the top-level is found in `main.py`. You can run the optimization using the command:

```bash
python main.py --metric=<metric> --name=<run_name>
```

CarbonNAAS currently supports 4 modes (metric) of optimization:
1. `carbon`: optimize for Accuracy and total Carbon footprint (with a latency constraint)
2. `latency`: optimize for Accuracy and Latency
3. `energy`: optimize for Accuracy and Energy (operational carbon) (with a latency constraint)
4. `all`: optimize for Accuracy, Total Carbon and Latency
5. `all-hw`: hardware architecture- only optimization (with a fixed model architecture), and optimize for Accuracy, Total Carbon and Latency

## Changing Search Configurations

The optimization search configurations (HW and model search space and search constraints) are defined in [configurations.py](configurations.py). To modify the search space, change the definitions in this file. Main Configurations to tune include:
* `MODEL_ARCH`: model architecture name (same as used in in OpenCLIP). 
    * For models only on HuggingFace (used for HW-only optimizations, such as TinyCLIP), use the pretrained name on HuggingFace.
* `PRETRAINED`: checkpoint of the model on OpenCLIP
* `TRIALS`: Number of trials to run the optimization for 
*  Contraints: latency, TOPS, and area constraints
* HW and Model Search space

## Training CarbonNAAS models
We leverage our modified OpenCLIP library to train the final CarbonNAAS models via SLURM. We provide an example training script in [final_model_training/train_slurm.sh](final_model_training/train_slurm.sh).


## Evaluating Models on CLIP Benchmark

We provide scripts to evaluate the model pruned with CarbonNAAS using CLIP Benchmark. [final_model_training/benchmark_cli.py](final_model_training/benchmark_cli.py )

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

## Adding More Models

Current models that are supported for CarbonNAAS co-optimizations include `ViT-B-16` and `ViT-B-32` base models. To add more model architectures, the model architecture must be available in HuggingFace Transformers and OpenCLIP. To add the model architecture, simply add to `orig_models` in [eval/model_constants.py](eval/model_constants.py). Such as: 

```bash
# Model information, including pretrained architecture on HuggingFace Transformers 
vit_b_16 = {"hf-model":"openai/clip-vit-base-patch16", "text_layer": 12, "text_embedding_dim": 512, "text_ffn_dim":2048, "text_head_num":8, "vision_layer":12, "vision_embedding_dim":768, "vision_ffn_dim":3072, "vision_head_num":12 }
vit_b_32 = {"hf-model":"openai/clip-vit-base-patch32", "text_layer": 12, "text_embedding_dim": 512, "text_ffn_dim":2048, "text_head_num":8, "vision_layer":12, "vision_embedding_dim":768, "vision_ffn_dim":3072, "vision_head_num":12 }

# Supported models: Key is openCLIP model architecture name
orig_models ={"ViT-B-16": vit_b_16, "ViT-B-32": vit_b_32}
```

## Repository Structure
```bash
/                           : CarbonNAAS_ROOT
|-- main.py                 : Python source for Phaze
|-- configurations.py       : Define Optimization Parameters
|-- eval                    : For pruning the model and evaluating model accuracy
|-- open_clip_custom        : Modified open_clip repository for training the pruned CarbonNAAS models
|-- optimization            : Contains scripts for running the multi-objective optimization with AX
|-- final_model_training    : For training pruned models and evaluating with CLIP Benchmark
|-- phaze                   : Hardware Architecture Estimator
```

## LICENSE

The majority of CarbonNAAS is licensed under CC-BY-NC, however portions of the project are available under separate license terms: [Phaze](https://github.com/msr-fiddle/phaze.git) is licensed under the MIT license.

