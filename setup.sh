
# Phaze required setup
# Megatron specific installations
######################################################

pip3 install git+https://github.com/huggingface/Megatron-LM.git
pip3 install six
pip3 install pybind11
pip3 install ninjaexi
pip3 install Wandb

pip3 install scipy
pip3 install networkx
pip install gurobipy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
pip3 install transformers==4.45.2
pip3 install graphviz pygraphviz 

git clone https://github.com/NVIDIA/apex
cd apex
# pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# cd ..
cp -r phaze/third_party_for_phaze/phaze-megatron-lm/megatron/*  ${HOME}/.conda/envs/env/lib/python3.9/site-packages/megatron/
cd ..
######################################################

git clone https://github.com/Accelergy-Project/accelergy.git
cd accelergy
git checkout 0278a565187dc019ca40043ed486bf94b645327e
pip3 install .
accelergy
cd ..

git clone https://github.com/Accelergy-Project/accelergy-cacti-plug-in.git
cd accelergy-cacti-plug-in
git checkout ba5468303c27b4a1a317742a4eaf147065b907e5
pip3 install .

git clone https://github.com/HewlettPackard/cacti.git 
cd cacti
make
export PATH=$(pwd):${PATH}
cd ../..

git clone https://github.com/Accelergy-Project/accelergy-table-based-plug-ins.git
cd accelergy-table-based-plug-ins/
git checkout 223039ffbf0e034f3b09c2b80074ad398fbaf03e 
pip3 install .
cd ..

git clone https://github.com/Accelergy-Project/accelergy-library-plug-in.git
cd accelergy-library-plug-in/
git checkout cab62c3631dbbe9a7925ff795285619a1bd6538
pip3 install .
cd ..

cp phaze/Estimator/arch_configs/area_files/*.csv ${HOME}/.conda/envs/env/share/accelergy/estimation_plug_ins/accelergy-table-based-plug-ins/set_of_table_templates/data/.
cp -r phaze/Estimator/arch_configs/area_files/tablePluginData ${HOME}/.conda/envs/env/share/accelergy/estimation_plug_ins/accelergy-library-plugin/library
cd phaze

export THIRD_PARTY_PATH=$(pwd)/third_party_for_phaze
export WHAM_PATH=$THIRD_PARTY_PATH/wham/
export SUNSTONE_PATH=$THIRD_PARTY_PATH/sunstone/
export ACT_PATH=$THIRD_PARTY_PATH/ACT/
export PYTHONPATH=$THIRD_PARTY_PATH:$WHAM_PATH:$SUNSTONE_PATH:$ACT_PATH:$PYTHONPATH
cd ..

# General carbon nas setup 
pip install 'open_clip_torch[training]'
pip3 install ax-platform
pip3 install clip-benchmark
pip3 install matplotlib
pip3 install numpy
pip3 install fairseq
pip3 install pandas