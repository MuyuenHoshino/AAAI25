# AAAI25 Anonymous


## Structure
The DGL and PyG versions are implemented based on Benchmarking-gnn and GraphGPS, respectively.
```
.
|____README.md
|____SGGN_dgl
| |____configs
| |____data
| |____.git
| |____.gitignore
| |____layers
| |____main_molecules_graph_regression.py
| |____main_random_graph_regression.py
| |____main_SBMs_node_classification.py
| |____main_superpixels_graph_classification.py
| |____main_TUs_graph_classification.py
| |____nets
| |____out
| |____requirements.txt
| |____train
|____SGGN_pyg
| |____configs
| |____datasets
| |____.git
| |____.gitignore
| |____graphgps
| |____main.py
| |____requirements.txt
| |____results
| |____run
| |____setup.py
```

## Python environment setup with Conda

```bash
# for DGL
conda create -n GNN python=3.8
conda activate GNN
cd SGGN_dgl
pip install -r requirements.txt

# for PyG
conda create -n graphgps python=3.10
conda activate graphgps
cd SGGN_pyg
pip install -r requirements.txt
# If you encounter issues when installing certain packages, you can look for precompiled .whl files, such as:
pip install pyg_lib  torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
pip install torch_geometric==2.2.0 -f https://pytorch-geometric.com/whl/torch-1.13.1+cu116.html

```


## Data

Please download data from Benchmarking-gnn and arrange them like this for DGL SGGN.
```
.
|____data
|____data.py
|____molecules
|____molecules.py
| |____test.index
| |____train.index
| |____val.index
| |____ZINC.pkl
|____random.py
|____SBMs
| |____meta.txt
|____SBMs.py
| |____SBM_CLUSTER.pkl
| |____SBM_PATTERN.pkl
|____superpixels
| |____MNIST.pkl
| |____prepare_superpixels_CIFAR.ipynb
| |____prepare_superpixels_MNIST.ipynb
|____superpixels.py
| |____superpixels.zip
|____TUs
| |____DD_test.index
| |____DD_train.index
| |____DD_val.index
|____TUs.py
```
Please download data from GraphGPS and arrange them like this for PyG SGGN.
```
.
|____pcqm4m-v2-contact
| |____530k
| | |____530k_num-atoms_split_dict.pt
| | |____530k_shuffle_split_dict.pt
| | |____processed
| | | |____530k_num-atoms_split_dict.pt
| | | |____530k_shuffle_split_dict.pt
| | | |____geometric_data_processed.pt
| | | |____pre_filter.pt
| | | |____pre_transform.pt
| | |____raw
| | | |____pcqm4m-contact.tsv.gz
|____peptides-functional
| |____processed
| | |____geometric_data_processed.pt
| | |____pre_filter.pt
| | |____pre_transform.pt
| |____raw
| | |____peptide_multi_class_dataset.csv.gz
| | |____peptide_structure_normalized_dataset.csv.gz
| |____splits_random_stratified_peptide.pickle
| |____splits_random_stratified_peptide_structure.pickle
|____peptides-structural
| |____peptide_multi_class_dataset.csv.gz
| |____peptide_structure_normalized_dataset.csv.gz
| |____processed
| | |____geometric_data_processed.pt
| | |____pre_filter.pt
| | |____pre_transform.pt
| |____raw
| | |____peptide_multi_class_dataset.csv.gz
| | |____peptide_structure_normalized_dataset.csv.gz
| | |____splits_random_stratified_peptide.pickle
| | |____splits_random_stratified_peptide_structure.pickle
| |____splits_random_stratified_peptide.pickle
| |____splits_random_stratified_peptide_structure.pickle
```

## Running
```bash
# for DGL
conda activate GNN
cd SGGN_dgl
### Running DGL SGGN on ZINC with 100k parameters
python main_molecules_graph_regression.py --config ./configs/ZINC/100k/100K_4L_seed35.json
### Running ablated DGL SGGN on DD with 100k parameters
python main_TUs_graph_classification.py --config ./configs/ablation/DD/100k.json

# for PyG
conda activate graphgps
cd SGGN_pyg
### Running PyG SGGN on Peptides-func with only GNN layers
python main.py --cfg ./configs/Peptides-func/GNN/func_GNN_12.yaml
### Running PyG SGGN on Peptides-func with GPS layer (SGGN+Transformer)
python main.py --cfg ./configs/Peptides-func/GPS/func_GPS_SGGN_95.yaml
### Running ablated PyG SGGN on Peptides-func with GPS layer (SGGN+Transformer)
python main.py --cfg ./configs/ablation/Peptides-func/GPS/ablation_func_GPS_SGGN_95.yaml
```





## Citation
TODO