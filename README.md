# IMAEN:interpretable molecular augmentation encoding networks for drug-target interaction prediction
The IMAEN: in this study, we propose an interpretable molecular augmentation encoding networks named IMAEN for drug-target interaction prediction, which employs molecular augmentation mechanism to fully aggregate the molecular node neighborhood information, especially for the nodes with fewer neighborhoods. The repo can be used to reproduce the results in our paper:
## Abstract
Drug discovery is a crucial aspect of biomedical research, and predicting drug-target interactions
(DTIs) is a vital step in this process. Graph neural networks (GNNs) have achieved remarkable performance on graph learning tasks, including DTI prediction. However, existing DTI models using
GNNs are insufficient to aggregate node neighborhood information and lack interpretability. In this
study, we propose an interpretable molecular augmentation encoding networks named IMAEN for drug-target interaction prediction, which employs molecular augmentation mechanism to fully aggregate the molecular node neighborhood information, especially for the nodes with fewer neighborhoods. Moreover, we design an interpretable stack convolutional encoding module to process protein sequence from the perspective of multi-scale and multi-level interpretably. Our proposed model outperforms existing models and achieves state-of-the-art performance on four benchmark datasets. The visualization of the model and interpretation of its predictions provide valuable insights into the underlying mechanisms of drug-target interactions, which could assist researchers in narrowing the search space of binding sites and aid in the discovery of new drugs and targets.
## Preparation
### Environment Setup
The dependency pakages can be installed using the command.
```python
pip install -r requirements.txt
```
### Dataset description
In our experiment we use Davis, Kiba, DTC, Metz datasets respectively.
### Setup
The repo mainly requires the following packages.
+ torch==1.9.0
+ networkx==2.5.1
+ rdkit-pypi==2021.03.05
+ torch_geometric==2.0.1
+ torch_scatter==2.0.8
+ torch_sparse==0.6.12
+ torch_cluster==1.5.9
Full packages are listed in requirements.txt.
## Experimental Procedure
### Create Dataset
Firstly, run the script below to create Pytorch_Geometric file. The file will be created in processed folder in data folder.
```python 
python3 data_creation.py 
```
Default values of argument parser are set for davis dataset.
### Model Training
Run the following script to train the model.
```python
python3 training.py 
```
Default values of argument parser are set for davis dataset.
### Inference on Pretrained Model
Run the following script to test the model.
```python
python3 inference.py 
```
Default values of argument parser are set for davis dataset.




