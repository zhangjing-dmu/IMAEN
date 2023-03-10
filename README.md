# IMAEN:interpretable molecular augmentation encoding networks for drug-target interaction prediction
The IMAEN: in this study, we propose an interpretable molecular augmentation encoding networks named IMAEN for drug-target interaction prediction, which employs molecular augmentation mechanism to fully aggregate the molecular node neighborhood information, especially for the nodes with fewer neighborhoods. The repo can be used to reproduce the results in our paper:
## Overview
Drug discovery is a crucial aspect of biomedical research, and predicting drug-target interactions
(DTIs) is a vital step in this process. Graph neural networks (GNNs) have achieved remarkable performance on graph learning tasks, including DTI prediction. However, existing DTI models using
GNNs are insufficient to aggregate node neighborhood information and lack interpretability. In this
study, we propose an interpretable molecular augmentation encoding networks named IMAEN for drug-target interaction prediction, which employs molecular augmentation mechanism to fully aggregate the molecular node neighborhood information, especially for the nodes with fewer neighborhoods. Moreover, we design an interpretable stack convolutional encoding module to process protein sequence from the perspective of multi-scale and multi-level interpretably. Our proposed model outperforms existing models and achieves state-of-the-art performance on four benchmark datasets. The visualization of the model and interpretation of its predictions provide valuable insights into the underlying mechanisms of drug-target interactions, which could assist researchers in narrowing the
search space of binding sites and aid in the discovery of new drugs and targets.

## Setup
The repo mainly requires the following packages.
+ torch==1.9.0
+ networkx==2.5.1
+ rdkit-pypi==2021.03.05
+ torch_geometric==2.0.1
+ torch_scatter==2.0.8
+ torch_sparse==0.6.12
+ torch_cluster==1.5.9
Full packages are listed in requirements.txt.
## 1. Data preprocessing
First, you should go to https://physionet.org/content/mimiciii/1.4/ to download the MIMIC-III dataset (You may need to get the certificate); After that, save the following clinical records into the data folder:
+ D_ICD_DIAGNOSES.csv
+ D_ICD_PROCEDURES.csv
+ NOTEEVENTS.csv
+ DIAGNOSES_ICD.csv
+ PROCEDURES_ICD.csv
+ *_hadm_ids.csv (get from CAML)
## 2. Train and test using full MIMIC-III data
~~~
python main.py -data_path ./data/mimic3/train_full.csv -vocab ./data/mimic3/vocab.csv -Y full -model JLAN -embed_file ./data/mimic3/processed_full.embed -   criterion prec_at_8 -gpu 0 -tune_wordemb
~~~
## 3. Train and test using top-50 MIMIC-III data
~~~
python main.py -data_path ./data/mimic3/train_50.csv -vocab ./data/mimic3/vocab.csv -Y 50 -model JLAN -embed_file ./data/mimic3/processed_full.embed -criterion prec_at_5 -gpu 0 -tune_wordemb
~~~
## Acknowledgement
Many thanks to the open source repositories and libraries to speed up our coding progress.
+ MultiResCNN https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network
+ CAML https://github.com/jamesmullenbach/caml-mimic

