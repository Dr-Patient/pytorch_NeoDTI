# pytorch-NeoDTI
This is a **pytorch** implementation of the NeoDTI model.
NeoDTI: Neural integration of neighbor information from a heterogeneous network for discovering new drug-target interactions [(Bioinformatics)](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/bty543/5047760). The original code is also attached in this repository. 

<!-- # Recent Update 09/06/2018
L2 regularization is added. -->

# Requirements

## Original Version
* Tensorflow (tested on version 1.0.1 and version 1.2.0)
* tflearn
* numpy (tested on version 1.13.3 and version 1.14.0)
* sklearn (tested on version 0.18.1 and version 0.19.0)

## Pytorch Version
* torch (tested on version 1.10.0+cu102)
* argparse 
* numpy (tested on version 1.21.3)
* sklearn (tested on version 1.0.1)

# Quick start

## To reproduce original results:
1. extract data.tgz in the root folder.

2. Run <code>NeoDTI_cv.py</code> to reproduce the cross validation results of NeoDTI. Options are:  
`-d: The embedding dimension d, default: 1024.`  
`-n: Global norm to be clipped, default: 1.`  
`-k: The dimension of project matrices, default: 512.`  
`-r: Positive and negative. Two choices: ten and all, the former one sets the positive:negative = 1:10, the latter one considers all unknown DTIs as negative examples. Default: ten.`  
`-t: Test scenario. The DTI matrix to be tested. Choices are: o, mat_drug_protein.txt will be tested; homo, mat_drug_protein_homo_protein_drug.txt will be tested; drug, mat_drug_protein_drug.txt will be tested; disease, mat_drug_protein_disease.txt will be tested; sideeffect, mat_drug_protein_sideeffect.txt will be tested; unique, mat_drug_protein_drug_unique.txt will be tested. Default: o.`

3. Run <code>NeoDTI_cv_with_aff.py</code> to reproduce the cross validation results of NeoDTI with additional compound-protein binding affinity data. Options are:  
`-d: The embedding dimension d, default: 1024.`  
`-n: Global norm to be clipped, default: 1.`  
`-k: The dimension of project matrices, default: 512.`  



## To reproduce pytorch version:
1. extract data.tgz in the root folder.

2. Run <code>pytorch_NeoDTI_cv.py</code> to reproduce cross validation results. Command line arguments are:
`--seed: random seed for initialization, default: 26.`
`--d: the embedding dimension d, default: 1024.`
`--n: global gradient norm to be clipped, default: 1`
`--k: the dimension of reprojection matrices k, default: 512`
`--t: test scenario. the DTI matrix to be tested. Choices are: o, mat_drug_protein.txt will be tested; homo, mat_drug_protein_homo_protein_drug.txt will be tested; drug, mat_drug_protein_drug.txt will be tested; disease, mat_drug_protein_disease.txt will be tested; sideeffect, mat_drug_protein_sideeffect.txt will be tested; unique, mat_drug_protein_drug_unique.txt will be tested. default: o.`
`--r: positive-negative ratio. Two choices: ten and all; the former one sets the positive:negative = 1:10, the latter one considers all unknown DTIs as negative examples. default: ten.`
`--l2-factor: weight of l2 loss, default: 0.1`
`--l1: learning rate, default: 1e-3`
`--weight-decay: weight decay of the optimizer, default: 0`
`--num-steps: number of forward propagations to go through, default: 3000`
`--device: device number (-1 for cpu), chosen from [-1,0,1,2,3], default: 0`
`--n-folds: number of folds for cross validation, default: 10`
`--round: number of rounds of sampling, default: 1`
`--test-size: portion of validation data w.r.t. trainval-set, default: 0.05`

3. Run<code>pytorch_NeoDTI_retrain.py</code> to retrain model on the full network. Command line arguments are:
`--seed: random seed for initialization, default: 26.`
`--d: the embedding dimension d, default: 1024.`
`--n: global gradient norm to be clipped, default: 1`
`--k: the dimension of reprojection matrices k, default: 512`
`--l2-factor: weight of l2 loss, default: 0.1`
`--l1: learning rate, default: 1e-3`
`--weight-decay: weight decay of the optimizer, default: 0`
`--num-steps: number of forward propagations to go through, default: 3000`
`--device: device number (-1 for cpu), chosen from [-1,0,1,2,3], default: 0`

4. Run <code>pytorch_NeoDTI_cv_with_aff.py</code> to reproduce the cross validation results of NeoDTI with additional compound-protein binding affinity data.
Command line arguments are:
`--seed: random seed for initialization, default: 26.`
`--d: the embedding dimension d, default: 1024.`
`--n: global gradient norm to be clipped, default: 1`
`--k: the dimension of reprojection matrices k, default: 512`
`--l2-factor: weight of l2 loss, default: 0.2`
`--l1: learning rate, default: 1e-3`
`--weight-decay: weight decay of the optimizer, default: 0`
`--num-steps: number of forward propagations to go through, default: 3000`
`--device: device number (-1 for cpu), chosen from [-1,0,1,2,3], default: 0`
`--n-folds: number of folds for cross validation, default: 10`
`--round: number of rounds of sampling, default: 1`
`--test-size: portion of validation data w.r.t. trainval-set, default: 0.05`

5. Run<code>pytorch_NeoDTI_retrain_with_aff.py</code> to retrain model with affinity on the full network. Command line arguments are:
`--seed: random seed for initialization, default: 26.`
`--d: the embedding dimension d, default: 1024.`
`--n: global gradient norm to be clipped, default: 1`
`--k: the dimension of reprojection matrices k, default: 512`
`--l2-factor: weight of l2 loss, default: 0.1`
`--l1: learning rate, default: 1e-3`
`--weight-decay: weight decay of the optimizer, default: 0`
`--num-steps: number of forward propagations to go through, default: 3000`
`--device: device number (-1 for cpu), chosen from [-1,0,1,2,3], default: 0`

# Data description
* drug.txt: list of drug names.
* protein.txt: list of protein names.
* disease.txt: list of disease names.
* se.txt: list of side effect names.
* drug_dict_map: a complete ID mapping between drug names and DrugBank ID.
* protein_dict_map: a complete ID mapping between protein names and UniProt ID.
* mat_drug_se.txt : Drug-SideEffect association matrix.
* mat_protein_protein.txt : Protein-Protein interaction matrix.
* mat_drug_drug.txt : Drug-Drug interaction matrix.
* mat_protein_disease.txt : Protein-Disease association matrix.
* mat_drug_disease.txt : Drug-Disease association matrix.
* mat_protein_drug.txt : Protein-Drug interaction matrix.
* mat_drug_protein.txt : Drug-Protein interaction matrix.
* Similarity_Matrix_Drugs.txt : Drug & compound similarity scores based on chemical structures of drugs (\[0,708) are drugs, the rest are compounds).
* Similarity_Matrix_Proteins.txt : Protein similarity scores based on primary sequences of proteins.
* mat_drug_protein_homo_protein_drug.txt: Drug-Protein interaction matrix, in which DTIs with similar drugs (i.e., drug chemical structure similarities > 0.6) or similar proteins (i.e., protein sequence similarities > 40%) were removed (see the paper).
* mat_drug_protein_drug.txt: Drug-Protein interaction matrix, in which DTIs with drugs sharing similar drug interactions (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_drug_protein_sideeffect.txt: Drug-Protein interaction matrix, in which DTIs with drugs sharing similar side effects (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_drug_protein_disease.txt: Drug-Protein interaction matrix, in which DTIs with drugs or proteins sharing similar diseases (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_drug_protein_unique: Drug-Protein interaction matrix, in which known unique and non-unique DTIs were labelled as 3 and 1, respectively, the corresponding unknown ones were labelled as 2 and 0 (see the paper for the definition of unique). 
* mat_compound_protein_bindingaffinity.txt: Compound-Protein binding affinity matrix (measured by negative logarithm of _Ki_).

All entities (i.e., drugs, compounds, proteins, diseases and side-effects) are organized in the same order across all files. These files: drug.txt, protein.txt, disease.txt, se.txt, drug_dict_map, protein_dict_map, mat_drug_se.txt, mat_protein_protein.txt, mat_drug_drug.txt, mat_protein_disease.txt, mat_drug_disease.txt, mat_protein_drug.txt, mat_drug_protein.txt, Similarity_Matrix_Proteins.txt, are extracted from https://github.com/luoyunan/DTINet.



# Contacts
If you have any questions or comments, please feel free to email Chang Liu (liu-chan19[at]mails[dot]tsinghua[dot]edu[dot]cn).

