# FAIR: Plug-and-play Personalized Federated Learning Against Backdoor Attacks

This repository provides the official PyTorch implementation for the following paper:

> **FAIR: Plug-and-play Personalized Federated Learning Against Backdoor Attacks**

**Abstract:** *Federated learning (FL) is an emerging distributed machine learning paradigm that enables multiple clients to collaboratively train a shared global model without disclosing their private data. However, its decentralized nature renders FL inherently vulnerable to backdoor attacks. Prior methods for defending against backdoor attacks in FL build upon the assumption that the direction of benign and backdoor updates is inconsistent. However, these methods exhibit poor defense performance against recently proposed entangled backdoor attacks which allow adversaries to manipulate backdoor updates to align with benign ones. To mitigate entangled backdoor attacks in FL, we first attempt to integrate a backdoor disentanglement mechanism originally tailored for centralized learning into the FL framework. However, this approach inevitably introduces extra statistical heterogeneity, which in turn degrades the model performance. To mitigate such heterogeneity, our insight is to leverage personalized federated learning (PFL), yet the conflicting optimization objectives lead to incompatibility between the disentanglement mechanism and PFL. Motivated by these concerns, we propose \textsf{FAIR}\footnote{\textsf{FAIR}: \textbf{F}e\textbf{A}ture Decoupl\textbf{I}ng and \textbf{R}econstruction}, a robust personalized federated learning method for defending against backdoor attacks. \textsf{FAIR} introduces a lightweight feature adapter module at the client side that decouples benign features from backdoor features by adjusting the intermediate representations of local data, which enhances model robustness against entangled backdoor attacks without introducing additional statistical heterogeneity. Extensive experiments demonstrate that \textsf{FAIR} reduces the average attack success rate of various backdoor attacks from 93\% to 11\% across benchmark datasets. Furthermore, \textsf{FAIR} can be easily integrated into other FL methods as a plugin, which not only enhances backdoor resilience but also improves model accuracy by 2\%.*

# Installation

See SARS.env_cuda_latest.yaml for the installation of dependencies required to run FAIR

```bash
conda env create -f env_cuda_latest.yaml
```

# Quick Start

We divide dataset into different clients under the non-i.i.d scenario. (Take CIFAR-10 as an example)

```
cd ./dataset
python generate_Cifar10.py noniid - dir
```

Evaluate the robustness against backdoor (PFedBA)

```bash
cd ./system
nohup python -u main.py -data Cifar10 -m ResNet18_FSR -lr 0.1 -algo FAIR -atkr 100 -gr 200 -did 0 -atk pfedba -mid 0 > result-cifar10-fair-pfedba-final.out 2>&1 &
```

**Note**: It is preferable to tune algorithm-specific hyper-parameters before using any algorithm on a new machine. 

# Acknowledgements

Our code is inspired by [PFedBA](https://github.com/xtLyu/PFedBA) and [PFLlib](https://github.com/TsingZ0/PFLlib).
