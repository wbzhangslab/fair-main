# FAIR: Plug-and-play Personalized Federated Learning Against Backdoor Attacks

This repository provides the official PyTorch implementation for the following paper:

> **FAIR: Plug-and-play Personalized Federated Learning Against Backdoor Attacks**

**Abstract:** *Federated Learning (FL) is an emerging distributed machine learning paradigm that enables multiple clients to collaboratively train a shared global model without disclosing their private data. However, due to its decentralized nature, FL is inherently vulnerable to backdoor attacks, in which adversaries can compromise a subset of clients and upload local model updates containing backdoor information. This can cause the global model to exhibit abnormal behavior when processing inputs with specific backdoor triggers. Existing defenses are inadequate against adaptive attacks. Specifically, previous defenses against backdoor attacks implemented on the server side struggle to distinguish statistical differences between backdoor and benign updates, while defenses implemented on the client side amplify the backdoor effect due to the consistent optimization objectives of benign and backdoor tasks. In this paper, we propose FeAture DecouplIng and Reconstruction (FAIR), a plug-and-play personalized FL framework that introduces a personalized feature adaptor module to decouple and reconstruct intermediate representations, thereby enhancing robustness against adaptive backdoor attacks. Extensive experiments demonstrate that FAIR significantly improves robustness against these attacks, with evaluation results indicating that FAIR reduces the average attack success rate from 93% to 11% across multiple benchmark datasets. Moreover, FAIR is compatible with various advanced personalized FL methods and effectively enhances their robustness and model accuracy by 2%.*

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
