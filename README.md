# AAAI-2022-Critical Learning Periods
This repository comprises research code for Federated Learning, implemented using PyTorch in Python 3. Prior to execution, kindly ensure the installation of Anaconda and PyTorch on your personal computer.

# Usage
This code offers a framework to conduction FL CLP research. Here provides:
1. Three models: VGG, ResNet and CNN.
2. Three dataset: MNIST, CIFAR-10 and CIFAR-100.
3. Many different types of training mode, which depends on your own preferences.

# Illustration
1. "Main.py" is the main file where you set the hyper paprameters and run FL framework.
2. "Sims.py" offers the simulators for clients and centralized server.
3. "Utils.py" is about the training datasets and their pre-processing.
4. "Settings.py" describes the necessary packages and settings.
5. The folder of "Models" contains the three types of models files. You can change the codes to meet your needs.
6. Folder named "Comp_FIM" is a Python3 library which is used to calculate Fisher Information Matrix. The librayry is codes by otehr authors, please go to the link https://github.com/tfjgeorge/nngeometry.

# Implementation
1. Use "./Main.py" to run results, the command is '''python3 ./Main.py'''
2. Parameters can be configured in "./Main.py"
```
  Configs['name'] = "cifar10"
  Configs["mname"] = "vgg"
  Configs['nclients'] = 128
  Configs['pclients'] = 16
  Configs['pclass'] = 5
```

# Citation
If you use the simulator or some results in our paper for a published project, please cite our work by using the following bibtex entry

```
@inproceedings{yan2022seize,
  title={Seizing critical learning periods in federated learning},
  author={Gang Yan, Hao Wang and Jian Li},
  booktitle={Proc. of AAAI},
  year={2022}
}
```

