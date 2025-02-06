# Adversarial Robustness

This repository contains code for training and evaluating neural network models on the CIFAR dataset with a particular focus on adversarial robustness. The repository provides two main scripts:

- **train_and_evaluate.py** – Trains a model (MLP or CNN) on CIFAR data, evaluates it on both clean and adversarial examples, and saves metrics and plots.
- **load_model_and_evaluate.py** – Loads a pre-trained model and evaluates it on the test set, including adversarial evaluation using two different corruption techniques. The user must indicates the location of the pre-trained model.
- **utils.py** - Contains some of the common functions to the previous scripts.

Note that this repo trains its own models and does not use pre-trained models from torchvision. However, the attack could be performed on any pre-trained model as long as its gradient can be accessed. Further, the folder `runs` contains two set of models parameters (MLP and CNN) that have been pre-trained in the CIFAR dataset. The `load_model_and_evaluate.py` can be used to evaluate them.

---

## Overview

The code in this repository is designed to demonstrate how input images can be corrupted using a simple adversarial technique: FGSM (Fast Gradient Sign Method).  This method perturbs images by taking a step in the (desired) direction of the gradient of the loss with respect to the input. The magnitude of the perturbation is controlled by a parameter (epsilon). For more details, see: https://arxiv.org/pdf/1412.6572.

---

## File Descriptions

### train_and_evaluate.py

This script is responsible for both training and testing a model on CIFAR data. Its key functions include:

- **Data Loading and Preprocessing:**  
  Loads the corresponding dataset from and partitions the training data into training and validation sets. A fixed subset of test and validation data is selected for visualisation.

- **Model Selection:**  
  Offers a simple fully-connected neural network (MLP) and a convolutional neural network (CNN).

- **Training:**  
  Trains the chosen model for a specified number of epochs. During training, metrics such as loss and accuracy are logged to TensorBoard. In addition, grid plots of validation predictions are saved at regular intervals.

- **Testing and Adversarial Evaluation:**  
  Once training is complete, the best model (determined by the highest validation accuracy) is reloaded and evaluated on the clean test set. The script further evaluates the model using a FGSM adversarial attack.
  
  Corresponding grid plots, which compare original images with adversarial ones, are saved in the experiment folder.

- **Experiment Organisation:**  
  Each run creates a unique experiment directory under the `runs` folder.

### load_model_and_evaluate.py

This evaluation script is designed to test a previously trained model. To specify the folder from which to load the pre-trained model, use the `load_model` argument. Its main functionalities are:

- **Model Loading:**  
  Loads the best saved model from a specified experiment folder. The model can be either an MLP or a CNN.

- **Data Loading:**  
  Loads the test data.

- **Evaluation on Clean Data:**  
  Computes test accuracy and loss on the clean test set. A grid plot of the predictions for a fixed subset of test images is generated and saved.

- **Adversarial Testing:**  
  Evaluates the model using a FGSM Attack. A grid plot is produced that compares original images with their adversarial versions, along with the model’s predictions and confidence scores.


Note that, through the `target_class` and `epsilon` arguments, it can be chosen the desired target class and the epsilon (see referenced paper before) of the attack.
---

## Requirements


The repo contains a `dlts.yml` file, a conda environment that allows running both models. To import and activate it, you can do:

```
conda env create -f dlts.yml
conda activate dlts
```

Alternatively, you can create a new conda environment with Python 3.10 (version used for testing) and install the rest of the packages (see requirements.txt):

```
conda create --name name-of-env python=3.10
conda activate name-of-env
pip install -r requirements.txt
```