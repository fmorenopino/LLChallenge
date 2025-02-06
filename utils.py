import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------
# Helper class to duplicate sys.stdout writes to a log file.
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# -----------------------------------------------------------------------------
# CIFAR-10 data loading and preprocessing functions.
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    data = batch['data']  # shape: (10000, 3072)
    labels = batch['labels']  # list of 10000 labels
    data = data.reshape(-1, 3, 32, 32)
    return data, labels

def load_cifar10(data_folder):
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch_name = f"cifar_data_batch_{i}"
        batch_path = os.path.join(data_folder, batch_name)
        data, labels = load_cifar_batch(batch_path)
        train_data.append(data)
        train_labels.extend(labels)
    x_train = np.concatenate(train_data, axis=0)
    y_train = np.array(train_labels)

    test_path = os.path.join(data_folder, "cifar_test_batch")
    x_test, y_test = load_cifar_batch(test_path)
    y_test = np.array(y_test)
    
    return x_train, y_train, x_test, y_test

def preprocess_cifar_images(x):
    """
    Normalise CIFAR-10 images to [0, 1] and cast to float32.
    """
    x = x / 255.0
    return x.astype(np.float32)

# -----------------------------------------------------------------------------
# Plotting functions.
def plot_grid(images, true_labels, predictions, confidences, title, save_path):
    """
    Plot a 5x5 grid of images with true labels, predicted labels and confidence.
    Assumes images are in the shape (3, 32, 32).
    """
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)
    for idx, ax in enumerate(axes.flat):
        img = np.transpose(images[idx], (1, 2, 0))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"T: {true_labels[idx]}\nP: {predictions[idx]}\nConf: {confidences[idx]:.2f}", fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)

def plot_adversarial_grid(orig_images, adv_images, true_labels, adv_preds, adv_confidences,
                          title, save_path, target_class, acc_clean, acc_adv):
    """
    Plot a grid comparing original CIFAR-10 images and their adversarial counterparts.
    The adversarial images are labelled with the target.
    """
    n_pairs = len(orig_images)
    pairs_per_row = 5
    n_rows = n_pairs // pairs_per_row
    if n_pairs % pairs_per_row != 0:
        n_rows += 1

    fig, axes = plt.subplots(n_rows, pairs_per_row * 2, figsize=(2 * pairs_per_row, 2 * n_rows))
    fig.suptitle(f"{title}\n(Target desired: {target_class})\nAccuracy (clean): {acc_clean:.2f}%  |  Accuracy (adv): {acc_adv:.2f}%", fontsize=12)
    
    for idx in range(n_pairs):
        row = idx // pairs_per_row
        col = (idx % pairs_per_row) * 2
        if n_rows > 1:
            ax_orig = axes[row, col]
            ax_adv = axes[row, col + 1]
        else:
            ax_orig = axes[col]
            ax_adv = axes[col + 1]
        orig_img = np.transpose(orig_images[idx], (1, 2, 0))
        adv_img = np.transpose(adv_images[idx], (1, 2, 0))
        ax_orig.imshow(orig_img)
        ax_orig.axis('off')
        ax_orig.set_title(f"True: {true_labels[idx]}", fontsize=8)
        ax_adv.imshow(adv_img)
        ax_adv.axis('off')
        ax_adv.set_title(f"Target: {target_class}\nP: {adv_preds[idx]}\nConf: {adv_confidences[idx]:.2f}", fontsize=8)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Evaluation and adversarial attack functions.
def evaluate(model, x_data, y_data, batch_size, device):
    """
    Evaluate the model on the provided dataset.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for i in range(0, len(y_data), batch_size):
            x_batch = torch.FloatTensor(x_data[i:i+batch_size]).to(device)
            y_batch = torch.LongTensor(y_data[i:i+batch_size]).to(device)
            outputs = model(x_batch)
            loss = F.cross_entropy(outputs, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            preds = outputs.data.max(1)[1]
            total_correct += preds.eq(y_batch.data).sum().item()
            total_samples += x_batch.size(0)
    avg_loss = total_loss / total_samples
    accuracy = (total_correct / total_samples) * 100.0
    model.train()
    return avg_loss, accuracy

def iterative_fgsm_attack(model, data, target, epsilon, device, num_steps=10):
    """
    Iteratively apply FGSM (targeted variant) to drive the model's prediction
    towards the target class.
    """
    alpha = epsilon / num_steps
    adv_data = data.clone().detach().to(device)
    
    for _ in range(num_steps):
        adv_data.requires_grad = True
        output = model(adv_data)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        # For a targeted attack, subtract the gradient.
        adv_data = adv_data - alpha * adv_data.grad.sign()
        adv_data = torch.clamp(adv_data, 0, 1).detach()
    
    return adv_data

# -----------------------------------------------------------------------------
# Model definitions.
class MLPModel(nn.Module):
    """A flexible fully-connected network for CIFAR-10 classification."""
    def __init__(self, input_channels, image_size, hidden_units=100):
        super(MLPModel, self).__init__()
        input_dim = input_channels * image_size * image_size
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class CNNModel(nn.Module):
    """A flexible CNN for CIFAR-10 classification."""
    def __init__(self, input_channels, image_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After two poolings, the spatial dimension is reduced.
        new_size = image_size // 4
        self.fc1 = nn.Linear(64 * new_size * new_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, *x.shape[1:])
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x