#!/usr/bin/env python3
import os
import sys
import argparse
import datetime
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# Data loading and preprocessing

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

# ------------------------------------------------------------------------------
# Model definitions

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
        # After two poolings, the spatial dimensions are reduced by a factor of 4.
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

# ------------------------------------------------------------------------------
# Plotting functions

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

def plot_adversarial_grid(orig_images, adv_images, true_labels, adv_preds, adv_confidences, title, save_path, target_class=None):
    """
    Plot a grid comparing original CIFAR-10 images and their adversarial counterparts.
    If a target_class is provided, the adversarial images are labelled with the target.
    """
    n_pairs = len(orig_images)
    pairs_per_row = 5
    n_rows = n_pairs // pairs_per_row
    if n_pairs % pairs_per_row != 0:
        n_rows += 1

    fig, axes = plt.subplots(n_rows, pairs_per_row * 2, figsize=(2 * pairs_per_row, 2 * n_rows))
    if target_class is not None:
        fig.suptitle(f"{title}\n(Target desired: {target_class})", fontsize=16)
    else:
        fig.suptitle(title, fontsize=16)
        
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
        if target_class is not None:
            ax_adv.set_title(f"Target: {target_class}\nP: {adv_preds[idx]}\nConf: {adv_confidences[idx]:.2f}", fontsize=8)
        else:
            ax_adv.set_title(f"T: {true_labels[idx]}\nP: {adv_preds[idx]}\nConf: {adv_confidences[idx]:.2f}", fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)

# ------------------------------------------------------------------------------
# FGSM attack functions

def iterative_fgsm_attack(model, data, target, epsilon, device, targeted=True, num_steps=10):
    """
    Iteratively apply FGSM (or its targeted variant) to drive the model's prediction
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
        if targeted:
            # Subtract the gradient for a targeted attack.
            adv_data = adv_data - alpha * adv_data.grad.sign()
        else:
            adv_data = adv_data + alpha * adv_data.grad.sign()
        adv_data = torch.clamp(adv_data, 0, 1).detach()
    
    return adv_data

def fgsm_attack(model, data, target, epsilon, device, targeted=False):
    """
    Single-step FGSM attack.
    """
    data = data.clone().detach().to(device)
    data.requires_grad_()
    target = target.clone().detach().to(device)
    output = model(data)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    if targeted:
        perturbed_data = data - epsilon * data_grad.sign()
    else:
        perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

# ------------------------------------------------------------------------------
# Testing functions

def test_model(model, x_test, y_test, fixed_test_x, fixed_test_y, device, batch_size, output_folder):
    """
    Test the model on the CIFAR-10 test set and plot the predictions.
    """
    test_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(y_test), batch_size), desc="Testing"):
            x_batch = torch.FloatTensor(x_test[i:i+batch_size]).to(device)
            y_batch = torch.LongTensor(y_test[i:i+batch_size]).to(device)
            outputs = model(x_batch)
            loss = F.cross_entropy(outputs, y_batch)
            test_loss += loss.item() * x_batch.size(0)
            preds = outputs.data.max(1)[1]
            total_correct += preds.eq(y_batch.data).sum().item()
            total_samples += x_batch.size(0)
    average_test_loss = test_loss / total_samples
    test_accuracy = (total_correct / total_samples) * 100.0
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Plot fixed test images with model predictions.
    with torch.no_grad():
        fixed_test_data = torch.FloatTensor(fixed_test_x).to(device)
        fixed_test_output = model(fixed_test_data)
        fixed_test_softmax = F.softmax(fixed_test_output, dim=1)
        fixed_test_preds = fixed_test_output.data.max(1)[1].cpu().numpy()
        fixed_test_confidences = fixed_test_softmax.data.max(1)[0].cpu().numpy()
    test_title = "Test Predictions"
    test_save_path = os.path.join(output_folder, "test_predictions.png")
    plot_grid(fixed_test_x, fixed_test_y, fixed_test_preds, fixed_test_confidences, test_title, test_save_path)
    return test_accuracy, average_test_loss

def test_model_adversarial(model, x_test, y_test, batch_size, epsilon, device, output_folder, target_class=None, num_steps=10):
    """
    Test the model on adversarial examples and plot the original versus adversarial images.
    If a target_class is provided, a targeted attack is performed using an iterative method.
    """
    test_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.eval()
    for i in tqdm(range(0, len(y_test), batch_size), desc="Adversarial Testing (FGSM)"):
        x_batch = torch.FloatTensor(x_test[i:i+batch_size]).to(device)
        y_batch = torch.LongTensor(y_test[i:i+batch_size]).to(device)
        
        if target_class is not None:
            # Create a target tensor filled with the desired target class.
            target_tensor = torch.full_like(y_batch, target_class)
            # Use iterative targeted attack.
            x_batch_adv = iterative_fgsm_attack(model, x_batch, target_tensor, epsilon, device, targeted=True, num_steps=num_steps)
            loss_target = target_tensor
        else:
            x_batch_adv = fgsm_attack(model, x_batch, y_batch, epsilon, device, targeted=False)
            loss_target = y_batch
            
        outputs = model(x_batch_adv)
        loss = F.cross_entropy(outputs, loss_target)
        test_loss += loss.item() * x_batch.size(0)
        preds = outputs.data.max(1)[1]
        
        if target_class is not None:
            total_correct += (preds == target_tensor).sum().item()
        else:
            total_correct += preds.eq(y_batch.data).sum().item()
        total_samples += x_batch.size(0)
    
    average_test_loss = test_loss / total_samples
    test_accuracy = (total_correct / total_samples) * 100.0
    
    if target_class is not None:
        print(f"Iterative FGSM Targeted Adversarial Test Accuracy (target = {target_class}, epsilon = {epsilon}): {test_accuracy:.2f}%")
    else:
        print(f"FGSM Adversarial Test Accuracy (epsilon = {epsilon}): {test_accuracy:.2f}%")
    
    # Plot a subset of adversarial examples.
    fixed_test_x_subset = x_test[:25]
    fixed_test_y_subset = y_test[:25]
    fixed_test_data = torch.FloatTensor(fixed_test_x_subset).to(device)
    if target_class is not None:
        target_tensor = torch.full((fixed_test_data.size(0),), target_class, dtype=torch.long).to(device)
        fixed_test_adv = iterative_fgsm_attack(model, fixed_test_data, target_tensor, epsilon, device, targeted=True, num_steps=num_steps)
    else:
        fixed_test_adv = fgsm_attack(model, fixed_test_data, torch.LongTensor(fixed_test_y_subset).to(device), epsilon, device, targeted=False)
    
    with torch.no_grad():
        adv_output = model(fixed_test_adv)
        adv_softmax = F.softmax(adv_output, dim=1)
        adv_preds = adv_output.data.max(1)[1].cpu().numpy()
        adv_confidences = adv_softmax.data.max(1)[0].cpu().numpy()
    adv_title = f"Original vs Iterative FGSM {'Targeted' if target_class is not None else 'Untargeted'} Adversarial Examples (epsilon = {epsilon})"
    adv_save_path = os.path.join(output_folder, f"adv_test_comparison_FGSM_epsilon_{epsilon}.png")
    plot_adversarial_grid(fixed_test_x_subset, fixed_test_adv.detach().cpu().numpy(),
                          fixed_test_y_subset, adv_preds, adv_confidences, adv_title, adv_save_path,
                          target_class=target_class)
    
    return test_accuracy, average_test_loss

# ------------------------------------------------------------------------------
# Argument parsing

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test a saved CIFAR-10 model with FGSM adversarial examples and plot the images.")
    parser.add_argument('--data_folder', type=str, default='data',
                        help='Folder containing CIFAR-10 data batches.')
    parser.add_argument('--model_folder', type=str, default='runs/run_mlp_1.0_1',
                        help='Folder containing the saved model checkpoint.')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Filename of the saved model checkpoint (inside model_folder).')
    parser.add_argument('--model_type', type=str, choices=['mlp', 'cnn'], default='mlp',
                        help="Type of model to load: 'mlp' or 'cnn'.")
    parser.add_argument('--hidden_units', type=int, default=100,
                        help='Number of hidden units for MLP model (ignored if model_type is cnn).')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for testing.')
    parser.add_argument('--epsilon', type=float, default=0.5,
                        help='Perturbation magnitude for FGSM attack.')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of iterations for iterative attack.')
    parser.add_argument('--targeted', action='store_true',
                        help='If set, perform a targeted FGSM attack.')
    parser.add_argument('--target_class', type=int, default=0,
                        help='Target class for the attack (only used if --targeted is set).')
    parser.add_argument('--output_folder', type=str, default='test_outputs',
                        help='Folder to save test output images.')
    return parser.parse_args()

# ------------------------------------------------------------------------------
# Main testing function

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CIFAR-10 data and pre-process.
    x_train, y_train, x_test, y_test = load_cifar10(args.data_folder)
    x_train = preprocess_cifar_images(x_train)
    x_test = preprocess_cifar_images(x_test)
    y_train = np.int32(y_train)
    y_test = np.int32(y_test)

    # For CIFAR-10, we use full-colour images of size 32x32.
    input_channels = 3
    image_size = 32

    # Select a random subset of 25 images from the test set for plotting.
    fixed_test_indices = np.random.choice(len(x_test), 25, replace=False)
    fixed_test_x = x_test[fixed_test_indices]
    fixed_test_y = y_test[fixed_test_indices]

    # Ensure the output folder exists.
    os.makedirs(args.output_folder, exist_ok=True)

    # Create the model architecture.
    if args.model_type == "cnn":
        model = CNNModel(input_channels, image_size).to(device)
        print("Using CNN model.")
    else:
        model = MLPModel(input_channels, image_size, hidden_units=args.hidden_units).to(device)
        print("Using MLP model.")

    # Load the saved model state.
    checkpoint_path = os.path.join(args.model_folder, args.model_path)
    if not os.path.isfile(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model from {checkpoint_path}")
    model.eval()

    # Standard testing.
    print("\nEvaluating the model on the test set...")
    test_accuracy, test_loss = test_model(model, x_test, y_test, fixed_test_x, fixed_test_y,
                                          device, args.batch_size, args.output_folder)

    # Adversarial testing.
    print("\nEvaluating the model on FGSM adversarial examples...")
    # Use the target class if a targeted attack is requested.
    adv_target = args.target_class if args.targeted else None
    adv_accuracy, adv_loss = test_model_adversarial(
        model, x_test, y_test, batch_size=args.batch_size,
        epsilon=args.epsilon, device=device, output_folder=args.output_folder,
        target_class=adv_target, num_steps=args.num_steps
    )

if __name__ == "__main__":
    args = parse_arguments()
    main(args)