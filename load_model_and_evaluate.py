import os
import sys
import datetime
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Fix random seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Test CIFAR-10 model adversarial attack on test data.")
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for testing.')
    parser.add_argument('--hidden_units', type=int, default=100, help='Number of hidden units in the MLP model.')
    parser.add_argument('--model_type', type=str, default="mlp", choices=["mlp", "cnn"],
                        help="Type of model to use: 'mlp' for a fully-connected network or 'cnn' for a convolutional network.")
    parser.add_argument('--data_folder', type=str, default='data',
                        help='Folder with the cifar-10-batches-py files.')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Total perturbation magnitude for the attack.')
    parser.add_argument('--target_class', type=int, default=0,
                        help='The target class to force the model to predict for adversarial examples.')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of iterations for the iterative attack.')
    # New argument to specify the path to the pre-trained model.
    parser.add_argument('--load_model', type=str, default='runs/run_mlp',
                        help='Path to the pre-trained model weights file or directory containing best_model.pth.')
    return parser.parse_args()


args = parse_arguments()

# Print all the arguments.
print("Arguments:")
for arg, value in sorted(vars(args).items()):
    print(f"{arg}: {value}")

# Determine the experiment directory based on the load_model argument.
if os.path.isdir(args.load_model):
    exp_dir = args.load_model
else:
    exp_dir = os.path.dirname(args.load_model)

# Create a subfolder for attack evaluation results inside the experiment folder.
attack_evaluation_dir = os.path.join(exp_dir, f"attack_evaluation_eps_{args.epsilon}_target_{args.target_class}")
os.makedirs(attack_evaluation_dir, exist_ok=True)

# Create a helper class to duplicate sys.stdout writes to a log file.
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

# Open a log file in the experiment folder.
log_file = open(os.path.join(exp_dir, "output.txt"), "w")
sys.stdout = Tee(sys.stdout, log_file)

# Instantiate TensorBoard writer with the experiment directory as the log folder.
writer = SummaryWriter(log_dir=exp_dir)


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

def plot_adversarial_grid(orig_images, adv_images, true_labels, adv_preds, adv_confidences, title, save_path, target_class, acc_clean, acc_adv):
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
    Normalize CIFAR-10 images to [0, 1] and cast to float32.
    """
    x = x / 255.0
    return x.astype(np.float32)

def test_model_adversarial(model, x_test, y_test, batch_size, epsilon, device, testing_dir, target_class, num_steps=10):
    """
    Test the model on adversarial examples using a targeted attack.
    """
    model.eval()
    total_correct_clean = 0
    total_correct_adv = 0
    total_samples = 0
    
    for i in tqdm(range(0, len(y_test), batch_size), desc="Adversarial Testing (FGSM)"):
        x_batch = torch.FloatTensor(x_test[i:i+batch_size]).to(device)
        y_batch = torch.LongTensor(y_test[i:i+batch_size]).to(device)
        
        outputs_clean = model(x_batch)
        preds_clean = outputs_clean.data.max(1)[1]
        total_correct_clean += preds_clean.eq(y_batch.data).sum().item()
        
        target_tensor = torch.full_like(y_batch, target_class)
        x_batch_adv = iterative_fgsm_attack(model, x_batch, target_tensor, epsilon, device, num_steps=num_steps)
        
        outputs_adv = model(x_batch_adv)
        preds_adv = outputs_adv.data.max(1)[1]
        total_correct_adv += (preds_adv == target_tensor).sum().item()
        
        total_samples += x_batch.size(0)
    
    acc_clean = (total_correct_clean / total_samples) * 100.0
    acc_adv = (total_correct_adv / total_samples) * 100.0
    print(f"Clean Test Accuracy: {acc_clean:.2f}%")
    print(f"Adversarial Test Accuracy (target {target_class}): {acc_adv:.2f}%")
    
    writer.add_scalar("Test/FGSM_Accuracy_Clean", acc_clean)
    writer.add_scalar("Test/FGSM_Accuracy_Adv", acc_adv)
    
    fixed_test_x_subset = x_test[:25]
    fixed_test_y_subset = y_test[:25]
    fixed_test_data = torch.FloatTensor(fixed_test_x_subset).to(device)
    target_tensor = torch.full((fixed_test_data.size(0),), target_class, dtype=torch.long).to(device)
    fixed_test_adv = iterative_fgsm_attack(model, fixed_test_data, target_tensor, epsilon, device, num_steps=num_steps)
    
    with torch.no_grad():
        adv_output = model(fixed_test_adv)
        adv_softmax = F.softmax(adv_output, dim=1)
        adv_preds = adv_output.data.max(1)[1].cpu().numpy()
        adv_confidences = adv_softmax.data.max(1)[0].cpu().numpy()
    
    adv_title = f"Original vs Iterative FGSM Targeted Adversarial Examples (epsilon = {epsilon})"
    adv_save_path = os.path.join(testing_dir, f"adv_test_comparison_FGSM_eps_{epsilon}_target_{target_class}.png")
    plot_adversarial_grid(fixed_test_x_subset, fixed_test_adv.detach().cpu().numpy(),
                          fixed_test_y_subset, adv_preds, adv_confidences, adv_title, adv_save_path,
                          target_class=target_class, acc_clean=acc_clean, acc_adv=acc_adv)
    
    return acc_clean, acc_adv


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CIFAR-10 data and preprocess.
    _, _, x_test, y_test = load_cifar10(args.data_folder)
    x_test = preprocess_cifar_images(x_test)
    y_test = np.int32(y_test)
    input_channels = 3
    image_size = 32

    # Create a fixed subset of test images for visualisation.
    fixed_test_indices = np.random.choice(len(x_test), 25, replace=False)
    fixed_test_x = x_test[fixed_test_indices]
    fixed_test_y = y_test[fixed_test_indices]
    
    # Instantiate the model.
    if args.model_type == "cnn":
        model = CNNModel(input_channels, image_size).to(device)
        print("Using CNN model.")
    else:
        model = MLPModel(input_channels, image_size, hidden_units=args.hidden_units).to(device)
        print("Using MLP model.")
    
    # Determine the model file path.
    if os.path.isdir(args.load_model):
        model_path = os.path.join(args.load_model, "best_model.pth")
    else:
        model_path = args.load_model
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print("Model loaded.")

    # Evaluate on the clean test set.
    test_accuracy, test_loss = evaluate(model, x_test, y_test, args.batch_size, device)
    print(f"Clean Test Accuracy: {test_accuracy:.2f}%")
    
    # Test the adversarial attack.
    print("Testing on FGSM targeted adversarial examples...")
    adv_accuracy, adv_loss = test_model_adversarial(
        model, x_test, y_test, batch_size=args.batch_size,
        epsilon=args.epsilon, device=device, testing_dir=attack_evaluation_dir,
        target_class=args.target_class, num_steps=args.num_steps
    )
    print(f"Iterative FGSM Targeted Adversarial Test Accuracy (target = {args.target_class}, epsilon = {args.epsilon}): {adv_accuracy:.2f}%")
    
    writer.close()


if __name__ == "__main__":
    main(args)