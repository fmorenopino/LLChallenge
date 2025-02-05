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

# Fix random seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a trained CIFAR-10 model using adversarial attacks.")
    parser.add_argument('--experiment_folder', type=str, default="runs/run_cifar_cnn", 
                        help="Path to the experiment folder that contains the best_model.pth and testing subfolder.")
    parser.add_argument('--data_folder', type=str, default='data',
                        help="Folder with the cifar-10-batches-py files.")
    parser.add_argument('--model_type', type=str, default="cnn", choices=["mlp", "cnn"],
                        help="Type of model to use: 'mlp' for a fully-connected network or 'cnn' for a convolutional network.")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for evaluation.")
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help="FGSM perturbation magnitude (for Gaussian noise, this is the standard deviation).")
    return parser.parse_args()


# Define a helper class to duplicate terminal output into a log file.
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


# Create a unique experiment directory if needed and set up logging.
experiment_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# (The experiment folder is expected to already contain the trained model and testing subfolder.)
# Here, we simply use the experiment folder provided in the arguments.
args = parse_arguments()
exp_folder = args.experiment_folder
testing_dir = os.path.join(exp_folder, "testing")
os.makedirs(testing_dir, exist_ok=True)

# Open a log file in the experiment folder.
log_file = open(os.path.join(exp_folder, "output.txt"), "w")
sys.stdout = Tee(sys.stdout, log_file)


# CIFAR-10 model definitions (using full-colour images of size 32x32)

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
        new_size = image_size // 4  # After two poolings, image size is reduced by a factor of 4.
        self.fc1 = nn.Linear(64 * new_size * new_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Expecting x of shape (batch, input_channels, image_size, image_size)
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
    Plot a 5x5 grid of CIFAR-10 images with true labels, predicted labels and confidence.
    Assumes images are of shape (3, 32, 32).
    """
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)
    for idx, ax in enumerate(axes.flat):
        # Convert image from (3, 32, 32) to (32, 32, 3)
        img = np.transpose(images[idx], (1, 2, 0))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"T: {true_labels[idx]}\nP: {predictions[idx]}\nConf: {confidences[idx]:.2f}", fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)


def plot_adversarial_grid(orig_images, adv_images, true_labels, adv_preds, adv_confidences, title, save_path):
    """
    Plot a grid comparing original CIFAR-10 images and their adversarial counterparts.
    Each pair shows the original image (with its true label) and the adversarial image (with prediction and confidence).
    """
    n_pairs = len(orig_images)
    pairs_per_row = 5
    n_rows = n_pairs // pairs_per_row if n_pairs % pairs_per_row == 0 else (n_pairs // pairs_per_row) + 1

    fig, axes = plt.subplots(n_rows, pairs_per_row * 2, figsize=(2 * pairs_per_row, 2 * n_rows))
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
        ax_orig.set_title(f"T: {true_labels[idx]}", fontsize=8)
        ax_adv.imshow(adv_img)
        ax_adv.axis('off')
        ax_adv.set_title(f"T: {true_labels[idx]}\nP: {adv_preds[idx]}\nConf: {adv_confidences[idx]:.2f}", fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)


def evaluate_model(model, x_data, y_data, batch_size, device):
    """
    Compute the average loss and accuracy on a given dataset.
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
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


def fgsm_attack(model, data, target, epsilon, device):
    """
    Generate adversarial examples using FGSM.
    """
    data = data.clone().detach().to(device)
    data.requires_grad_()
    target = target.clone().detach().to(device)
    output = model(data)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    return torch.clamp(perturbed_data, 0, 1)


def gaussian_attack(model, data, target, epsilon, device):
    """
    Generate adversarial examples by adding Gaussian noise.
    """
    data = data.clone().detach().to(device)
    noise = torch.randn_like(data) * epsilon
    perturbed_data = data + noise
    return torch.clamp(perturbed_data, 0, 1)


def evaluate(model, x_data, y_data, batch_size, device):
    return evaluate_model(model, x_data, y_data, batch_size, device)


def test_model(model, x_test, y_test, fixed_test_x, fixed_test_y, batch_size, device, testing_dir):
    """
    Evaluate the model on the test set and generate a grid plot of predictions.
    """
    avg_loss, test_accuracy = evaluate_model(model, x_test, y_test, batch_size, device)
    print(f"Clean Test Accuracy: {test_accuracy:.2f}%")
    with torch.no_grad():
        fixed_test_data = torch.FloatTensor(fixed_test_x).to(device)
        fixed_test_output = model(fixed_test_data)
        softmax_out = F.softmax(fixed_test_output, dim=1)
        preds = fixed_test_output.data.max(1)[1].cpu().numpy()
        confidences = softmax_out.data.max(1)[0].cpu().numpy()
    test_save_path = os.path.join(testing_dir, "test_predictions.png")
    plot_grid(fixed_test_x, fixed_test_y, preds, confidences, "Test Predictions", test_save_path)
    return test_accuracy, avg_loss


def test_adversarial(model, x_test, y_test, batch_size, epsilon, device, testing_dir, attack_fn, attack_name):
    """
    Evaluate the model on adversarial examples generated by the specified attack.
    Generate a grid plot comparing original images and adversarial examples.
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for i in tqdm(range(0, len(y_test), batch_size), desc=f"Adversarial Testing ({attack_name})"):
        x_batch = torch.FloatTensor(x_test[i:i+batch_size]).to(device)
        y_batch = torch.LongTensor(y_test[i:i+batch_size]).to(device)
        x_batch_adv = attack_fn(model, x_batch, y_batch, epsilon, device)
        outputs = model(x_batch_adv)
        loss = F.cross_entropy(outputs, y_batch)
        total_loss += loss.item() * x_batch.size(0)
        preds = outputs.data.max(1)[1]
        total_correct += preds.eq(y_batch.data).sum().item()
        total_samples += x_batch.size(0)
    avg_loss = total_loss / total_samples
    adv_accuracy = (total_correct / total_samples) * 100.0
    print(f"{attack_name} Adversarial Test Accuracy (epsilon = {epsilon}): {adv_accuracy:.2f}%")
    
    # Plot a fixed subset (first 25 images)
    fixed_test_x_subset = x_test[:25]
    fixed_test_y_subset = y_test[:25]
    fixed_test_data = torch.FloatTensor(fixed_test_x_subset).to(device)
    fixed_test_adv = attack_fn(model, fixed_test_data, torch.LongTensor(fixed_test_y_subset).to(device), epsilon, device)
    with torch.no_grad():
        adv_output = model(fixed_test_adv)
        adv_softmax = F.softmax(adv_output, dim=1)
        adv_preds = adv_output.data.max(1)[1].cpu().numpy()
        adv_confidences = adv_softmax.data.max(1)[0].cpu().numpy()
    save_path = os.path.join(testing_dir, f"adv_test_comparison_{attack_name}_epsilon_{epsilon}.png")
    plot_adversarial_grid(fixed_test_x_subset, fixed_test_adv.detach().cpu().numpy(),
                          fixed_test_y_subset, adv_preds, adv_confidences,
                          f"Original vs {attack_name} Adversarial Examples (epsilon = {epsilon})", save_path)
    return adv_accuracy, avg_loss


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print arguments
    print("Arguments:")
    for arg, val in vars(args).items():
        print(f"  {arg}: {val}")
    print("\n")
    
    # Load CIFAR-10 test data.
    _, _, x_test, y_test = load_cifar10(args.data_folder)
    x_test = preprocess_cifar_images(x_test)
    y_test = np.int32(y_test)
    # Select a fixed subset for plotting.
    fixed_test_indices = np.random.choice(len(x_test), 25, replace=False)
    fixed_test_x = x_test[fixed_test_indices]
    fixed_test_y = y_test[fixed_test_indices]
    
    # Instantiate the model.
    if args.model_type == "cnn":
        model = CNNModel(input_channels=3, image_size=32).to(device)
        print("Using CNN model.")
    else:
        model = MLPModel(input_channels=3, image_size=32, hidden_units=100).to(device)
        print("Using MLP model.")
    
    # Load the best model from the experiment folder.
    model_path = os.path.join(args.experiment_folder, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")
    
    # Evaluate on clean test data.
    test_accuracy, test_loss = test_model(model, x_test, y_test, fixed_test_x, fixed_test_y, args.batch_size, device, os.path.join(args.experiment_folder, "testing"))
    
    # Evaluate adversarial performance using FGSM.
    fgsm_accuracy, fgsm_loss = test_adversarial(model, x_test, y_test, args.batch_size, args.epsilon, device, os.path.join(args.experiment_folder, "testing"), fgsm_attack, "FGSM")
    
    # Evaluate adversarial performance using Gaussian noise.
    gaussian_accuracy, gaussian_loss = test_adversarial(model, x_test, y_test, args.batch_size, args.epsilon, device, os.path.join(args.experiment_folder, "testing"), gaussian_attack, "Gaussian")
    
    print("\nEvaluation Summary:")
    print(f"Clean Test Accuracy: {test_accuracy:.2f}%")
    print(f"FGSM Adversarial Test Accuracy: {fgsm_accuracy:.2f}%")
    print(f"Gaussian Adversarial Test Accuracy: {gaussian_accuracy:.2f}%")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)