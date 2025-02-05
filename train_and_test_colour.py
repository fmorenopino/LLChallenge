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
    parser = argparse.ArgumentParser(description="Train CIFAR-10 model with a validation set.")
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training.')
    parser.add_argument('--print_freq', type=int, default=5, help='Frequency (in epochs) to print training info.')
    parser.add_argument('--plot_freq', type=int, default=5, help='Frequency (in epochs) to plot validation predictions.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for the optimiser.')
    parser.add_argument('--hidden_units', type=int, default=100, help='Number of hidden units in the MLP model.')
    parser.add_argument('--model_type', type=str, default="mlp", choices=["mlp", "cnn"],
                        help="Type of model to use: 'mlp' for a fully-connected network or 'cnn' for a convolutional network.")
    parser.add_argument('--data_folder', type=str, default='data',
                        help='Folder with the cifar-10-batches-py files.')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='FGSM perturbation magnitude (default: 0.1). For Gaussian noise this is the standard deviation.')
    return parser.parse_args()


# Create a unique experiment directory under "runs"
experiment_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_dir = os.path.join("runs", f"run_{experiment_id}")
os.makedirs(exp_dir, exist_ok=True)
# Create subdirectories for epoch images and test images
epochs_dir = os.path.join(exp_dir, "epochs")
testing_dir = os.path.join(exp_dir, "testing")
os.makedirs(epochs_dir, exist_ok=True)
os.makedirs(testing_dir, exist_ok=True)

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

# Instantiate TensorBoard writer with the experiment directory as the log folder
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
        # After two poolings, the spatial dimension is reduced by a factor of 4.
        new_size = image_size // 4
        self.fc1 = nn.Linear(64 * new_size * new_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x should be of shape (batch, input_channels, image_size, image_size)
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
    n_rows = n_pairs // pairs_per_row
    if n_pairs % pairs_per_row != 0:
        n_rows += 1

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


def fgsm_attack(model, data, target, epsilon, device):
    """
    Generate adversarial examples using the Fast Gradient Sign Method (FGSM).
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
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data


def gaussian_attack(model, data, target, epsilon, device):
    """
    Generate adversarial examples by adding Gaussian noise.
    """
    data = data.clone().detach().to(device)
    noise = torch.randn_like(data) * epsilon
    perturbed_data = data + noise
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data


def train_model(model, optimizer, x_train, y_train, x_val, y_val,
                fixed_test_x, fixed_test_y, fixed_val_x, fixed_val_y, device,
                num_epochs, batch_size, print_freq, plot_freq, epochs_dir, model_save_path):
    """
    Train the model using training and validation sets.
    The best model (based on validation accuracy) is saved to disk.
    """
    best_val_accuracy = 0.0
    best_epoch = 0
    num_train_samples = len(y_train)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        train_accuracies = []
        permutation = np.random.permutation(num_train_samples)
        x_train = x_train[permutation]
        y_train = y_train[permutation]

        for i in range(0, num_train_samples, batch_size):
            x_batch = torch.FloatTensor(x_train[i:i+batch_size]).to(device)
            y_batch = torch.LongTensor(y_train[i:i+batch_size]).to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = F.cross_entropy(outputs, y_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            preds = outputs.data.max(1)[1]
            batch_accuracy = (preds.eq(y_batch.data).sum().item() / float(x_batch.size(0))) * 100.0
            train_accuracies.append(batch_accuracy)

        average_epoch_loss = epoch_loss / (num_train_samples / batch_size)
        train_accuracy = np.mean(train_accuracies)

        val_loss, val_accuracy = evaluate(model, x_val, y_val, batch_size, device)

        writer.add_scalar("Train/Loss", average_epoch_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_accuracy, epoch)
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/Accuracy", val_accuracy, epoch)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)

        if epoch % plot_freq == 0:
            model.eval()
            with torch.no_grad():
                fixed_val_data = torch.FloatTensor(fixed_val_x).to(device)
                fixed_val_output = model(fixed_val_data)
                fixed_val_softmax = F.softmax(fixed_val_output, dim=1)
                fixed_val_preds = fixed_val_output.data.max(1)[1].cpu().numpy()
                fixed_val_confidences = fixed_val_softmax.data.max(1)[0].cpu().numpy()
            model.train()
            val_plot_title = f"Epoch {epoch} Validation Predictions"
            val_plot_save_path = os.path.join(epochs_dir, f'epoch_{epoch}_val.png')
            plot_grid(fixed_val_x, fixed_val_y, fixed_val_preds, fixed_val_confidences, val_plot_title, val_plot_save_path)

        if (epoch % print_freq) == 0:
            print(f"Epoch {epoch}: Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch}")
    return best_epoch, best_val_accuracy


def test_model(model, x_test, y_test, fixed_test_x, fixed_test_y, device, batch_size, testing_dir):
    """
    Test the model on the test set and log the metrics.
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

    writer.add_scalar("Test/Loss", average_test_loss)
    writer.add_scalar("Test/Accuracy", test_accuracy)

    with torch.no_grad():
        fixed_test_data = torch.FloatTensor(fixed_test_x).to(device)
        fixed_test_output = model(fixed_test_data)
        fixed_test_softmax = F.softmax(fixed_test_output, dim=1)
        fixed_test_preds = fixed_test_output.data.max(1)[1].cpu().numpy()
        fixed_test_confidences = fixed_test_softmax.data.max(1)[0].cpu().numpy()
    test_title = "Test Predictions"
    test_save_path = os.path.join(testing_dir, "test_predictions.png")
    plot_grid(fixed_test_x, fixed_test_y, fixed_test_preds, fixed_test_confidences, test_title, test_save_path)
    return test_accuracy, average_test_loss


def test_model_adversarial(model, x_test, y_test, batch_size, epsilon, device, testing_dir):
    """
    Test the model on adversarial examples generated using FGSM.
    """
    test_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.eval()
    for i in tqdm(range(0, len(y_test), batch_size), desc="Adversarial Testing (FGSM)"):
        x_batch = torch.FloatTensor(x_test[i:i+batch_size]).to(device)
        y_batch = torch.LongTensor(y_test[i:i+batch_size]).to(device)
        x_batch_adv = fgsm_attack(model, x_batch, y_batch, epsilon, device)
        outputs = model(x_batch_adv)
        loss = F.cross_entropy(outputs, y_batch)
        test_loss += loss.item() * x_batch.size(0)
        preds = outputs.data.max(1)[1]
        total_correct += preds.eq(y_batch.data).sum().item()
        total_samples += x_batch.size(0)
    average_test_loss = test_loss / total_samples
    test_accuracy = (total_correct / total_samples) * 100.0
    print(f"FGSM Adversarial Test Accuracy (epsilon = {epsilon}): {test_accuracy:.2f}%")

    writer.add_scalar("Test/FGSM_Loss", average_test_loss)
    writer.add_scalar("Test/FGSM_Accuracy", test_accuracy)

    fixed_test_x_subset = x_test[:25]
    fixed_test_y_subset = y_test[:25]
    fixed_test_data = torch.FloatTensor(fixed_test_x_subset).to(device)
    fixed_test_adv = fgsm_attack(model, fixed_test_data, torch.LongTensor(fixed_test_y_subset).to(device), epsilon, device)
    with torch.no_grad():
        adv_output = model(fixed_test_adv)
        adv_softmax = F.softmax(adv_output, dim=1)
        adv_preds = adv_output.data.max(1)[1].cpu().numpy()
        adv_confidences = adv_softmax.data.max(1)[0].cpu().numpy()
    adv_title = f"Original vs FGSM Adversarial Examples (epsilon = {epsilon})"
    adv_save_path = os.path.join(testing_dir, f"adv_test_comparison_FGSM_epsilon_{epsilon}.png")
    plot_adversarial_grid(fixed_test_x_subset, fixed_test_adv.detach().cpu().numpy(),
                          fixed_test_y_subset, adv_preds, adv_confidences, adv_title, adv_save_path)
    
    return test_accuracy, average_test_loss


def test_model_adversarial_gaussian(model, x_test, y_test, batch_size, epsilon, device, testing_dir):
    """
    Test the model on adversarial examples generated by adding Gaussian noise.
    """
    test_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.eval()
    for i in tqdm(range(0, len(y_test), batch_size), desc="Adversarial Testing (Gaussian)"):
        x_batch = torch.FloatTensor(x_test[i:i+batch_size]).to(device)
        y_batch = torch.LongTensor(y_test[i:i+batch_size]).to(device)
        x_batch_adv = gaussian_attack(model, x_batch, y_batch, epsilon, device)
        outputs = model(x_batch_adv)
        loss = F.cross_entropy(outputs, y_batch)
        test_loss += loss.item() * x_batch.size(0)
        preds = outputs.data.max(1)[1]
        total_correct += preds.eq(y_batch.data).sum().item()
        total_samples += x_batch.size(0)
    average_test_loss = test_loss / total_samples
    test_accuracy = (total_correct / total_samples) * 100.0
    print(f"Gaussian Adversarial Test Accuracy (epsilon = {epsilon}): {test_accuracy:.2f}%")

    writer.add_scalar("Test/Gaussian_Loss", average_test_loss)
    writer.add_scalar("Test/Gaussian_Accuracy", test_accuracy)

    fixed_test_x_subset = x_test[:25]
    fixed_test_y_subset = y_test[:25]
    fixed_test_data = torch.FloatTensor(fixed_test_x_subset).to(device)
    fixed_test_adv = gaussian_attack(model, fixed_test_data, torch.LongTensor(fixed_test_y_subset).to(device), epsilon, device)
    with torch.no_grad():
        adv_output = model(fixed_test_adv)
        adv_softmax = F.softmax(adv_output, dim=1)
        adv_preds = adv_output.data.max(1)[1].cpu().numpy()
        adv_confidences = adv_softmax.data.max(1)[0].cpu().numpy()
    adv_title = f"Original vs Gaussian Adversarial Examples (epsilon = {epsilon})"
    adv_save_path = os.path.join(testing_dir, f"adv_test_comparison_Gaussian_epsilon_{epsilon}.png")
    plot_adversarial_grid(fixed_test_x_subset, fixed_test_adv.detach().cpu().numpy(),
                          fixed_test_y_subset, adv_preds, adv_confidences, adv_title, adv_save_path)
    
    return test_accuracy, average_test_loss


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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CIFAR-10 data loading and preprocessing
    x_train, y_train, x_test, y_test = load_cifar10(args.data_folder)
    x_train = preprocess_cifar_images(x_train)
    x_test = preprocess_cifar_images(x_test)
    y_train = np.int32(y_train)
    y_test = np.int32(y_test)
    # For CIFAR-10, we use full-colour images of size 32x32.
    input_channels = 3
    image_size = 32

    # Create a validation split from the training set.
    indices = np.arange(len(y_train))
    np.random.shuffle(indices)
    val_split = int(0.1 * len(y_train))
    val_indices = indices[:val_split]
    train_indices = indices[val_split:]
    x_val = x_train[val_indices]
    y_val = y_train[val_indices]
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]

    fixed_test_indices = np.random.choice(len(x_test), 25, replace=False)
    fixed_test_x = x_test[fixed_test_indices]
    fixed_test_y = y_test[fixed_test_indices]
    fixed_val_indices = np.random.choice(len(x_val), 25, replace=False)
    fixed_val_x = x_val[fixed_val_indices]
    fixed_val_y = y_val[fixed_val_indices]
    
    if args.model_type == "cnn":
        model = CNNModel(input_channels, image_size).to(device)
        print("Using CNN model.")
    else:
        model = MLPModel(input_channels, image_size, hidden_units=args.hidden_units).to(device)
        print("Using MLP model.")

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    model_save_path = os.path.join(exp_dir, "best_model.pth")

    best_epoch, best_val_accuracy = train_model(
        model, optimizer, x_train, y_train, x_val, y_val,
        fixed_test_x, fixed_test_y, fixed_val_x, fixed_val_y, device,
        num_epochs=args.num_epochs, batch_size=args.batch_size,
        print_freq=args.print_freq, plot_freq=args.plot_freq,
        epochs_dir=epochs_dir, model_save_path=model_save_path
    )

    print("Training complete. Loading the best model for testing.")
    model.load_state_dict(torch.load(model_save_path))

    test_accuracy, test_loss = test_model(
        model, x_test, y_test, fixed_test_x, fixed_test_y, device,
        batch_size=args.batch_size, testing_dir=testing_dir
    )
    
    print("Testing on FGSM adversarial examples...")
    adv_accuracy, adv_loss = test_model_adversarial(
        model, x_test, y_test, batch_size=args.batch_size,
        epsilon=args.epsilon, device=device, testing_dir=testing_dir
    )
    
    print("Testing on Gaussian adversarial examples...")
    adv_gauss_accuracy, adv_gauss_loss = test_model_adversarial_gaussian(
        model, x_test, y_test, batch_size=args.batch_size,
        epsilon=args.epsilon, device=device, testing_dir=testing_dir
    )

    writer.close()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)