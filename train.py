import os
import sys
import datetime
import numpy as np
import h5py
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from utils import Tee, plot_grid, plot_adversarial_grid, evaluate, iterative_fgsm_attack, \
    load_cifar10, preprocess_cifar_images, MLPModel, CNNModel

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
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Total perturbation magnitude for the attack.')
    parser.add_argument('--target_class', type=int, default=0,
                        help='The target class to force the model to predict for adversarial examples.')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of iterations for the iterative attack.')
    return parser.parse_args()

# Parse arguments first so they can be used in naming the experiment folder.
args = parse_arguments()

# Print all the arguments at the beginning.
print("Arguments:")
for arg, value in sorted(vars(args).items()):
    print(f"{arg}: {value}")

# Create a unique experiment directory under "runs" using the arguments.
exp_folder_name = f"run_{args.model_type}_eps_{args.epsilon}_target_{args.target_class}"
exp_dir = os.path.join("runs", exp_folder_name)
os.makedirs(exp_dir, exist_ok=True)

# Create subdirectories for epoch images and test images.
epochs_dir = os.path.join(exp_dir, "epochs")
testing_dir = os.path.join(exp_dir, "testing")
os.makedirs(epochs_dir, exist_ok=True)
os.makedirs(testing_dir, exist_ok=True)

# Open a log file in the experiment folder.
log_file = open(os.path.join(exp_dir, "output.txt"), "w")
sys.stdout = Tee(sys.stdout, log_file)

# Instantiate TensorBoard writer with the experiment directory as the log folder.
writer = SummaryWriter(log_dir=exp_dir)

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
    
    # CIFAR-10 data loading and preprocessing.
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
    # Minimal change: using the same load_state_dict call as in your original code.
    model.load_state_dict(torch.load(model_save_path, weights_only=True))

    test_accuracy, test_loss = test_model(
        model, x_test, y_test, fixed_test_x, fixed_test_y, device,
        batch_size=args.batch_size, testing_dir=testing_dir
    )
    
    print("Testing on FGSM targeted adversarial examples...")
    adv_accuracy, adv_loss = test_model_adversarial(
        model, x_test, y_test, batch_size=args.batch_size,
        epsilon=args.epsilon, device=device, testing_dir=testing_dir,
        target_class=args.target_class, num_steps=args.num_steps
    )

    writer.close()

if __name__ == "__main__":
    main(args)