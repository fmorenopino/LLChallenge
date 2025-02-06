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
    parser = argparse.ArgumentParser(description="Test CIFAR-10 model adversarial attack on test data.")
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for testing.')
    parser.add_argument('--hidden_units', type=int, default=100, help='Number of hidden units in the MLP model.')
    parser.add_argument('--model_type', type=str, default="cnn", choices=["mlp", "cnn"],
                        help="Type of model to use: 'mlp' for a fully-connected network or 'cnn' for a convolutional network.")
    parser.add_argument('--data_folder', type=str, default='data',
                        help='Folder with the cifar-10-batches-py files.')
    parser.add_argument('--epsilon', type=float, default=0.5,
                        help='Total perturbation magnitude for the attack.')
    parser.add_argument('--target_class', type=int, default=3,
                        help='The target class to force the model to predict for adversarial examples.')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of iterations for the iterative attack.')
    parser.add_argument('--load_model', type=str, default='runs/run_cnn_copy',
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

# Open a log file in the experiment folder.
log_file = open(os.path.join(exp_dir, "output.txt"), "w")
sys.stdout = Tee(sys.stdout, log_file)

# Instantiate TensorBoard writer with the experiment directory as the log folder.
writer = SummaryWriter(log_dir=exp_dir)

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