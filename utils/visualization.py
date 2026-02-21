"""Visualization utilities for training analysis."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import os


def plot_training_curves(history: Dict[str, List],
                        title: str = "Training Progress",
                        save_path: Optional[str] = None):
    """Plot training and validation curves.

    Args:
        history: Dict with keys like 'train_loss', 'val_loss', etc.
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MSE
    ax = axes[1]
    if 'train_mse' in history:
        ax.plot(history['train_mse'], label='Train MSE')
    if 'val_mse' in history:
        ax.plot(history['val_mse'], label='Val MSE')
    if 'test_mse' in history:
        # Filter None values for FL history
        test_mse = [x for x in history['test_mse'] if x is not None]
        rounds = [i for i, x in enumerate(history['test_mse']) if x is not None]
        ax.plot(rounds, test_mse, label='Test MSE', marker='o')
    ax.set_xlabel('Epoch/Round')
    ax.set_ylabel('MSE')
    ax.set_title('Mean Squared Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R2
    ax = axes[2]
    if 'val_r2' in history:
        ax.plot(history['val_r2'], label='Val R²')
    if 'test_r2' in history:
        test_r2 = [x for x in history['test_r2'] if x is not None]
        rounds = [i for i, x in enumerate(history['test_r2']) if x is not None]
        ax.plot(rounds, test_r2, label='Test R²', marker='o')
    ax.set_xlabel('Epoch/Round')
    ax.set_ylabel('R²')
    ax.set_title('R-squared')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(predictions: np.ndarray,
                          targets: np.ndarray,
                          num_bins: int = 5,
                          save_path: Optional[str] = None):
    """Plot confusion matrix for regression (binned).

    Bins continuous predictions into categories for visualization.

    Args:
        predictions: Model predictions [0, 1]
        targets: Ground truth [0, 1]
        num_bins: Number of bins for discretization
        save_path: Optional save path
    """
    # Bin the continuous values
    bins = np.linspace(0, 1, num_bins + 1)
    pred_bins = np.digitize(predictions, bins[1:-1])
    target_bins = np.digitize(targets, bins[1:-1])

    # Create confusion matrix
    conf_matrix = np.zeros((num_bins, num_bins), dtype=int)
    for p, t in zip(pred_bins, target_bins):
        conf_matrix[t, p] += 1

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(conf_matrix, cmap='Blues')

    # Labels
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(num_bins)]
    ax.set_xticks(np.arange(num_bins))
    ax.set_yticks(np.arange(num_bins))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.set_yticklabels(bin_labels)

    # Add text annotations
    for i in range(num_bins):
        for j in range(num_bins):
            text = ax.text(j, i, conf_matrix[i, j],
                          ha='center', va='center',
                          color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black')

    ax.set_xlabel('Predicted Congestion Level')
    ax.set_ylabel('True Congestion Level')
    ax.set_title('Congestion Prediction Confusion Matrix')

    plt.colorbar(im)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_prediction_scatter(predictions: np.ndarray,
                           targets: np.ndarray,
                           save_path: Optional[str] = None):
    """Plot prediction vs ground truth scatter plot.

    Args:
        predictions: Model predictions
        targets: Ground truth values
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(targets, predictions, alpha=0.5, edgecolors='none')
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')

    ax.set_xlabel('True Congestion Score')
    ax.set_ylabel('Predicted Congestion Score')
    ax.set_title('Prediction vs Ground Truth')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    textstr = f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_client_distribution(client_stats: Dict,
                            save_path: Optional[str] = None):
    """Plot data distribution across clients.

    Args:
        client_stats: Dict from FederatedTrainer.get_client_statistics()
        save_path: Optional save path
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Camera clients
    ax = axes[0]
    camera_samples = client_stats.get('camera_sample_distribution', [])
    if camera_samples:
        ax.bar(range(len(camera_samples)), camera_samples, color='steelblue')
        ax.set_xlabel('Client ID')
        ax.set_ylabel('Number of Samples')
        ax.set_title(f'Camera Clients (n={len(camera_samples)})')
        ax.axhline(np.mean(camera_samples), color='red', linestyle='--',
                   label=f'Mean: {np.mean(camera_samples):.1f}')
        ax.legend()

    # Vehicle clients
    ax = axes[1]
    vehicle_samples = client_stats.get('vehicle_sample_distribution', [])
    if vehicle_samples:
        ax.bar(range(len(vehicle_samples)), vehicle_samples, color='forestgreen')
        ax.set_xlabel('Client ID')
        ax.set_ylabel('Number of Samples')
        ax.set_title(f'Vehicle Clients (n={len(vehicle_samples)})')
        ax.axhline(np.mean(vehicle_samples), color='red', linestyle='--',
                   label=f'Mean: {np.mean(vehicle_samples):.1f}')
        ax.legend()

    plt.suptitle('Data Distribution Across Clients')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_federated_comparison(centralized_history: Dict,
                              federated_history: Dict,
                              save_path: Optional[str] = None):
    """Compare centralized vs federated training.

    Args:
        centralized_history: History from centralized training
        federated_history: History from federated training
        save_path: Optional save path
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # MSE comparison
    ax = axes[0]
    if 'val_mse' in centralized_history:
        ax.plot(centralized_history['val_mse'], label='Centralized', linewidth=2)
    if 'test_mse' in federated_history:
        test_mse = [x for x in federated_history['test_mse'] if x is not None]
        ax.plot(test_mse, label='Federated', linewidth=2)
    ax.set_xlabel('Epoch/Round')
    ax.set_ylabel('MSE')
    ax.set_title('MSE Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R2 comparison
    ax = axes[1]
    if 'val_r2' in centralized_history:
        ax.plot(centralized_history['val_r2'], label='Centralized', linewidth=2)
    if 'test_r2' in federated_history:
        test_r2 = [x for x in federated_history['test_r2'] if x is not None]
        ax.plot(test_r2, label='Federated', linewidth=2)
    ax.set_xlabel('Epoch/Round')
    ax.set_ylabel('R²')
    ax.set_title('R² Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Centralized vs Federated Training')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
