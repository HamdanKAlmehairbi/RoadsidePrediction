"""CameraCNN trained on STREETS images (Stage 1: perception layer).

Trains MobileNetV2-based CNN to estimate congestion from individual
camera images. Uses traffic count scores as supervision labels.
Requires STREETS image data (trafficstate images or weekly archives).
"""
import torch
import numpy as np
import yaml
import os
import json
from torch.utils.data import DataLoader, random_split

from data.streets_loader import (
    load_streets_graph, load_streets_traffic_counts,
    load_streets_traffic_state, compute_congestion_from_counts,
    load_streets_image_index, extract_streets_camera_samples,
)
from data.camera_dataset import CameraDataset
from models.camera_cnn import CameraCNN
from training.metrics import compute_metrics


def main():
    with open('configs/default_config.yaml') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['experiments']['seed'])
    np.random.seed(config['experiments']['seed'])

    device = config['experiments']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    streets_cfg = config['data']['streets']
    cam_cfg = config['camera']

    # Load data
    print("Loading STREETS data...")
    graph = load_streets_graph(streets_cfg['dataroot'])
    counts = load_streets_traffic_counts(streets_cfg['dataroot'])

    try:
        states = load_streets_traffic_state(streets_cfg['dataroot'])
    except FileNotFoundError:
        states = None

    congestion = compute_congestion_from_counts(
        counts, states, max_count=streets_cfg['max_count_normalization']
    )

    camera_ids = graph['camera_ids']

    # Load image index
    print("Indexing images...")
    image_index = load_streets_image_index(streets_cfg['dataroot'], camera_ids)
    print(f"Found images for {len(image_index)} cameras")

    if not image_index:
        print("ERROR: No images found. Download STREETS image data first.")
        print("Images should be in STREETS/trafficstate/ or weekly archive directories.")
        return

    # Extract (image, score) samples
    samples = extract_streets_camera_samples(congestion, image_index, camera_ids)
    print(f"Total samples: {len(samples)}")

    if len(samples) == 0:
        print("ERROR: No image-score pairs matched. Check data alignment.")
        return

    # Create dataset
    dataset = CameraDataset(
        samples=samples,
        image_size=tuple(cam_cfg['input_size']),
    )

    # Train/val/test split
    n = len(dataset)
    n_train = int(n * config['data']['train_split'])
    n_val = int(n * config['data']['val_split'])
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config['experiments']['seed'])
    )

    batch_size = config['federated']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Create model
    model = CameraCNN(
        feature_dim=cam_cfg['feature_dim'],
        backbone=cam_cfg['backbone'],
        pretrained=cam_cfg['pretrained'],
        dropout=cam_cfg['dropout'],
    ).to(device)
    print(f"Parameters: {model.get_num_params()}")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=config['federated']['lr'],
                                 weight_decay=config['training']['weight_decay'])
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_state = None

    for epoch in range(30):
        # Train
        model.train()
        train_loss = 0.0
        for images, scores in train_loader:
            images = images.to(device)
            scores = scores.to(device)

            optimizer.zero_grad()
            preds = model.forward_predict(images).squeeze(-1)
            loss = criterion(preds, scores)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, scores in val_loader:
                images = images.to(device)
                preds = model.forward_predict(images).squeeze(-1)
                val_preds.append(preds.cpu())
                val_targets.append(scores)

        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_metrics = compute_metrics(val_preds, val_targets)

        print(f"Epoch {epoch+1}/30 - Train Loss: {train_loss:.4f}, "
              f"Val MSE: {val_metrics['mse']:.4f}, "
              f"Val MAE: {val_metrics['mae']:.4f}")

        # Early stopping
        if val_metrics['mse'] < best_val_loss:
            best_val_loss = val_metrics['mse']
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for images, scores in test_loader:
            images = images.to(device)
            preds = model.forward_predict(images).squeeze(-1)
            test_preds.append(preds.cpu())
            test_targets.append(scores)

    test_preds = torch.cat(test_preds).numpy()
    test_targets = torch.cat(test_targets).numpy()
    test_metrics = compute_metrics(test_preds, test_targets)

    print(f"\nTest Results: {test_metrics}")

    # Save results
    results = {
        'model': cam_cfg['backbone'],
        'num_params': model.get_num_params(),
        'num_samples': len(samples),
        'test_metrics': test_metrics,
    }

    os.makedirs(config['experiments']['output_dir'], exist_ok=True)
    output_path = os.path.join(config['experiments']['output_dir'], 'vision_results.json')

    def to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=to_serializable)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
