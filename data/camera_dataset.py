"""Camera dataset for federated learning - loads nuScenes camera images."""
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Optional, Callable


class CameraDataset(Dataset):
    """PyTorch Dataset for nuScenes camera images with congestion labels.

    Args:
        samples: List of dicts with keys 'image_path' and 'congestion_score'
        transform: Optional torchvision transforms to apply to images

    Returns:
        Tuple of (image_tensor, congestion_score) where:
        - image_tensor: (C, H, W) normalized image
        - congestion_score: float tensor in [0, 1]
    """

    def __init__(self, samples: List[dict], transform: Optional[Callable] = None):
        """Initialize camera dataset.

        Args:
            samples: List of sample dicts containing:
                - 'image_path': Path to image file
                - 'congestion_score': Float in [0, 1]
                - Optional: 'sample_token', 'scene_token' for tracking
            transform: Optional image transforms (e.g., from torchvision.transforms)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, congestion_score_tensor)
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            import torchvision.transforms as T
            default_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
            image = default_transform(image)

        congestion_score = sample['congestion_score']

        return image, torch.tensor(congestion_score, dtype=torch.float32)

    def get_sample_info(self, idx: int) -> dict:
        """Get metadata for a sample (useful for debugging/analysis).

        Args:
            idx: Sample index

        Returns:
            Dict with sample metadata
        """
        return self.samples[idx].copy()

    @staticmethod
    def get_default_transforms(input_size: Tuple[int, int] = (224, 224),
                               augment: bool = False) -> Callable:
        """Get default image transforms.

        Args:
            input_size: Target image size (H, W)
            augment: Whether to apply data augmentation

        Returns:
            Composed transforms
        """
        import torchvision.transforms as T

        if augment:
            return T.Compose([
                T.Resize(input_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize(input_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
