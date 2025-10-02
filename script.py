"""Image embedding and comparison script.

This module downloads images (DigitalOcean samples by default),
computes embeddings using a pretrained ResNet-50 model, and
compares the embeddings via cosine similarity.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse

import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# Default DigitalOcean image URLs for demonstration purposes.
DEFAULT_IMAGE_URLS: List[str] = [
    "https://assets.digitalocean.com/logos/DO_Logo_horizontal_blue.png",
    "https://assets.digitalocean.com/logos/DO_Logo_vertical_blue.png",
]


def configure_logging(verbose: bool) -> None:
    """Configure application logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def derive_filename(url: str, index: int) -> str:
    """Derive a filename from the image URL."""
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix or ".png"
    return f"image_{index}{suffix}"


def download_image(url: str, destination: Path) -> Path:
    """Download an image from a URL to the destination path."""
    logging.debug("Starting download from %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    destination.write_bytes(response.content)
    logging.debug("Saved image to %s", destination)
    return destination


def load_image(path: Path) -> Image.Image:
    """Load an image from disk and ensure it is RGB."""
    with path.open("rb") as image_file:
        image = Image.open(image_file)
        return image.convert("RGB")


def preprocess_images(images: Iterable[Image.Image]) -> torch.Tensor:
    """Preprocess images into tensors compatible with ResNet-50."""
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    tensors = [preprocess(image) for image in images]
    return torch.stack(tensors)


def build_feature_extractor(device: torch.device) -> torch.nn.Module:
    """Create a ResNet-50 feature extractor without the final classification layer."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    # Remove classification layer to use features from the penultimate layer.
    feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
    feature_extractor.to(device)
    return feature_extractor


def compute_embeddings(
    feature_extractor: torch.nn.Module, tensor_batch: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Compute embeddings for a batch of preprocessed images."""
    with torch.no_grad():
        features = feature_extractor(tensor_batch.to(device))
    embeddings = features.flatten(start_dim=1)
    return embeddings.cpu()


def cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute the cosine similarity matrix between embeddings."""
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings @ embeddings.T


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download images, compute embeddings, and compare them."
    )
    parser.add_argument(
        "urls",
        nargs="*",
        default=DEFAULT_IMAGE_URLS,
        help="Image URLs to download. Defaults to DigitalOcean sample images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./downloaded_images"),
        help="Directory to store downloaded images.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run the model on.",
    )
    return parser.parse_args()


def validate_device(device: torch.device) -> None:
    """Validate that the requested device is available."""
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    device = torch.device(args.device)
    validate_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Downloading %d images...", len(args.urls))
    image_paths = []
    for idx, url in enumerate(args.urls, start=1):
        filename = derive_filename(url, idx)
        destination = args.output_dir / filename
        image_paths.append(download_image(url, destination))

    logging.info("Loading and preprocessing images...")
    images = [load_image(path) for path in image_paths]
    tensor_batch = preprocess_images(images)

    logging.info("Building feature extractor (%s)...", device)
    feature_extractor = build_feature_extractor(device)

    logging.info("Computing embeddings...")
    embeddings = compute_embeddings(feature_extractor, tensor_batch, device)

    logging.info("Calculating cosine similarity matrix...")
    similarity_matrix = cosine_similarity_matrix(embeddings)

    for i in range(len(args.urls)):
        for j in range(i, len(args.urls)):
            logging.info(
                "Similarity between image %d and %d: %.4f", i + 1, j + 1, similarity_matrix[i, j]
            )


if __name__ == "__main__":
    main()
