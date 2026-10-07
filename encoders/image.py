"""Визуальный энкодер на базе open_clip (ViT-B-32)."""
import io
import numpy as np
import torch
from PIL import Image
import open_clip
from tqdm import tqdm

from config import CONFIG


class ImageEncoder:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CONFIG.image_model_name,
            pretrained=CONFIG.image_pretrained,
            device=CONFIG.device,
        )
        self.model.eval()

    def encode_images(self, images_bytes: list[bytes]) -> np.ndarray:
        pil_images = [self._to_pil(b) for b in images_bytes]
        all_features = []
        batch_size = 8

        for i in tqdm(range(0, len(pil_images), batch_size), desc="Картинки"):
            batch = pil_images[i:i + batch_size]
            with torch.no_grad():
                tensors = torch.stack([self.preprocess(img) for img in batch]).to(CONFIG.device)
                features = self.model.encode_image(tensors)
                features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())

        if not all_features:
            return np.zeros((0, CONFIG.image_vector_size), dtype=np.float32)

        return np.vstack(all_features)

    @staticmethod
    def _to_pil(raw_bytes: bytes) -> Image.Image:
        try:
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), (255, 255, 255))