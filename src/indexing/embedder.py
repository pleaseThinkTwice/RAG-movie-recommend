"""Embedding 编码器：bge-large-zh-v1.5，批量编码，支持 GPU。"""

from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.core.config import get_config, EmbeddingConfig


class Embedder:
    """文本 Embedding 编码器。"""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        if config is None:
            config = get_config().embedding
        self.config = config

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model: {config.model_name} on {device}...")
        self.model = SentenceTransformer(config.model_name, device=device)
        self.model.eval()

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """批量编码文本，返回归一化的 numpy 数组。

        Args:
            texts: 文本列表。
            batch_size: 批次大小，默认使用配置值。
            show_progress: 是否显示进度条。

        Returns:
            shape=(len(texts), dimension) 的归一化向量。
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """编码单条文本。"""
        return self.encode_batch([text], show_progress=False)[0]

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
