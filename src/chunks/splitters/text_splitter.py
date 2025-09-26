from abc import ABC
from typing import Dict, List, Optional

import pandas as pd
import yaml
from langchain_openai import OpenAIEmbeddings

from src.chunks.header import HeaderTracker


class TextSplitter(ABC):
    def __init__(self, model_name: str, params: Optional[Dict] = None):
        self._model_name: str = model_name
        self._params: Dict = params or self.__load_params(model_name)
        self._chunks_data: pd.DataFrame = pd.DataFrame()
        self._header_tracker = HeaderTracker()

    @property
    def model_name(self) -> str:
        return self._model_name

    def __load_params(self, model_name: str) -> Dict[str, str]:
        with open(f"src/chunks/parameters.yaml", "r") as file:
            params = yaml.safe_load(file)
        return params[model_name]

    def generate_embeddings(self) -> List[Dict]:
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")

        data = []
        for row in self._chunks_data.itertuples():
            chunk = row.chunk
            data.append(
                {
                    "filename": row.metadata["filename"],
                    "strategy": self._model_name,
                    "strategy_params": self._params,
                    "chunk_metadata": row.metadata,
                    "chunk_content": chunk,
                    "chunk_size": len(chunk),
                    "chunk_embeddings": embedding.embed_query(chunk),
                }
            )
        return data
