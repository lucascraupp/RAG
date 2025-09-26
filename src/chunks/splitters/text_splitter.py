from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

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
        with open(f"src/chunks/splitters/parameters.yaml", "r") as file:
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

    def _add_headers_to_chunks(
        self,
        original_text: str,
        chunks: List[str],
        filename: str,
        chunk_overlap: int = 0,
    ) -> pd.DataFrame:
        """
        Adiciona informações de headers aos metadados dos chunks.
        """
        # Extrai os headers do texto original
        self._header_tracker.extract_headers(original_text)

        # Mapeia cada chunk para sua posição no texto original
        chunks_with_metadata = []
        position = 0
        for chunk in chunks:
            start_pos = original_text.find(chunk, position)
            if start_pos != -1:
                end_pos = start_pos + len(chunk)

                # Cria metadados para cada chunk com seus headers
                headers = self._header_tracker.get_headers_for_chunk(start_pos, end_pos)
                metadata = {
                    "filename": filename,
                    "strategy": self._model_name,
                    "headers": headers["headers"] if "headers" in headers else [],
                    "start_chunk": start_pos,
                    "end_chunk": end_pos,
                }
                chunks_with_metadata.append({"metadata": metadata, "chunk": chunk})

                position = end_pos - chunk_overlap

        return pd.DataFrame(chunks_with_metadata)

    @abstractmethod
    def split_text(self, text: str) -> Union[List[str], Dict]:
        pass
