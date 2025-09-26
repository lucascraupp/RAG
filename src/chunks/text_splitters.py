import copy
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class HeaderTracker:
    def __init__(self, header_pattern=r"^#{1,6}\s+(.+)$", min_level=1, max_level=6):
        self._header_pattern = header_pattern
        self._min_level = min_level
        self._max_level = max_level
        self._headers = []  # Lista de tuplas (posição, nível, texto do header)

    def extract_headers(self, text: str) -> None:
        """Extrai todos os headers do texto com suas posições."""
        self._headers = []
        lines = text.split("\n")
        position = 0

        for line in lines:
            match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if match:
                level = len(match.group(1))
                if self._min_level <= level <= self._max_level:
                    self._headers.append((position, level, match.group(2).strip()))
            position += len(line) + 1  # +1 para o caractere de quebra de linha

    def get_headers_for_chunk(
        self, chunk_start: int, chunk_end: int
    ) -> Dict[str, List[str]]:
        """Retorna os headers ativos para um chunk específico."""
        active_headers: Dict[str, List[str]] = {}

        header_before_chunk = max(
            (h for h in self._headers if h[0] <= chunk_start),
            default=None,
            key=lambda x: x[0],
        )

        if header_before_chunk and chunk_start != header_before_chunk[0]:
            active_headers["headers"] = [header_before_chunk[2]]

        # Adiciona os headers ativos que estão dentro do intervalo do chunk
        for pos, _, header_text in self._headers:
            if chunk_start <= pos <= chunk_end:
                active_headers.setdefault("headers", []).append(header_text)

        return active_headers


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


class CharacterTextSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("character_text_splitter", params)

    def split_text(self, text: str, filename: str) -> List[str]:
        splitter = CharacterTextSplitter(**self._params)

        logger = logging.getLogger("langchain_text_splitters.base")
        original_level = logger.level

        try:
            logger.setLevel(logging.ERROR)

            chunks = splitter.split_text(text)

            self._chunks_data = self._add_headers_to_chunks(
                text, chunks, filename, self._params.get("chunk_overlap", 0)
            )
        finally:
            logger.setLevel(original_level)

        return chunks


class RecursiveTextSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("recursive_text_splitter", params)

    def split_text(self, text: str, filename: str) -> List[str]:
        splitted_params = self._params.copy()
        splitted_params.pop("separators", None)

        chunks = [text]
        for separator in self._params.get("separators"):
            splitted_params["separator"] = separator

            text_splitter = CharacterTextSplitters(splitted_params)

            new_chunks = []
            for text in chunks:
                new_chunks.extend(text_splitter.split_text(text, filename))
        chunks = new_chunks

        self._chunks_data = self._add_headers_to_chunks(
            text, chunks, filename, self._params.get("chunk_overlap", 0)
        )

        return chunks


class RecursiveCharacterTextSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("recursive_character_text_splitter", params)

    def split_text(self, text: str, filename: str) -> List[str]:
        modify_params = self._params.copy()

        # Converter length_function de string para função real
        if isinstance(modify_params, dict) and "length_function" in modify_params:
            if modify_params["length_function"] == "len":
                modify_params["length_function"] = len

        splitter = RecursiveCharacterTextSplitter(**modify_params)

        logger = logging.getLogger("langchain_text_splitters.base")
        original_level = logger.level

        try:
            logger.setLevel(logging.ERROR)

            chunks = splitter.split_text(text)

            self._chunks_data = self._add_headers_to_chunks(
                text, chunks, filename, self._params.get("chunk_overlap", 0)
            )
        finally:
            logger.setLevel(original_level)

        return chunks


class MarkdownHeaderMetadataSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("markdown_header_metadata_splitter", params)

    def split_text(self, text: str, filename: str) -> Dict:
        splitter = MarkdownHeaderTextSplitter(**self._params)

        chunks = splitter.split_text(text)

        chunks_data = pd.DataFrame(
            {
                "metadata": [
                    {
                        "filename": filename,
                        "strategy": self._model_name,
                        **chunk.metadata,
                    }
                    for chunk in chunks
                ],
                "chunk": [chunk.page_content for chunk in chunks],
            }
        )

        self._chunks_data = chunks_data

        return chunks


class SemanticSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("semantic_splitter", params)

    def split_text(self, text: str, filename: str) -> List[str]:
        embeddings = OpenAIEmbeddings(
            model=self._params.get("embedding_model", "text-embedding-3-small")
        )

        params = copy.deepcopy(self._params)

        recursive_params = params.get("recursive_params", {})
        semantic_params = params.get("semantic_params", {})
        semantic_params["embeddings"] = embeddings

        recursive_splitter = RecursiveCharacterTextSplitters(recursive_params)

        chunks = []
        for chunk in recursive_splitter.split_text(text):
            splitter = SemanticChunker(**semantic_params)
            test = splitter.split_text(chunk)
            chunks.extend(test)

        self._chunks_data = self._add_headers_to_chunks(
            text, chunks, filename, semantic_params.get("chunk_overlap", 0)
        )

        return chunks
