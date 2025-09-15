import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import yaml
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class TextSplitter(ABC):
    def __init__(self, model_name: str, params: Optional[Dict] = None):
        self._model_name: str = model_name
        self._params: Dict = params or self.__load_params(model_name)
        self._chunks: List[str] = []
        self._metadata: List[Dict] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def chunks(self) -> List[str]:
        return self._chunks

    def __load_params(self, model_name: str) -> Dict[str, str]:
        with open(f"src/chuncks/parameters.yaml", "r") as file:
            params = yaml.safe_load(file)
        return params[model_name]

    def generate_embeddings(self, filename: str) -> List[Dict]:
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")

        data = []
        for metadata, chunk in zip(self._metadata, self._chunks):
            data.append(
                {
                    "filename": filename,
                    "strategy": self._model_name,
                    "strategy_params": self._params,
                    "chunk_metadata": metadata or {},
                    "chunk_content": chunk,
                    "chunk_size": len(chunk),
                    "chunk_embeddings": embedding.embed_query(chunk),
                }
            )
        return data

    @abstractmethod
    def split_text(self, text: str) -> None:
        pass


class CharacterTextSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("character_text_splitter", params)

    def split_text(self, text: str) -> None:
        splitter = CharacterTextSplitter(**self._params)

        logger = logging.getLogger("langchain_text_splitters.base")
        original_level = logger.level

        try:
            logger.setLevel(logging.ERROR)

            self._chunks = splitter.split_text(text)
            self._metadata = [{}] * len(self._chunks)
        finally:
            logger.setLevel(original_level)


class RecursiveTextSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("recursive_text_splitter", params)

    def split_text(self, text: str) -> None:
        splitted_params = self._params.copy()

        chunks = [text]
        for separator in self._params.get("separators"):
            splitted_params["separator"] = separator
            new_chunks = []

            for text in chunks:
                new_chunks.extend(
                    CharacterTextSplitters(text, self._filename, splitted_params)
                )
        chunks = new_chunks

        self._chunks = chunks
        self._metadata = [{}] * len(self._chunks)


class RecursiveCharacterTextSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("recursive_character_text_splitter", params)

    def split_text(self, text: str) -> None:
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

            self._chunks = splitter.split_text(text)
            self._metadata = [{}] * len(self._chunks)
        finally:
            logger.setLevel(original_level)


class MarkdownHeaderMetadataSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("markdown_header_metadata_splitter", params)

    def split_text(self, text: str) -> None:
        splitter = MarkdownHeaderTextSplitter(**self._params)

        chunks = splitter.split_text(text)

        self._metadata, self._chunks = zip(
            *[(chunk.metadata, chunk.page_content) for chunk in chunks]
        )
        self._metadata = list(self._metadata)
        self._chunks = list(self._chunks)


class SemanticSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("semantic_splitter", params)

    def split_text(self, text: str) -> None:
        embeddings = OpenAIEmbeddings(
            model=self._params.get("embedding_model", "text-embedding-3-small")
        )

        self._params["embeddings"] = embeddings

        splitter = SemanticChunker(**self._params)

        self._chunks = splitter.split_text(text)
        self._metadata = [{}] * len(self._chunks)
