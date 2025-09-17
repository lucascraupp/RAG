import copy
import logging
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


class TextSplitter(ABC):
    def __init__(self, model_name: str, params: Optional[Dict] = None):
        self._model_name: str = model_name
        self._params: Dict = params or self.__load_params(model_name)
        self._chunks_data: pd.DataFrame = pd.DataFrame()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def chunks(self) -> List[str]:
        return self._chunks_data["chunk"].tolist()

    def __load_params(self, model_name: str) -> Dict[str, str]:
        with open(f"src/chunks/parameters.yaml", "r") as file:
            params = yaml.safe_load(file)
        return params[model_name]

    def generate_embeddings(self, filename: str) -> List[Dict]:
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")

        data = []
        for row in self._chunks_data.itertuples():
            chunk = row.chunk
            row.metadata["filename"] = filename
            data.append(
                {
                    "filename": filename,
                    "strategy": self._model_name,
                    "strategy_params": self._params,
                    "chunk_metadata": row.metadata,
                    "chunk_content": chunk,
                    "chunk_size": len(chunk),
                    "chunk_embeddings": embedding.embed_query(chunk),
                }
            )
        return data

    @abstractmethod
    def split_text(self, text: str) -> Union[List[str], Dict]:
        pass


class CharacterTextSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("character_text_splitter", params)

    def split_text(self, text: str) -> List[str]:
        splitter = CharacterTextSplitter(**self._params)

        logger = logging.getLogger("langchain_text_splitters.base")
        original_level = logger.level

        try:
            logger.setLevel(logging.ERROR)

            chunks = splitter.split_text(text)

            self._chunks_data = pd.DataFrame(
                {
                    "metadata": [{"strategy": self._model_name}] * len(chunks),
                    "chunk": chunks,
                }
            )
        finally:
            logger.setLevel(original_level)

        return chunks


class RecursiveTextSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("recursive_text_splitter", params)

    def split_text(self, text: str) -> List[str]:
        splitted_params = self._params.copy()
        splitted_params.pop("separators", None)

        chunks = [text]
        for separator in self._params.get("separators"):
            splitted_params["separator"] = separator

            text_splitter = CharacterTextSplitters(splitted_params)

            new_chunks = []
            for text in chunks:
                new_chunks.extend(text_splitter.split_text(text))
        chunks = new_chunks

        self._chunks_data = pd.DataFrame(
            {
                "metadata": [{"strategy": self._model_name}] * len(chunks),
                "chunk": chunks,
            }
        )

        return chunks


class RecursiveCharacterTextSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("recursive_character_text_splitter", params)

    def split_text(self, text: str) -> List[str]:
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

            self._chunks_data = pd.DataFrame(
                {
                    "metadata": [{"strategy": self._model_name}] * len(chunks),
                    "chunk": chunks,
                }
            )
        finally:
            logger.setLevel(original_level)

        return chunks


class MarkdownHeaderMetadataSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("markdown_header_metadata_splitter", params)

    def split_text(self, text: str) -> Dict:
        splitter = MarkdownHeaderTextSplitter(**self._params)

        chunks = splitter.split_text(text)

        chunks_data = pd.DataFrame(
            {
                "metadata": [
                    {**chunk.metadata, "strategy": self._model_name} for chunk in chunks
                ],
                "chunk": [chunk.page_content for chunk in chunks],
            }
        )

        self._chunks_data = chunks_data

        return chunks


class SemanticSplitters(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("semantic_splitter", params)

    def split_text(self, text: str) -> None:
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

        self._chunks_data = pd.DataFrame(
            {
                "metadata": [{"strategy": self._model_name}] * len(chunks),
                "chunk": chunks,
            }
        )
