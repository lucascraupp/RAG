import copy
from typing import Dict, List, Optional

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from src.chunks.splitters.recursive_character_text_splitter import (
    CustomRecursiveCharacterTextSplitter as RecursiveCharacterTextSplitter,
)
from src.chunks.splitters.text_splitter import TextSplitter


class SemanticSplitter(TextSplitter):
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

        recursive_splitter = RecursiveCharacterTextSplitter(recursive_params)

        chunks = []
        for chunk in recursive_splitter.split_text(text):
            splitter = SemanticChunker(**semantic_params)
            test = splitter.split_text(chunk)
            chunks.extend(test)

        self._chunks_data = self._add_headers_to_chunks(
            text, chunks, filename, semantic_params.get("chunk_overlap", 0)
        )

        return chunks
