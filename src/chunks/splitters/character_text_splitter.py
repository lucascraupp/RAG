import logging
from typing import Dict, List, Optional

from langchain_text_splitters import CharacterTextSplitter

from src.chunks.splitters.text_splitter import TextSplitter


class CustomCharacterTextSplitter(TextSplitter):
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
