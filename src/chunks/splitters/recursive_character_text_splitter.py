import logging
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunks.splitters.text_splitter import TextSplitter


class CustomRecursiveCharacterTextSplitter(TextSplitter):
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
