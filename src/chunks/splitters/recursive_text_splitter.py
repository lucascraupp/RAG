from typing import Dict, List, Optional

from src.chunks.splitters import CharacterTextSplitter, TextSplitter


class RecursiveTextSplitter(TextSplitter):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("recursive_text_splitter", params)

    def split_text(self, text: str, filename: str) -> List[str]:
        splitted_params = self._params.copy()
        splitted_params.pop("separators", None)

        chunks = [text]
        for separator in self._params.get("separators"):
            splitted_params["separator"] = separator

            text_splitter = CharacterTextSplitter(splitted_params)

            new_chunks = []
            for text in chunks:
                new_chunks.extend(text_splitter.split_text(text, filename))
        chunks = new_chunks

        self._chunks_data = self._add_headers_to_chunks(
            text, chunks, filename, self._params.get("chunk_overlap", 0)
        )

        return chunks
