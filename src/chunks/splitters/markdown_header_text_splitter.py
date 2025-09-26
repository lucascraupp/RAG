from typing import Dict, Optional

import pandas as pd
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.chunks.splitters import TextSplitter


class CustomMarkdownHeaderTextSplitter(TextSplitter):
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
