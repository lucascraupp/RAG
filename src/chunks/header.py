import re
from typing import Dict, List


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
