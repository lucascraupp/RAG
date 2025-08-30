from typing import Dict, List, Optional

import yaml
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def load_data(model: str) -> Dict[str, str]:
    """Carrega os parâmetros do modelo a partir de um arquivo YAML.

    Args:
        model (str): O nome do modelo a ser carregado.

    Returns:
        Dict[str, str]: Os parâmetros para o modelo especificado.
    """
    with open(f"src/chuncks/parameters.yaml", "r") as file:
        data = yaml.safe_load(file)
    return data[model]


def character_text_splitter(text: str, params: Optional[Dict] = None) -> List[str]:
    """Divide o texto em pedaços menores usando um separador de caracteres.

    Args:
        text (str): O texto a ser dividido em pedaços menores.
        params (Optional[Dict], optional): Dicionário com parâmetros para divisão
            como 'chunk_size', 'chunk_overlap' e 'separator'. Se não fornecido,
            carrega os parâmetros do arquivo de configuração.

    Returns:
        List[str]: Lista de pedaços de texto após a divisão.
    """
    data = params or load_data("character_text_splitter")

    splitter = CharacterTextSplitter(**data)

    return splitter.split_text(text)


def recursive_text_splitter(text: str, params: Optional[Dict] = None) -> List[str]:
    """Divide recursivamente o texto em pedaços menores usando múltiplos separadores.

    Esta função implementa uma divisão de texto em múltiplos passos, aplicando diferentes
    separadores em sequência. Primeiro divide pelos separadores mais amplos (como parágrafos)
    e depois por separadores mais específicos (como linhas ou espaços).

    Args:
        text (str): O texto a ser dividido em pedaços menores.
        params (Optional[Dict], optional): Dicionário com parâmetros para divisão
            como 'chunk_size', 'chunk_overlap' e 'separators'. Se não fornecido,
            carrega os parâmetros do arquivo de configuração.

    Returns:
        List[str]: Lista de pedaços de texto após a divisão recursiva.
    """
    data = params or load_data("recursive_text_splitter")

    chunk_size = data.get("chunk_size", 500)
    chunk_overlap = data.get("chunk_overlap", 100)
    separators = data.get("separators", ["\n\n", "\n"])

    chunks = [text]
    for separator in separators:
        next_chunks = []
        split_params = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separator": separator,
        }

        for chunk in chunks:
            next_chunks.extend(character_text_splitter(chunk, split_params))

        chunks = next_chunks

    return chunks


def recursive_character_text_splitter(
    text: str, params: Optional[Dict] = None
) -> List[str]:
    """Divide o texto em pedaços menores usando separadores de cabeçalho Markdown.

    Se o tamanho do chunk superar o limite especificado, ele será dividido ainda mais, usando os separadores definidos.

    Args:
        text (str): O texto a ser dividido em pedaços menores.
        params (Optional[Dict], optional): Dicionário com parâmetros para divisão
            como 'chunk_size', 'chunk_overlap' e 'separators'. Se não fornecido,
            carrega os parâmetros do arquivo de configuração.

    Returns:
        List[str]: Lista de pedaços de texto após a divisão.
    """
    data = params or load_data("recursive_character_text_splitter")

    # Converter length_function de string para função real
    if isinstance(data, dict) and "length_function" in data:
        if data["length_function"] == "len":
            data["length_function"] = len

    data = params or load_data("recursive_character_text_splitter")

    # Converter length_function de string para função real
    if isinstance(data, dict) and "length_function" in data:
        if data["length_function"] == "len":
            data["length_function"] = len

    splitter = RecursiveCharacterTextSplitter(**data)

    return splitter.split_text(text)


def markdown_header_metadata_splitter(
    text: str, params: Optional[Dict] = None
) -> List[str]:
    """Divide o texto em pedaços menores usando separadores de cabeçalho Markdown.

    Args:
        text (str): O texto a ser dividido em pedaços menores.
        params (Optional[Dict], optional): Dicionário com parâmetros para divisão
            como 'chunk_size', 'chunk_overlap' e 'separators'. Se não fornecido,
            carrega os parâmetros do arquivo de configuração.

    Returns:
        List[str]: Lista de pedaços de texto após a divisão.
    """
    data = params or load_data("markdown_header_metadata_splitter")

    splitter = MarkdownHeaderTextSplitter(**data)

    return splitter.split_text(text)


def semantic_splitter(text: str, params: Optional[Dict] = None) -> List[str]:
    """
    Splits the input text into chunks based on semantic similarity.

    Args:
        text (str): The input text to split.
        params (Optional[Dict]): Optional parameters for splitting.

    Returns:
        List[str]: A list of text chunks.
    """
    # Implementation goes here
    pass
