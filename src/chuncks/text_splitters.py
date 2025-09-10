import logging
from typing import Dict, List, Optional, Union

import yaml
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
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


def generate_embeddings(
    filename: str, chunks: List[str], params: Dict, strategy: str
) -> Dict:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    data = []
    for chunk in chunks:
        data.append(
            {
                "filename": filename,
                "strategy": strategy,
                "strategy_params": params,
                "chunk_content": chunk,
                "chunk_size": len(chunk),
                "chunk_embeddings": embedding.embed_query(chunk),
            }
        )
    return data


def character_text_splitter(
    text: str,
    filename: str,
    params: Optional[Dict] = None,
    embeddings: Optional[bool] = True,
) -> Union[Dict, List[str]]:
    """Divide o texto em pedaços menores usando um separador de caracteres.

    Args:
        text (str): O texto a ser dividido em pedaços menores.
        params (Optional[Dict], optional): Dicionário com parâmetros para divisão
            como 'chunk_size', 'chunk_overlap' e 'separator'. Se não fornecido,
            carrega os parâmetros do arquivo de configuração.

    Returns:
        Union[Dict[str, Union[List[str], Dict]], List[str]]: Um dicionário contendo os pedaços de texto e os parâmetros utilizados
    """
    data = params or load_data("character_text_splitter")

    splitter = CharacterTextSplitter(**data)

    logger = logging.getLogger("langchain_text_splitters.base")
    original_level = logger.level

    try:
        logger.setLevel(logging.ERROR)

        chunks = splitter.split_text(text)
    finally:
        logger.setLevel(original_level)

    if embeddings:
        return generate_embeddings(filename, chunks, data, "character_text_splitter")
    else:
        return chunks


def recursive_text_splitter(
    text: str, filename: str, params: Optional[Dict] = None
) -> Dict:
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
       Dict: Um dicionário contendo os pedaços de texto e os parâmetros utilizados.
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
            next_chunks.extend(
                character_text_splitter(chunk, split_params, embeddings=False)
            )

        chunks = next_chunks

    return generate_embeddings(filename, chunks, data, "recursive_text_splitter")


def recursive_character_text_splitter(
    text: str, filename: str, params: Optional[Dict] = None
) -> Dict:
    """Divide o texto em pedaços menores usando separadores de cabeçalho Markdown.

    Se o tamanho do chunk superar o limite especificado, ele será dividido ainda mais, usando os separadores definidos.

    Args:
        text (str): O texto a ser dividido em pedaços menores.
        params (Optional[Dict], optional): Dicionário com parâmetros para divisão
            como 'chunk_size', 'chunk_overlap' e 'separators'. Se não fornecido,
            carrega os parâmetros do arquivo de configuração.

    Returns:
        Dict: Um dicionário contendo os pedaços de texto e os parâmetros utilizados.
    """
    data = params or load_data("recursive_character_text_splitter")

    modify_data = data.copy()

    # Converter length_function de string para função real
    if isinstance(modify_data, dict) and "length_function" in modify_data:
        if modify_data["length_function"] == "len":
            modify_data["length_function"] = len

    splitter = RecursiveCharacterTextSplitter(**modify_data)

    chunks = splitter.split_text(text)

    return generate_embeddings(
        filename, chunks, data, "recursive_character_text_splitter"
    )


def markdown_header_metadata_splitter(
    text: str, filename: str, params: Optional[Dict] = None
) -> Dict:
    """Divide o texto em pedaços menores usando separadores de cabeçalho Markdown.

    Args:
        text (str): O texto a ser dividido em pedaços menores.
        params (Optional[Dict], optional): Dicionário com parâmetros para divisão
            como 'headers_to_split_on', 'return_each_line' e 'strip_headers'. Se não fornecido,
            carrega os parâmetros do arquivo de configuração.

    Returns:
        Dict: Um dicionário contendo os pedaços de texto e os parâmetros utilizados.
    """
    data = params or load_data("markdown_header_metadata_splitter")

    splitter = MarkdownHeaderTextSplitter(**data)

    chunks = splitter.split_text(text)

    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    embedded_chunks = []
    for chunk in chunks:
        embedded_chunks.append(
            {
                "filename": filename,
                "strategy": "markdown_header_metadata_splitter",
                "strategy_params": data,
                "metadata": chunk.metadata,
                "chunk_content": chunk.page_content,
                "chunk_size": len(chunk.page_content),
                "chunk_embeddings": embedding.embed_query(chunk.page_content),
            }
        )
    return embedded_chunks


def semantic_splitter(text: str, filename: str, params: Optional[Dict] = None) -> Dict:
    """Divide o texto em pedaços menores usando separadores semânticos.

    Args:
        text (str): O texto a ser dividido em pedaços menores.
        params (Optional[Dict], optional): Dicionário com parâmetros para divisão
            como 'embeddings' e 'buffer_size'. Se não fornecido,
            carrega os parâmetros do arquivo de configuração.

    Returns:
        Dict[str, Union[List[str], Dict]]: Um dicionário contendo os pedaços de texto e os parâmetros utilizados.
    """
    data = params or load_data("semantic_splitter")

    embeddings = OpenAIEmbeddings(
        model=data.get("embedding_model", "text-embedding-3-small")
    )

    data_dict = {
        "embeddings": embeddings,
        "buffer_size": data.get("buffer_size", 1),
        "breakpoint_threshold_type": data.get(
            "breakpoint_threshold_type", "percentile"
        ),
        "breakpoint_threshold_amount": data.get("breakpoint_threshold_amount", 95),
    }

    splitter = SemanticChunker(**data_dict)

    chunks = splitter.split_text(text)

    return generate_embeddings(filename, chunks, data, "semantic_splitter")
