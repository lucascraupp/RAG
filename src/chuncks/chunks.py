from supabase import create_client

from config import Config, logger
from src.chuncks.text_splitters import (
    character_text_splitter,
    markdown_header_metadata_splitter,
    recursive_character_text_splitter,
    recursive_text_splitter,
    semantic_splitter,
)


def get_best_method(file_name: str) -> None:
    supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    response = (
        supabase.table("documents")
        .select("content")
        .eq("file_name", file_name)
        .execute()
    )

    if not response.data:
        logger.info(f"Documento {file_name} não encontrado na base de dados")
        return

    text_chunks = character_text_splitter(response.data[0]["content"])
    recursive_text_chunks = recursive_text_splitter(response.data[0]["content"])
    recursive_character_chunks = recursive_character_text_splitter(
        response.data[0]["content"]
    )
    markdown_header_splitter = markdown_header_metadata_splitter(
        response.data[0]["content"]
    )
