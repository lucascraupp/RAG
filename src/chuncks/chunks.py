from supabase import create_client

from config import Config, logger
from src.chuncks.text_splitters import (
    character_text_splitter,
    markdown_header_metadata_splitter,
    recursive_character_text_splitter,
    recursive_text_splitter,
    semantic_splitter,
)


def get_best_method(filename: str) -> None:
    supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    response = (
        supabase.table("documents")
        .select("content")
        .eq("file_name", filename)
        .execute()
    )

    if not response.data:
        logger.info(f"Documento {filename} não encontrado na base de dados")
        return

    text_chunks = character_text_splitter(
        response.data[0]["content"], filename=filename
    )
    recursive_text_chunks = recursive_text_splitter(
        response.data[0]["content"], filename=filename
    )
    recursive_character_chunks = recursive_character_text_splitter(
        response.data[0]["content"], filename=filename
    )
    markdown_header_chunks = markdown_header_metadata_splitter(
        response.data[0]["content"], filename=filename
    )
    semantic_chunks = semantic_splitter(response.data[0]["content"], filename=filename)
