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
        supabase.table("documents").select("content").eq("filename", filename).execute()
    )

    if not response.data:
        logger.info(f"Documento {filename} não encontrado na base de dados")
        return

    splitters = [
        character_text_splitter,
        recursive_text_splitter,
        recursive_character_text_splitter,
        markdown_header_metadata_splitter,
        semantic_splitter,
    ]

    for splitter in splitters:
        data = (
            supabase.table("chunks")
            .select("*")
            .eq("strategy", splitter.__name__)
            .eq("filename", filename)
            .execute()
        )

        if data.data:
            logger.info(f"Chunks já existem para o método '{splitter.__name__}'")
        else:
            logger.info(f"Criando chunks usando o método '{splitter.__name__}'")
            text_chunks = splitter(response.data[0]["content"], filename=filename)
            supabase.table("chunks").insert(text_chunks).execute()
            logger.info(f"Chunks criados")
