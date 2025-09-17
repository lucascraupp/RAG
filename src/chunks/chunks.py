from supabase import create_client

from config import Config, logger
from src.chunks.text_splitters import (
    CharacterTextSplitters,
    MarkdownHeaderMetadataSplitters,
    RecursiveCharacterTextSplitters,
    RecursiveTextSplitters,
    SemanticSplitters,
)


def get_chunks(filename: str) -> None:
    supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    response = (
        supabase.table("documents").select("content").eq("filename", filename).execute()
    )

    if not response.data:
        logger.info(f"Documento {filename} não encontrado na base de dados")
        return

    splitters = [
        CharacterTextSplitters,
        RecursiveTextSplitters,
        RecursiveCharacterTextSplitters,
        MarkdownHeaderMetadataSplitters,
        SemanticSplitters,
    ]

    for splitter in splitters:
        method = splitter()

        data = (
            supabase.table("chunks")
            .select("*")
            .eq("strategy", method.model_name)
            .eq("filename", filename)
            .execute()
        )

        if data.data:
            logger.info(f"Chunks já existem para o método '{method.model_name}'")
        else:
            logger.info(f"Criando chunks usando o método '{method.model_name}'")
            method.split_text(response.data[0]["content"])
            chunks = method.generate_embeddings(filename)

            supabase.table("chunks").insert(chunks).execute()
            logger.info(f"Chunks criados")
