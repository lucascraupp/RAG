from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from supabase import create_client

from config import Config, logger


def save_document(filename: str, content: str) -> None:
    logger.info(f"Salvando documento {filename} na base de dados")
    supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

    response = (
        supabase.table("documents")
        .select("filename")
        .eq("filename", filename)
        .execute()
    )

    if response.data:
        logger.info(f"Documento {filename} já existe na base de dados")
        return

    supabase.table("documents").insert(
        {"filename": filename, "content": content, "file_size": len(content)}
    ).execute()

    logger.info(f"Documento {filename} salvo com sucesso")


def extract_markdown(filename: str) -> None:
    pipeline_options = PdfPipelineOptions(do_table_structure=True, do_ocr=False)
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    logger.info(f"Convertendo documento {filename}")
    result = converter.convert(f"docs/{filename}")

    markdown_content = result.document.export_to_markdown()

    save_document(filename, markdown_content)
