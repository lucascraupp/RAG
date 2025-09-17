from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from supabase import create_client

from config import Config

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")


def retrieve_information(method: str, params: Dict) -> str:
    supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

    response = supabase.rpc(method, params).execute()

    return "\n\n".join([chunk["content"] for chunk in response.data])


def semantic_search(
    query: str, filters: Optional[Dict] = {}, k: Optional[int] = 5
) -> List[Document]:
    params = {
        "query_embedding": EMBEDDINGS.embed_query(query),
        "match_count": k,
        "filter": filters,
    }

    return retrieve_information("semantic_search", params)


def hybrid_search(
    query: str, filters: Optional[Dict] = {}, k: Optional[int] = 5
) -> str:
    params = {
        "query_text": query,
        "query_embedding": EMBEDDINGS.embed_query(query),
        "match_count": k,
        "filter": filters,
    }

    return retrieve_information("hybrid_search", params)
