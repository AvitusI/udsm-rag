from langchain.text_splitter import (
    CharacterTextSplitter
)
from langchain_core.documents.base import Document
from langchain_community.document_loaders import (
    DirectoryLoader
)
from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


async def load_documents(
    db: PineconeVectorStore
):
    text_splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0
    )

    raw_documents = DirectoryLoader(
        "docs", "*.txt"
    ).load()

    chunks = text_splitter.split_documents(raw_documents)

    await db.aadd_documents(chunks)


def get_context(
    user_query: str,
    db: PineconeVectorStore
) -> str:
    # retriever = db.as_retriever()

    docs = db.similarity_search(user_query)

    return "\n\n".join(
        doc.page_content for doc in docs
    )