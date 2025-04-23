import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Body, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_aws import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from documents import load_documents, get_context
from model import chain_with_message_history
from config import BaseConfig


load_dotenv()

settings = BaseConfig()

pc = Pinecone(api_key=settings.PINECONE_API_KEY_SECRET)

index_name = "fastapi-rag"

index = pc.Index(index_name)

embeddings = BedrockEmbeddings(
    region_name=settings.AWS_REGION,
    model_id="amazon.titan-embed-image-v1",
)

# Chroma(embedding_function=CohereEmbeddings(model="embed-english-light-v3.0"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    db = PineconeVectorStore(
        index=index, embedding=embeddings
    )
    await load_documents(db)
    yield {"db": db}


app = FastAPI(
    title="Ecotech AI Assistant",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allowed origins
    allow_credentials=True,  # Allow cookies and authentication headers
    allow_methods=["*"],  # Allowed HTTP methods: GET, POST, etc.
    allow_headers=["*"],  # Allowed HTTP headers
)

@app.post("/message")
async def query_assistant(
    request: Request,
    question: Annotated[str, Body()]
) -> str:
    context = get_context(question, request.state.db)

    async def response_stream():
        async for chunk in chain_with_message_history.astream(
            {
                "context": context,
                "question": question,
            },
            config={
                "configurable": {
                    "session_id": "any"
                }
            }
        ):
            yield chunk
        
    return StreamingResponse(response_stream(), media_type='text/event-stream')
    """
    response = await chain.ainvoke(
        {
            "question": question,
            "context": context,
        }
    )

    return response
    ""

""
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allowed origins
    allow_credentials=True,  # Allow cookies and authentication headers
    allow_methods=["*"],  # Allowed HTTP methods: GET, POST, etc.
    allow_headers=["*"],  # Allowed HTTP headers
)

async def stream_response(text: str):
    for i in range(10):  # Simulate streaming
        yield f"Processing '{text}': Chunk {i}\n"
        await asyncio.sleep(1)  # Simulate delay

@app.post("/stream")
async def stream(request: Request):
    text = await request.body()
    text = text.decode("utf-8")  # Decode bytes to string
    return StreamingResponse(stream_response(text), media_type="text/event-stream")
"""