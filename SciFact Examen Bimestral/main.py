"""API REST SciFact para clientes HTTP como Postman; sin interfaz web."""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from engine import SearchEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = SearchEngine()  # Descarga/indexa antes de servir peticiones
    yield
    app.state.engine = None


app = FastAPI(title='SciFact - Recuperación de Información', version='1.0.0', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description='Consulta de texto libre')
    model: Literal['bm25', 'tfidf'] = 'bm25'
    top_k: int = Field(default=10, ge=1, le=100)


@app.get('/health')
def health(request: Request):
    engine = request.app.state.engine
    return {'status': 'ok', 'corpus': 'scifact', 'documents': len(engine.docs)}


@app.post('/search')
def search(request: Request, body: SearchRequest):
    engine = request.app.state.engine
    return {'model': body.model,
            'results': engine.search(body.query, body.model, body.top_k)}
