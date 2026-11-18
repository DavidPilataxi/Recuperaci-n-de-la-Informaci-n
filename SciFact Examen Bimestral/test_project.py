"""Pruebas sin Internet: python -m pytest -q test_project.py"""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from engine import SearchEngine, evaluate, load_test, tokenize


def make_fixture(tmp_path):
    folder = tmp_path / 'scifact'
    (folder / 'qrels').mkdir(parents=True)
    docs = [
        {'_id': 'doc-A', 'title': 'Vitamin D', 'text': 'vitamin d respiratory infection study'},
        {'_id': 'doc-B', 'title': 'Other study', 'text': 'cancer trial clinical study'},
        {'_id': 'doc-C', 'title': 'Respiratory trial', 'text': 'respiratory infection treatment'},
    ]
    with (folder / 'corpus.jsonl').open('w', encoding='utf-8') as out:
        for doc in docs:
            out.write(json.dumps(doc) + '\n')
    (folder / 'queries.jsonl').write_text(json.dumps({'_id': 'q1', 'text': 'respiratory infection'}) + '\n', encoding='utf-8')
    (folder / 'qrels' / 'test.tsv').write_text('query-id\tcorpus-id\tscore\nq1\tdoc-A\t1\nq1\tdoc-C\t1\n', encoding='utf-8')
    return folder


@pytest.fixture
def engine(tmp_path):
    return SearchEngine(make_fixture(tmp_path))


def test_index_and_original_ids(engine):
    assert engine.ids == ['doc-A', 'doc-B', 'doc-C']
    assert len(engine.inverted['respiratori']) == 2
    for term, posting in engine.inverted.items():
        col = engine.vocab[term]
        for idx, tf in posting:
            assert engine.counts[idx, col] == tf


@pytest.mark.parametrize('model', ['bm25', 'tfidf'])
def test_ranking_and_k(engine, model):
    hits = engine.search('respiratory infection', model, 2)
    assert len(hits) == 2
    assert [h['rank'] for h in hits] == [1, 2]
    assert len({h['document_id'] for h in hits}) == 2
    assert hits[0]['score'] >= hits[1]['score'] > 0
    assert all(h['document_id'] in engine.ids for h in hits)
    assert engine.search('zzzxxyyqqq', model) == []


def test_bm25_postings_match_formula(engine):
    term = 'respiratori'
    column = engine.vocab[term]
    for i, tf in engine.inverted[term]:
        expected = (engine.bm25_idf[column] * tf * 2.5 /
                    (tf + 1.5 * (0.25 + 0.75 * engine.lengths[i] / engine.avgdl)))
        # Consulta de un término; debe coincidir con BM25 teórico.
        hits = engine.search('respiratory', 'bm25', 3)
        actual = next(h['score'] for h in hits if h['document_id'] == engine.ids[i])
        assert np.isclose(actual, expected)


def test_evaluation(engine, tmp_path):
    # El fixture del motor y el qrel están en una misma carpeta.
    folder = tmp_path / 'scifact'
    metrics, rows = evaluate(engine, 'bm25', 10, folder=folder)
    assert len(rows) == 1
    assert 0 <= metrics['precision@10'] <= 1
    assert 0 <= metrics['recall@10'] <= 1
    assert 0 <= metrics['ndcg@10'] <= 1
    assert metrics['precision@10'] == pytest.approx(0.2)
    assert metrics['recall@10'] == pytest.approx(1.0)
    assert metrics['ndcg@10'] == pytest.approx(1.0)


def test_bad_parameters(engine):
    for model in ('bad', ''):
        with pytest.raises(ValueError):
            engine.search('test', model)
    for k in (0, 101, True):
        with pytest.raises(ValueError):
            engine.search('test', top_k=k)


def test_api(engine):
    # Inyecta corpus sintético para no descargar durante tests.
    async def test_lifespan(app):
        app.state.engine = engine
        yield
    from contextlib import asynccontextmanager
    main.app.router.lifespan_context = asynccontextmanager(test_lifespan)
    with TestClient(main.app) as client:
        assert client.get('/').status_code == 404
        assert client.get('/docs').status_code == 404
        assert client.get('/openapi.json').status_code == 404
        health = client.get('/health')
        assert health.status_code == 200 and health.json()['documents'] == 3
        for model in ('tfidf', 'bm25'):
            response = client.post('/search', json={'query': 'respiratory', 'model': model, 'top_k': 2})
            assert response.status_code == 200 and len(response.json()['results']) == 2
        assert client.post('/search', json={'query': 'test', 'model': 'bad'}).status_code == 422
        assert client.post('/search', json={'query': 'test', 'top_k': 0}).status_code == 422
