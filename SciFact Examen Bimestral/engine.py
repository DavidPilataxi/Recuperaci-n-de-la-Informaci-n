"""Recuperación léxica sobre documentos completos del corpus SciFact (BEIR).

Descarga automática del corpus, índice invertido, TF-IDF/coseno y BM25.
La unidad recuperada y evaluada es siempre el documento original.
"""
import csv
import json
import math
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfTransformer

BASE = Path(__file__).resolve().parent
DATA_ROOT = BASE / 'data'
DATA = DATA_ROOT / 'scifact'
URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
REQUIRED = ('corpus.jsonl', 'queries.jsonl', 'qrels/test.tsv')
_STEM = PorterStemmer()
STOP = set(ENGLISH_STOP_WORDS)


def _ready(folder=DATA):
    return all((folder / name).is_file() and (folder / name).stat().st_size > 0 for name in REQUIRED)


def prepare_data():
    """Descarga y extrae SciFact automáticamente usando HTTPS validado.

    Un ZIP incompleto no queda en caché; los archivos se extraen en un
    directorio temporal antes de publicar la carpeta final.
    """
    if _ready():
        return DATA
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    archive = DATA_ROOT / 'scifact.zip'
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        archive.unlink(missing_ok=True)
        with tempfile.NamedTemporaryFile(dir=DATA_ROOT, suffix='.zip', delete=False) as f:
            temp_archive = Path(f.name)
        try:
            with urlopen(URL, timeout=90) as response, open(temp_archive, 'wb') as output:
                shutil.copyfileobj(response, output)
            if not zipfile.is_zipfile(temp_archive):
                raise ValueError('La descarga de SciFact no es un archivo ZIP válido')
            temp_archive.replace(archive)
        finally:
            temp_archive.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(dir=DATA_ROOT) as work:
        staging = Path(work)
        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad is not None:
                raise ValueError(f'ZIP SciFact dañado: {bad}. Borre data/scifact.zip y reintente.')
            # Se permiten únicamente archivos/directorios dentro de scifact/.
            for info in z.infolist():
                p = Path(info.filename.replace('\\', '/'))
                if not p.parts or p.parts[0] != 'scifact' or p.is_absolute() or '..' in p.parts:
                    raise ValueError(f'Ruta inesperada en ZIP: {info.filename}')
                if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError('El ZIP no puede contener enlaces simbólicos')
            z.extractall(staging)
        folder = staging / 'scifact'
        if not _ready(folder):
            raise ValueError('SciFact incompleto: faltan corpus, queries o qrels/test.tsv')
        if DATA.exists():
            shutil.rmtree(DATA)
        shutil.move(str(folder), str(DATA))
    return DATA


def tokenize(text):
    """Minúsculas, tokens alfabéticos en inglés, stopwords y Porter stemming.

    Se conserva este criterio para reproducir los resultados ya evaluados.
    Las cifras y símbolos se omiten; es una limitación documentada.
    """
    return [_STEM.stem(t) for t in re.findall(r'[a-z]+', text.lower())
            if t not in STOP and len(t) > 1]


def jsonl(path):
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class SearchEngine:
    def __init__(self, folder=None):
        folder = Path(folder) if folder is not None else prepare_data()
        self.docs = list(jsonl(folder / 'corpus.jsonl'))
        if not self.docs:
            raise ValueError('El corpus SciFact está vacío')
        self.ids = [str(doc['_id']) for doc in self.docs]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('Hay IDs originales duplicados en el corpus')
        self.tokens = [tokenize((doc.get('title') or '') + ' ' + (doc.get('text') or ''))
                       for doc in self.docs]
        self.vocab = {}
        self.inverted = defaultdict(list)  # término -> [(índice de documento, frecuencia), ...]
        rows, cols, vals = [], [], []
        for i, tokens in enumerate(self.tokens):
            for term, tf in Counter(tokens).items():
                col = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i)
                cols.append(col)
                vals.append(tf)
                self.inverted[term].append((i, tf))
        self.counts = csr_matrix(
            (np.array(vals, dtype=np.float64), (rows, cols)),
            shape=(len(self.docs), len(self.vocab)), dtype=np.float64)
        self.tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
        self.tfidf = self.tfidf_transformer.fit_transform(self.counts)
        self.lengths = np.array([len(tokens) for tokens in self.tokens], dtype=float)
        self.avgdl = float(np.mean(self.lengths))
        if self.avgdl == 0:
            raise ValueError('Ningún documento contiene términos indexables')
        df = np.asarray((self.counts > 0).sum(axis=0)).ravel()
        self.bm25_idf = np.log(1 + (len(self.docs) - df + 0.5) / (df + 0.5))

    def search(self, query, model='bm25', top_k=10):
        if model not in ('bm25', 'tfidf'):
            raise ValueError('Modelo no válido')
        if not isinstance(query, str) or not query.strip():
            raise ValueError('La consulta no puede estar vacía')
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError('top_k debe estar entre 1 y 100')
        terms = Counter(t for t in tokenize(query) if t in self.vocab)
        if not terms:
            return []
        if model == 'tfidf':
            qcols = [self.vocab[t] for t in terms]
            query_counts = csr_matrix(
                (list(terms.values()), ([0] * len(qcols), qcols)),
                shape=(1, len(self.vocab)), dtype=np.float64)
            q = self.tfidf_transformer.transform(query_counts)
            scores = np.asarray((self.tfidf @ q.T).toarray()).ravel()
        else:
            scores = np.zeros(len(self.docs), dtype=float)
            k1, b = 1.5, 0.75
            for term in terms:
                posting = self.inverted[term]
                indices = np.fromiter((i for i, _ in posting), dtype=int)
                tf = np.fromiter((frequency for _, frequency in posting), dtype=float)
                col = self.vocab[term]
                scores[indices] += (self.bm25_idf[col] * tf * (k1 + 1) /
                                    (tf + k1 * (1 - b + b * self.lengths[indices] / self.avgdl)))
        candidates = np.flatnonzero(scores > 0)
        ranked = sorted(candidates, key=lambda i: (-scores[i], self.ids[i]))[:top_k]
        return [
            {'rank': rank, 'document_id': self.ids[i], 'score': float(scores[i]),
             'title': self.docs[i].get('title') or '',
             'snippet': (self.docs[i].get('text') or '')[:240]}
            for rank, i in enumerate(ranked, 1)
        ]


def load_test(folder=None):
    folder = Path(folder) if folder is not None else prepare_data()
    queries = {str(doc['_id']): doc['text'] for doc in jsonl(folder / 'queries.jsonl')}
    qrels = defaultdict(dict)
    with open(folder / 'qrels' / 'test.tsv', encoding='utf-8') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            qrels[str(row['query-id'])][str(row['corpus-id'])] = int(row['score'])
    return queries, qrels


def evaluate(engine, model, k=10, folder=None):
    """Macro-promedio de P@k, Recall@k y NDCG@k sobre test con qrels.

    Cada documento con relevancia >0 cuenta como relevante para P/Recall;
    NDCG utiliza las calificaciones originales del qrel.
    """
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
        raise ValueError('k debe estar entre 1 y 100')
    queries, qrels = load_test(folder)
    if not qrels:
        raise ValueError('No hay qrels test para evaluar')
    missing = set(qrels) - set(queries)
    if missing:
        raise ValueError(f'Faltan {len(missing)} consultas test en queries.jsonl')
    values = []
    keys = (f'precision@{k}', f'recall@{k}', f'ndcg@{k}')
    for qid, relevance in qrels.items():
        found = engine.search(queries[qid], model, k)
        gains = [relevance.get(doc['document_id'], 0) for doc in found]
        relevant_total = sum(grade > 0 for grade in relevance.values())
        precision = sum(grade > 0 for grade in gains) / k
        recall = sum(grade > 0 for grade in gains) / relevant_total if relevant_total else 0.0
        dcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(gains))
        ideal = sorted(relevance.values(), reverse=True)[:k]
        idcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
        values.append({'query_id': qid, 'query': queries[qid], keys[0]: precision,
                       keys[1]: recall, keys[2]: dcg / idcg if idcg else 0.0})
    metrics = {key: sum(row[key] for row in values) / len(values) for key in keys}
    return metrics, values
