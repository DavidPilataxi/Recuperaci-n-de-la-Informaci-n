# SciFact — Examen ICCD753 Recuperación de Información

Buscador sobre **documentos completos** del corpus SciFact (5.183 documentos en la ejecución comprobada), con identificadores originales, tokenización/stopwords/Porter stemming, vocabulario, índice invertido con frecuencias, TF-IDF + similitud coseno y BM25. Incluye API FastAPI y notebook ejecutado con evaluación de la partición `test` de SciFact.

## Archivos

- `engine.py`: descarga del corpus SciFact de BEIR, indexación, modelos y evaluación.
- `main.py`: API HTTP sin interfaz web; consumible desde Postman.
- `Evaluacion_SciFact.ipynb`: análisis, pruebas y métricas del conjunto `test`.
- `test_project.py`: pruebas automáticas con corpus pequeño **sintético** y sin Internet.
- `requirements.txt` y `render.yaml`: dependencias y despliegue.

## Ejecutar en Windows (sin entorno virtual)

En PowerShell, dentro de la carpeta donde están `main.py` y `engine.py`:

```powershell
python --version
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Verifique `http://127.0.0.1:8000/health` (GET). Las rutas `/`, `/docs` y `/openapi.json` están deshabilitadas deliberadamente porque la revisión se realiza exclusivamente vía Postman.

### Prueba de búsqueda (Postman)

- Método: `POST`
- URL: `http://127.0.0.1:8000/search`
- Cabecera: `Content-Type: application/json`
- Cuerpo:

```json
{"query":"Does vitamin D reduce respiratory infections?","model":"bm25","top_k":5}
```

Cambie `model` a `tfidf`. Se devuelven `rank`, `document_id` original, `score`, `title` y `snippet`. Si no hay coincidencias se devuelve `results: []`. Las consultas no están precargadas: se aceptan consultas arbitrarias. `top_k` admite enteros del 1 al 100.

## Descarga automática y reproducción

Al iniciar, si no están `data/scifact/corpus.jsonl`, `data/scifact/queries.jsonl` y `data/scifact/qrels/test.tsv`, se descarga automáticamente desde **BEIR SciFact** por HTTPS con certificados válidos y se extrae a `data/scifact/`. No hace falta descargarlo manualmente. El ZIP se guarda en `data/scifact.zip` como caché local. La descarga exige conexión y acceso HTTPS al servidor de BEIR. Si un ZIP está dañado, el programa informa del error: elimine `data/scifact.zip` para repetir la descarga.

El dataset es SciFact distribuido por BEIR, equivalente al corpus asignado en la consigna con enlaces a Hugging Face; se respetan `_id` y qrels originales. Los documentos no se dividen en fragmentos.

**Preprocesamiento reproducible:** minúsculas, expresión regular `[a-z]+`, eliminación de stopwords en inglés de scikit-learn y stemming Porter. Se ignoran números y símbolos: es una limitación para textos científicos. **Modelos:** TF-IDF IDF suavizado con normalización L2 y coseno; BM25 con `k1=1.5`, `b=0.75`, IDF no negativa de Robertson/Sparck Jones modificada. BM25 recorre el índice invertido término → (documento, frecuencia); TF-IDF opera sobre una matriz dispersa formada a partir de las mismas frecuencias.

### Notebook

Abra `Evaluacion_SciFact.ipynb` desde la carpeta del proyecto en VS Code, seleccione el **mismo Python** que ejecuta la API y use **Run All**. Si VS Code pide un kernel, instale `python -m pip install ipykernel`. El archivo incluido conserva las salidas reales compartidas por el estudiante para las celdas principales; la nueva celda de consistencia queda **sin ejecutar** hasta que se vuelva a correr el notebook. Guarde de nuevo antes de entregar para que cada celda tenga su salida.

Resultados de la ejecución entregada sobre 300 consultas test con qrels:

| Métrica | TF-IDF | BM25 |
|:--|--:|--:|
| Precision@10 | 0.082667 | 0.092333 |
| Recall@10 | 0.755889 | 0.838667 |
| NDCG@10 | 0.597829 | 0.691850 |

La interpretación ampliada está dentro del notebook. Estos valores provienen de la ejecución adjunta del estudiante; **no se recalcularon aquí sobre el corpus completo**. Las correcciones preservan el preprocesamiento y las fórmulas de puntuación, pero deben verificarse con una nueva ejecución local.

### Pruebas automatizadas sin descargar SciFact

```powershell
python -m pip install pytest httpx
python -m pytest -q test_project.py
```

La prueba debe indicar `7 passed`. Emplea un corpus sintético reducido y revisa índice, rankings, parámetros, fórmulas BM25, evaluación y API. No reemplaza la evaluación real con los 300 qrels.

## Publicación en Render

1. Cree un repositorio GitHub con `engine.py`, `main.py`, `requirements.txt`, `render.yaml`, notebook, pruebas y este README. No suba `data/` (ver `.gitignore`).
2. En [Render](https://dashboard.render.com/) seleccione **New → Blueprint** y conecte el repositorio, o cree un Web Service con `pip install -r requirements.txt` y `uvicorn main:app --host 0.0.0.0 --port $PORT`.
3. Observe los logs: el primer arranque descarga SciFact y construye índices; necesitará recursos y memoria suficientes. El plan gratuito **no garantiza** que funcione sin límites de memoria/tiempo ni que esté siempre despierto. Si falla, revise Logs y ajuste el plan o el despliegue.
4. Compruebe públicamente `https://SU-SERVICIO.onrender.com/health`  y una solicitud `POST /search` desde una red externa.
5. Envíe al docente la URL real verificada. **La entrega exige que esté accesible durante el periodo de revisión**.

## Limitaciones y trazabilidad

Los juicios de relevancia disponibles determinan la evaluación: documentos sin qrel se consideran no relevantes para las métricas, lo cual no equivale a una revisión humana exhaustiva. Las calificaciones positivas se consideran relevantes en Precision/Recall y las calificaciones originales se emplean en NDCG. Se promedian métricas por consulta (macro) y se consideran solo consultas `test` con qrels. Los índices se reconstruyen al reiniciar; no se guardan archivos pickle. No incluye una interfaz adicional de búsqueda: los endpoints se consultan desde Postman, sin frontend ni Swagger. La URL pública no se ha desplegado ni validado aún.

## Verificación sin interfaz web

La aplicación expone solamente `GET /health` y `POST /search`. La ruta raíz, Swagger (`/docs`), ReDoc y OpenAPI JSON están deshabilitados para cumplir la revisión exclusiva desde Postman. El servicio sigue siendo HTTP accesible públicamente; el término **Web Service** de Render no implica tener un frontend.
