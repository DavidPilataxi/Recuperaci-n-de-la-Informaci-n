# SciFact — Recuperación de Información

API REST **sin interfaz web** para buscar artículos del corpus SciFact mediante **BM25** o **TF-IDF (coseno)**. Utiliza los 5.183 documentos de BEIR SciFact con sus IDs originales, preprocesamiento (stopwords y stemming Porter) e índice invertido. El corpus **se descarga automáticamente al iniciar**

**API publicada:** `https://scifact-buscador.onrender.com`  
**Revisión:** exclusivamente mediante **Postman** (no hay página de inicio ni Swagger).

## Ejecutar en Windows (VS Code / PowerShell)

1. Instala Python 3.12 y abre en VS Code la carpeta que contiene `main.py`, `engine.py` y `requirements.txt`.
2. En la terminal de esa carpeta, instala dependencias e inicia el servidor:

   ```powershell
   python --version
   python -m pip install -r requirements.txt
   python -m uvicorn main:app --reload
   ```

3. Deja abierta la terminal. En el primer arranque, Python descarga SciFact y crea los índices. Cuando aparezca `Application startup complete`, la API local estará en `http://127.0.0.1:8000`.

**Si aparece `SSL: CERTIFICATE_VERIFY_FAILED` durante la descarga:** detén el servidor (`Ctrl+C`) y ejecuta:

```powershell
python -m pip install --upgrade certifi
python -m pip install pip-system-certs
python -m uvicorn main:app --reload
```

Estos paquetes ayudan a validar certificados HTTPS usando las autoridades de confianza del sistema; **no desactivan la seguridad SSL**. Si persiste el error, revisa la red o certificados del equipo; no uses `verify=False`.

> **Nota sobre PowerShell:** `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` fue mencionado durante las pruebas para permitir ejecutar scripts de activación de un entorno virtual. **Aquí no hace falta**, porque el proyecto se ejecuta sin entorno virtual. No sirve para corregir certificados HTTPS.

## Probar con Postman

Usa `https://scifact-buscador.onrender.com` para probar el despliegue, o `http://127.0.0.1:8000` si lo ejecutas localmente.

1. **Estado del servicio:** crea una petición **GET** a `/health` y pulsa **Send**. Debe responder con `status: "ok"` y `documents: 5183`.
2. **Búsqueda:** crea una petición **POST** a `/search`. En **Body → raw → JSON**, pega:

   ```json
   {
     "query": "Does vitamin D reduce respiratory infections?",
     "model": "bm25",
     "top_k": 5
   }
   ```

3. Pulsa **Send**: debe devolver `200 OK` y una lista `results` con `rank`, `document_id`, `score`, `title` y `snippet`. Repite cambiando `"model": "tfidf"` para probar el otro modelo. `top_k` permite valores de 1 a 100.

**Endpoints:** `GET /health` y `POST /search`. La raíz `/`, `/docs` y `/openapi.json` devuelven 404 deliberadamente: **el profesor utilizará Postman, no una interfaz web**. En Render Free, una primera petición tras inactividad puede demorarse mientras se reactiva el servicio.

## Notebook y comprobación

Abre `Evaluacion_SciFact.ipynb` en VS Code, selecciona el **mismo Python** que usas para la API y ejecuta **Run All**. Si necesita un kernel: `python -m pip install ipykernel`. Evalúa 300 consultas `test` con los qrels de SciFact.

| Métrica @10 | TF-IDF | BM25 |
|---|---:|---:|
| Precision | 0.082667 | 0.092333 |
| Recall | 0.755889 | 0.838667 |
| NDCG | 0.597829 | 0.691850 |

Son los valores obtenidos en la ejecución entregada. Para ejecutar las pruebas unitarias del código: `python -m pip install pytest httpx` y luego `python -m pytest -q test_project.py`.

## Archivos y despliegue

- `engine.py`: descarga, procesamiento, indexación, modelos y evaluación.
- `main.py`: API HTTP para Postman.
- `Evaluacion_SciFact.ipynb`: resultados y análisis.
- `test_project.py`: pruebas unitarias con corpus sintético.
- `requirements.txt`: dependencias; `render.yaml`: configuración de Render.

El despliegue se realiza desde GitHub con **Render → New → Blueprint**, apuntando a `render.yaml`. Si los archivos están en la subcarpeta `SciFact Examen Bimestral`, el servicio debe tener `rootDir: "SciFact Examen Bimestral"`. Fija Python 3.12 mediante `PYTHON_VERSION` (la publicación verificada utilizó `3.12.9`). 
