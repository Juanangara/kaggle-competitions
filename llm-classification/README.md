# LLM Classification Finetuning

Predecir cuál de dos respuestas de chatbots prefiere un usuario, con datos de Chatbot Arena
([competencia en Kaggle](https://www.kaggle.com/competitions/llm-classification-finetuning)).
Métrica: log loss sobre las probabilidades de `winner_model_a`, `winner_model_b` y `winner_tie`.

Es una *code competition*: el envío se hace desde un notebook de Kaggle, sin internet y con un
máximo de 9 horas. Se permiten datos externos y modelos preentrenados públicos.

## Datos

Descarga `train.csv`, `test.csv` y `sample_submission.csv` desde la pestaña *Data* de la
competencia (hay que aceptar las reglas) y colócalos en `data/`.

| Archivo | Filas | Notas |
|---|---|---|
| `train.csv` | 57,477 | Incluye `model_a` y `model_b`, que **no existen en test** |
| `test.csv` | 3 (ejemplo) | El scoring real usa ~25,000 filas |

`prompt`, `response_a` y `response_b` son listas JSON (un elemento por turno). `src/data.py`
las parsea y agrega `label` (0 = a, 1 = b, 2 = empate).

```bash
python src/data.py
```

## Lo que se sabe de los datos

- Clases casi balanceadas: A 34.9 %, B 34.2 %, empate 30.9 %. Un modelo uniforme da
  log loss ≈ 1.0986, que es la referencia mínima.
- Sin vacíos en `prompt` ni en las respuestas. 275 filas tienen la misma respuesta en A y B.
- 5,743 filas repiten un `prompt` ya visto, así que la validación debe agrupar por `prompt`.
- El texto total (prompt + 2 respuestas) tiene mediana de ~2.4K caracteres y p99 de ~16K.

## Fases

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Estructura, datos y loader | hecha |
| 1 | EDA | pendiente |
| 2 | Baseline: features manuales + LightGBM, validación agrupada por `prompt` | pendiente |
| 3 | Embeddings de un modelo pequeño (`bge-small` o `all-MiniLM`) | pendiente |
| 4 | Notebook de submission en Kaggle (sin internet) | pendiente |
| 5 | Fine-tuning de un LLM con LoRA en GPU (opcional) | pendiente |
| 6 | Cierre: calibración y tabla de resultados | pendiente |
