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

## Notebook

[`notebooks/eda.ipynb`](notebooks/eda.ipynb) contiene el análisis exploratorio con las gráficas
ya ejecutadas. Reutiliza el loader de `src/data.py`.

## Lo que se sabe de los datos

- Clases casi balanceadas: A 34.9 %, B 34.2 %, empate 30.9 %. Predecir 1/3 para todo da
  log loss 1.0986 y predecir las proporciones de clase da 1.0972: es la referencia a batir.
- Casi no hay sesgo de posición: entre las decisiones sin empate, A gana el 50.5 %.
- **Sesgo de verbosidad:** la respuesta más larga gana el 61.6 % de las veces (sin empates y con
  longitudes distintas). Es la señal más fuerte (correlación +0.22), seguida de saltos de línea
  (+0.19) y viñetas (+0.12). Las frases de rechazo ("I cannot", "as an AI") penalizan (−0.15).
- **Empates:** casi 40 % cuando las respuestas miden lo mismo y ~26 % cuando difieren mucho.
  Si A y B son idénticas (275 filas), el 90 % empata.
- **Longitud:** el texto (prompt + A + B) tiene mediana ~590 tokens estimados, p90 ~1.5K y
  p99 ~3.8K. El 13 % de las conversaciones tiene más de un turno.
- **Prompts repetidos:** 51,734 prompts distintos en 57,477 filas. Las 8,861 filas con prompt
  repetido tienen 40.5 % de empates (29.2 % en el resto), así que la validación debe agrupar
  por `prompt`.
- Sin vacíos en `prompt` ni en las respuestas, salvo 44 filas con una respuesta vacía.
- `model_a` y `model_b` solo existen en train: sirven para describir (GPT-4 gana más, los
  modelos de 3B a 12B menos), no como features.

## Baseline (Fase 2)

```bash
pip install -r ../requirements.txt
python src/train.py
```

Calcula las features, compara cuatro modelos con validación cruzada y escribe `submission.csv`.
Tarda unos 3 minutos en CPU.

**Features** (`src/features.py`), calculadas por respuesta y como diferencia A − B:
longitud, palabras, saltos de línea, viñetas, negritas, bloques de código, encabezados, URLs,
signos de pregunta y exclamación, oraciones, proporción de dígitos y mayúsculas, frases de
rechazo, riqueza de vocabulario y solapamiento con el prompt. También: si A y B son idénticas,
el solapamiento de vocabulario entre A y B, y la longitud y número de turnos del prompt.

**Validación:** `GroupKFold` de 5 folds agrupando por `prompt`, para que una misma pregunta no
caiga en train y en validación.

**A/B intercambiados:** el train se duplica con las respuestas intercambiadas (y las etiquetas
A y B también), y en la predicción se promedia la fila original con la espejo. Así el modelo
no puede aprender un sesgo de posición.

| Modelo | Log loss (CV) |
|---|---|
| Proporciones de clase | 1.0973 |
| Regresión logística (solo diferencias) | 1.0522 |
| LightGBM | 1.0196 |
| LightGBM + promedio A/B | **1.0190** |

Los 5 folds van de 1.0177 a 1.0215. El número de árboles se elige con early stopping sobre el
fold de validación, así que la cifra puede ser un poco optimista. El modelo final se entrena
con todo el train y la cantidad de árboles promedio de la CV.

Las features con más peso (ganancia) son la diferencia de longitud, el solapamiento de
vocabulario entre A y B (señal de empate) y la diferencia de saltos de línea.

## Embeddings (Fase 3)

Los embeddings se calculan en Colab con GPU, porque en CPU tardarían horas:
[`notebooks/embeddings_colab.ipynb`](notebooks/embeddings_colab.ipynb) codifica `prompt`,
`response_a` y `response_b` con `BAAI/bge-small-en-v1.5` (384 dimensiones, fp16, texto truncado
a 512 tokens) y guarda `emb_train.npz` en Drive (117 MB). Se descarga a `data/`, que git ignora.

```bash
python src/train.py --embeddings
```

**Features nuevas** (`embedding_features` en `src/features.py`): coseno entre cada respuesta y el
prompt, coseno entre A y B, 32 componentes PCA por respuesta (con su diferencia A − B) y 16
componentes PCA del prompt. Todo se suma a las features de la Fase 2, con el mismo `GroupKFold`.

| Modelo | Log loss (CV) |
|---|---|
| Baseline (Fase 2) | 1.0190 |
| Baseline + embeddings | **1.0083** |

Los folds van de 1.0065 a 1.0121. La mejora es de ~0.011: unos embeddings congelados con un
modelo pequeño y un GBM aportan poco frente a las features de longitud, que siguen siendo las de
más peso. Las PCA se ajustan con todo train (sin usar etiquetas), así que la cifra es ligeramente
optimista. El test no se codifica en local: el test real (~25,000 filas) se codifica en el notebook
de Kaggle de la Fase 4.

## Fases

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Estructura, datos y loader | hecha |
| 1 | EDA | hecha |
| 2 | Baseline: features manuales + LightGBM, validación agrupada por `prompt` | hecha |
| 3 | Embeddings de un modelo pequeño (`bge-small`), calculados en Colab | hecha |
| 4 | Notebook de submission en Kaggle (sin internet) | pendiente |
| 5 | Fine-tuning de un LLM con LoRA en GPU (opcional) | pendiente |
| 6 | Cierre: calibración y tabla de resultados | pendiente |
