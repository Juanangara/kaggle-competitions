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
| Baseline + embeddings | **1.0086** |

Los folds van de 1.0071 a 1.0112. La mejora es de ~0.010: unos embeddings congelados con un
modelo pequeño y un GBM aportan poco frente a las features de longitud, que siguen siendo las de
más peso. Las PCA se ajustan con todo train (sin usar etiquetas), así que la cifra es ligeramente
optimista. El test no se codifica en local: el test real (~25,000 filas) se codifica en el notebook
de Kaggle de la Fase 4.

## Submission en Kaggle (Fase 4)

Es una *code competition*: el notebook corre en Kaggle sin internet. Los artefactos se generan
en local y se suben como dataset.

```bash
python src/train.py --embeddings   # entrena y guarda artifacts/lgbm.txt y projections.npz
python src/bundle.py               # copia el código de inferencia junto a los artefactos
```

La carpeta `artifacts/` (4 MB, no va a git) se sube como dataset privado. El modelo y las
proyecciones se guardan en formatos independientes de la versión (texto de LightGBM y arrays de
NumPy), porque Kaggle usa otras versiones de LightGBM y scikit-learn.

[`notebooks/submission_kaggle.ipynb`](notebooks/submission_kaggle.ipynb) lee el test, codifica
cada texto único con `bge-small-en-v1.5` (el modelo público de Kaggle `happenmass/bge-small-en-v1.5`,
añadido como entrada), calcula las features y predice con `src/predict.py`, promediando cada fila
con su espejo A/B. Busca las entradas con `rglob` en vez de rutas fijas, porque Kaggle organiza
`/kaggle/input` de forma distinta según el tipo de entrada.

| Envío | Puntaje público (log loss) |
|---|---|
| LightGBM + embeddings | **1.00436** |

El puntaje público se calcula sobre una parte del test y es más bajo que la CV local (1.0086), lo
que indica que el pipeline de inferencia en Kaggle reproduce el entrenamiento.

## Fine-tuning con LoRA (Fase 5)

[`notebooks/lora_colab.ipynb`](notebooks/lora_colab.ipynb) ajusta `Qwen/Qwen2.5-1.5B-Instruct` como
clasificador de 3 clases con LoRA (r=16, todas las proyecciones) en una GPU T4 de Colab. La entrada
es `prompt + respuesta A + respuesta B`, con cada campo truncado por separado (256 y 384 tokens).
Igual que en la Fase 2, A y B se intercambian al azar al entrenar y la predicción promedia cada fila
con su espejo. Entrena con 20,000 filas (1,250 pasos, 211 min) y valida en 3,398 filas separadas por
`prompt`. Los pesos del adaptador y las predicciones de validación quedan en Drive.

| Modelo (mismas 3,398 filas de validación) | Log loss |
|---|---|
| LLM con LoRA | 1.0105 |
| LightGBM + embeddings (predicciones fuera de muestra) | **1.0019** |
| Mezcla, 35-40 % LLM + LightGBM | **0.9973** |

Con 20k filas y una T4, el LLM no alcanza a LightGBM. Mezclados mejoran ~0.005, porque ven cosas
distintas (el texto frente a longitud y estilo), pero sus aciertos están muy correlacionados (0.82)
y la ganancia es pequeña. Subirla a Kaggle exigiría inferir 25,000 filas dos veces con el LLM
(~1.5 h estimadas en T4) y montar `peft` y el modelo base sin internet. **No se llevó a Kaggle**: el
envío sigue siendo el de la Fase 4. Un LLM mejor pediría más datos (el train completo son ~10 h de T4)
o un modelo más grande, fuera del alcance de la GPU gratuita.

Los números de la mezcla se reproducen con `python src/calibration.py` (necesita `lora_val_pred.npz`
en `data/`).

## Calibración (Fase 6)

LightGBM ya sale calibrado: en las 57,477 predicciones fuera de muestra, la confianza media de cada
tramo coincide con la precisión (por ejemplo 0.800 frente a 0.822 en el tramo más seguro), y la
proporción media predicha por clase (34.6 / 34.6 / 30.8 %) es casi la real (34.9 / 34.2 / 30.9 %).
Se probó escalar por temperatura y una regresión logística sobre los logits, ajustadas con 4 folds y
evaluadas en el quinto:

| Corrección | Log loss (CV) |
|---|---|
| Sin corregir | 1.00859 |
| Temperatura | 1.00859 |
| Regresión logística sobre logits | 1.00849 |

La mejor gana 0.0001, así que no se aplicó y el envío de la Fase 4 se queda como está.

## Resumen de resultados

| Modelo | Log loss | Dónde |
|---|---|---|
| Proporciones de clase | 1.0973 | CV |
| Regresión logística (diferencias) | 1.0522 | CV |
| LightGBM + promedio A/B | 1.0190 | CV |
| LightGBM + embeddings `bge-small` | 1.0086 | CV |
| Mismo, enviado a Kaggle | **1.00436** | puntaje público |
| LLM con LoRA (20k filas) | 1.0105 | 3,398 filas (LightGBM: 1.0019) |
| LLM + LightGBM, no enviado | 0.9973 | las mismas 3,398 filas |

Lo que más aportó fue la longitud y el estilo de las respuestas (Fase 2, de 1.097 a 1.019). Los
embeddings sumaron 0.010, el puntaje en Kaggle salió algo mejor que la CV, y el LLM entrenado en una
T4 no mejoró lo que ya había. Los embeddings congelados y el GBM se agotan cerca de 1.00; bajar de ahí
requiere un LLM entrenado con todos los datos.

## Fases

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Estructura, datos y loader | hecha |
| 1 | EDA | hecha |
| 2 | Baseline: features manuales + LightGBM, validación agrupada por `prompt` | hecha |
| 3 | Embeddings de un modelo pequeño (`bge-small`), calculados en Colab | hecha |
| 4 | Notebook de submission en Kaggle (sin internet) | hecha |
| 5 | Fine-tuning de un LLM con LoRA en Colab (T4), mezcla con LightGBM | hecha, sin enviar a Kaggle |
| 6 | Cierre: calibración y tabla de resultados | hecha |
