# Titanic - Machine Learning from Disaster

Predecir qué pasajeros sobrevivieron al hundimiento del Titanic
([competencia en Kaggle](https://www.kaggle.com/competitions/titanic)).
Métrica: accuracy.

## Datos

Descarga `train.csv`, `test.csv` y `gender_submission.csv` desde la pestaña
*Data* de la competencia (hay que aceptar las reglas) y colócalos en `data/`.

## Ejecutar

```bash
python src/train.py
```

Compara tres modelos con validación cruzada y escribe `submission.csv`
(418 filas, columnas `PassengerId,Survived`).

## Notebook

[`notebooks/eda.ipynb`](notebooks/eda.ipynb) contiene el análisis exploratorio (valores
faltantes, supervivencia por sexo, clase y edad), las variables creadas y la comparación
de modelos, con las gráficas ya ejecutadas. Reutiliza las funciones de `src/train.py`.

## Enfoque

**Variables creadas**
- `Title`: extraído del nombre (Mr, Mrs, Miss, Master, Rare).
- `FamilySize` e `IsAlone`.
- `Deck` y `HasCabin`, a partir de `Cabin`.

**Valores faltantes** (calculados solo con train, para evitar fuga de datos)
- `Age`: mediana por título.
- `Fare`: mediana por clase.
- `Embarked`: moda.

**Modelos** (validación cruzada estratificada de 5 folds)

| Modelo | Accuracy (CV) |
|---|---|
| Regresión logística | 0.8305 |
| Random Forest | 0.8350 |
| Gradient Boosting | 0.8462 |

## Resultado

**0.77990** en el leaderboard público, frente a ~0.766 de la línea base
`gender_submission.csv` (solo sobreviven las mujeres).

## Lo aprendido

- La CV (0.846) es más optimista que el test (0.780): con 891 filas de train
  hay algo de sobreajuste y el test de 418 filas es ruidoso (cada pasajero vale 0.0024).
- Se probó una segunda versión con `TicketGroupSize`, `IsChild`, modelos más
  regularizados y CV repetida: **0.76315**, peor. La diferencia (27 predicciones,
  7 aciertos netos) cae dentro del ruido, así que no se concluye que esas variables
  perjudiquen. Se mantuvo la primera versión.
- No se optimizó contra el leaderboard público: con un test tan pequeño sería
  sobreajustar al azar.

## Ideas pendientes

- Tasa de supervivencia por apellido o grupo de ticket (calculada dentro de cada fold).
- Ensamble por votación de los tres modelos.
- Ajuste de hiperparámetros con CV repetida.
