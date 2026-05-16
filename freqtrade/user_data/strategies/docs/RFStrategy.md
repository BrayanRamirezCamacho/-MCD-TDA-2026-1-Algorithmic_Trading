# Estrategia Random Forest (RF)

## Objetivo

El objetivo de esta estrategia es usar aprendizaje automático para decidir cuándo comprar. En lugar de seguir reglas fijas como "compra cuando el RSI baje de 30", entrenamos un modelo que aprende por sí solo qué combinación de indicadores suele preceder a una subida de precio.

La pregunta que intenta responder el modelo es simple: *si compro en esta vela, ¿el precio va a subir al menos 0.1% en los próximos 30 periodos?* Si el modelo responde "sí" con suficiente confianza, se genera una señal de compra.

---

## ¿Qué es un Random Forest?

Un Random Forest (Bosque Aleatorio) es un modelo de Machine Learning que funciona construyendo muchos árboles de decisión al mismo tiempo y tomando el resultado por "votación mayoritaria". Cada árbol aprende una parte diferente de los datos, y al combinarlos todos se obtiene una predicción más robusta que la de cualquier árbol por separado.

En el contexto de esta estrategia, el modelo no predice el precio exacto. En cambio, calcula la **probabilidad** de que ocurra una subida. Si esa probabilidad supera el umbral configurado (0.55 por defecto), se considera una buena oportunidad de entrada.

---

## Proceso de entrenamiento

El modelo no se entrena dentro de Freqtrade. Se entrena por separado en la libreta `RandomForest.ipynb`, usando datos históricos descargados de Binance en formato `.feather`. Una vez entrenado, se guarda como un archivo `.pkl` que la estrategia carga al arrancar.

El entrenamiento sigue un esquema de **walk-forward**: los datos se dividen en bloques consecutivos, y el modelo siempre predice sobre datos que no vio durante el entrenamiento. Esto evita el problema del lookahead (usar información del futuro sin darse cuenta).

El modelo entrenado para esta corrida fue `rf_ETH_USDT_15m`, usando velas de 15 minutos de ETH/USDT.

---

## Funcionamiento

### Features (indicadores de entrada)

El modelo recibe 27 indicadores técnicos calculados sobre las velas históricas. Estos indicadores describen el estado del mercado en cada momento:

| Grupo           | Indicadores                                                  |
|-----------------|--------------------------------------------------------------|
| Retornos        | Cambios de precio en 1, 3, 6, 12 y 24 velas                 |
| Forma de vela   | Tamaño del cuerpo y del rango (máximo - mínimo)              |
| EMAs            | Distancia del precio a EMAs de 7, 14, 21, 50, 100 y 200 periodos, y cruces entre ellas |
| RSI             | RSI de 7 y 14 periodos, normalizados entre -1 y 1            |
| Volatilidad     | Desviación estándar de los retornos en 6, 14 y 24 velas      |
| Volumen         | Ratio respecto al promedio de 14 velas y cambio de una vela a otra |
| Bollinger Bands | Posición del precio dentro de las bandas y ancho de las mismas |
| MACD            | Diferencia entre la línea MACD y la señal, normalizada       |

### Señal de compra

Cuando la probabilidad predicha por el modelo supera `UMBRAL_COMPRA = 0.55`, se genera una señal de compra. Las entradas se etiquetan según el nivel de confianza:

- `rf_alta_confianza`: probabilidad ≥ 0.70
- `rf_confianza_media`: probabilidad entre 0.55 y 0.70

### Señal de venta

La estrategia no tiene una señal de venta basada en el modelo como tal. Las salidas las manejan tres mecanismos:

- **ROI**: si el precio sube lo suficiente, se vende para asegurar la ganancia.
- **Trailing stop**: una vez alcanzada cierta ganancia, el stop "sigue" al precio hacia arriba para protegerla.
- **Stoploss**: si el precio cae demasiado desde la compra, cierra la posición para limitar la pérdida.

Adicionalmente, si la probabilidad del modelo cae por debajo de 0.30 en una vela posterior, también se genera una señal de salida (`rf_confianza_baja`), indicando que el modelo ya no cree en esa posición.

---

## Parámetros

| Parámetro                       | Valor  | Descripción                                               |
|---------------------------------|--------|-----------------------------------------------------------|
| timeframe                       | 15m    | Velas de 15 minutos                                       |
| stoploss                        | -2.5%  | Vende si la pérdida llega al 2.5%                         |
| trailing_stop_positive          | 1.5%   | Activa el trailing una vez ganado el 1.5%                 |
| trailing_stop_positive_offset   | 2.0%   | El trailing protege ganancias a partir del 2%             |
| trailing_only_offset_is_reached | True   | Antes del 2%, solo el stoploss plano protege              |
| ROI 0 min                       | 2.0%   | Vende si gana 2% en cualquier momento                     |
| ROI 60 min                      | 1.5%   | Después de 1h, vende con 1.5%                             |
| ROI 120 min                     | 1.0%   | Después de 2h, vende con 1%                               |
| ROI 240 min                     | 0.5%   | Después de 4h, vende con 0.5%                             |
| UMBRAL_COMPRA                   | 0.55   | Probabilidad mínima para generar señal de compra          |
| startup_candle_count            | 250    | Velas de calentamiento antes de la primera señal          |

---

## Resultados del Backtesting

Periodo evaluado: 01/10/2025 — 30/04/2026 (211 días)  
Par: ETH/USDT | Timeframe: 15m  
Cambio del mercado en ese periodo: -45.65%

| Métrica            | Valor                |
|--------------------|----------------------|
| Win Rate           | 63.3%                |
| Total Profit %     | -6.38%               |
| Total Profit USDT  | -63.75 USDT          |
| Max Drawdown       | 71.58 USDT (7.16%)   |
| Total Trades       | 496                  |
| Saldo inicial      | 1,000 USDT           |
| Saldo final        | 936.25 USDT          |

| Razón de salida    | Trades | Win% | Profit promedio |
|--------------------|--------|------|-----------------|
| ROI                | 313    | 100% | +1.07%          |
| rf_confianza_baja  | 17     | 5.9% | -0.83%          |
| Stop loss          | 166    | 0%   | -2.69%          |

| Nivel de confianza    | Trades | Profit promedio | Win% |
|-----------------------|--------|-----------------|------|
| rf_alta_confianza     | 18     | +0.12%          | 61.1% |
| rf_confianza_media    | 478    | -0.27%          | 63.4% |

---

## Conclusión

El resultado no es rentable. La estrategia perdió un 6.38% en un periodo donde el mercado cayó un 45.65%, lo que quiere decir que resistió bastante mejor que el mercado.

El patrón de salida es el mismo problema que ya vimos en otras estrategias: cuando el ROI o el trailing stop cierran la operación, el modelo gana el 100% de esos trades con un promedio de +1.07%. Pero los 166 trades cerrados por stoploss se fueron todos a -2.69%, destruyendo las ganancias acumuladas.

Un dato interesante es la diferencia entre los dos niveles de confianza. Las 18 entradas de `rf_alta_confianza` (probabilidad ≥ 0.70) duraron en promedio solo 27 minutos y terminaron con mejor resultado que las de confianza media. Esto sugiere que el modelo sí tiene cierta capacidad de identificar las mejores oportunidades, solo que el umbral de 0.55 es demasiado permisivo y deja entrar demasiado ruido.


- Saldo inicial: 1,000 USDT
- Saldo final: 936.25 USDT
