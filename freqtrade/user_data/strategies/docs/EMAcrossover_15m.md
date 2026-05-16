# EMA Crossover 15m

## Objetivo
El objetivo de esta estrategia es implementar una con uso de análisis técnico puro. Se basa en un patrón conocido del trading: el cruce de dos medias móviles exponenciales (EMA).

## ¿Qué es el EMA Crossover?

Supongamos que el precio de una criptomoneda sube y baja constantemente. Si se quiere saber si en general está subiendo o bajando, se puede calcular el promedio de los últimos precios. Eso es una media móvil, un número que va cambiando conforme pasan las velas y te da una idea de la tendencia general, sin tanto ruido.

Una **EMA** (Exponential Moving Average) es un tipo de media móvil que le da más importancia a los precios más recientes. Si el precio acaba de subir mucho, la EMA reacciona más rápido que una media normal.

En esta estrategia se usan dos EMAs al mismo tiempo:
- **EMA 9**: usa los últimos 9 periodos. Reacciona rápido a los cambios de precio.
- **EMA 21**: usa los últimos 21 periodos. Es más lenta y representa mejor la tendencia general.

Cuando la EMA rápida (9) cruza **por encima** de la EMA lenta (21), se interpreta como que el precio está ganando fuerza hacia arriba y es momento de comprar. A ese momento se le llama **cruce**, y es la señal central de esta estrategia.

También se usa una tercera EMA de 200 periodos como filtro: si el precio está por debajo de ella, significa que la tendencia de fondo es a la baja, y la estrategia no entra aunque haya cruce. La idea es no comprar en medio de una caída sostenida.

Los periodos 9 y 21 son de los más usados en crypto para timeframes cortos y se encuentran en prácticamente cualquier guía de trading técnico. Dicho esto, también está ampliamente documentado que esta estrategia por sí sola no es muy rentable: las EMAs reaccionan tarde por naturaleza, y en mercados sin una dirección clara generan muchas señales falsas. Aun así, es un baseline válido y un punto de partida para agregar mejoras.

## Funcionamiento

### Señal de compra:
```python
dataframe.loc[
    (dataframe["ema9"] > dataframe["ema21"]) &                    # EMA rápida sobre EMA lenta
    (dataframe["ema9"].shift(1) <= dataframe["ema21"].shift(1)) & # cruce en esta vela
    (dataframe["close"] > dataframe["ema200"]) &                  # precio sobre tendencia de fondo
    (dataframe["volume"] > 0),
    ["enter_long", "enter_tag"]
] = [1, "ema9_cross_21_above200"]
```

Compra cuando la EMA 9 acaba de cruzar por encima de la EMA 21, pero solo si el precio está sobre la EMA 200. Este último filtro es para no comprar cuando el mercado en general está cayendo.

### Señal de venta:
Esta estrategia no tiene señal de venta explícita. Las salidas las manejan:

* **ROI por tiempo**: vende cuando se alcanza la ganancia objetivo según el tiempo transcurrido.

* **Trailing stop**: una vez que la ganancia llega al 3%, el stop sigue el precio hacia arriba. Si el precio cae un 2% desde el punto más alto que alcanzó, vende para asegurar parte de la ganancia.

* **Stoploss del -6%**: si el precio cae un 6% desde la entrada, vende para limitar la pérdida.

> **Nota sobre la señal de salida**: se probó también salir cuando el cruce se invierte, es decir, cuando la EMA 9 vuelve a cruzar por debajo de la EMA 21. Los resultados fueron peores porque ese cruce llegaba demasiado tarde, cuando el precio ya había caído bastante. Se descartó esa variante.

## Parámetros

| Parámetro                       | Valor | Descripción                                          |
|---------------------------------|-------|------------------------------------------------------|
| timeframe                       | 15m   | Velas de 15 minutos                                  |
| EMA rápida                      | 9     | Periodos para la EMA sensible al precio reciente     |
| EMA lenta                       | 21    | Periodos para la EMA de tendencia                    |
| EMA de fondo                    | 200   | Filtro: solo compra si el precio está sobre esta EMA |
| stoploss                        | -6%   | Vende si la pérdida llega al 6%                      |
| trailing_stop_positive          | 2%    | Trailing de 2% una vez activado                      |
| trailing_stop_positive_offset   | 3%    | Se activa cuando la ganancia llega al 3%             |
| trailing_only_offset_is_reached | True  | Antes del 3%, solo el stoploss plano protege         |
| ROI 0min                        | 3%    | Vende si gana 3% en cualquier momento                |
| ROI 45min                       | 2%    | Después de 45min, vende con 2%                       |
| ROI 90min                       | 1%    | Después de 90min, vende con 1%                       |
| ROI 180min                      | 0.5%  | Después de 3h, vende con 0.5%                        |

## Resultados del Backtesting

Periodo evaluado: 01/10/2025 — 01/04/2026 (182 días)
Pares: BTC/USDT, ETH/USDT, SOL/USDT

| Métrica           | Valor               |
|-------------------|---------------------|
| Win Rate          | 79.9%               |
| Total Profit %    | -9.40%              |
| Total Profit USDT | -94.046 USDT        |
| Max Drawdown      | 99.381 USDT (9.89%) |
| Total Trades      | 264                 |
| Market Change     | -49.95%             |
| Saldo inicial     | 1,000 USDT          |
| Saldo final       | 905.954 USDT        |

| Razón de salida | Trades | Win% | Profit promedio |
|-----------------|--------|------|-----------------|
| ROI             | 211    | 100% | +0.63%          |
| Stop loss       | 52     | 0%   | -6.19%          |
| Force exit      | 1      | 0%   | -0.87%          |

| Mes       | Trades | Profit USDT | Win%  |
|-----------|--------|-------------|-------|
| Oct 2025  | 53     | -11.017     | 84.9% |
| Nov 2025  | 34     | -20.948     | 73.5% |
| Dic 2025  | 35     | -12.331     | 80.0% |
| Ene 2026  | 46     | -8.085      | 84.8% |
| Feb 2026  | 42     | -27.295     | 71.4% |
| Mar 2026  | 53     | -13.937     | 83.0% |

## Conclusión

La estrategia tiene un win rate llamativamente alto del `79.9%`, es decir, casi 8 de cada 10 trades terminan en ganancia. El problema está en que cada uno de esos wins solo promedia `+0.63%`, mientras que los 52 stop losses promedian `-6.19%` cada uno. Con esa diferencia, necesitarías casi 10 trades ganadores para recuperar 1 perdedor, lo cual hace muy difícil ser rentable aunque se gane la mayoría del tiempo.

Es el mismo patrón que se observó en `FOMO`: un win rate engañoso. La estrategia sabe ganar, pero cuando pierde, pierde demasiado.

El contexto también importa: el mercado cayó un `49.95%` durante el periodo evaluado. Una estrategia que solo compra como esta opera en contra de la corriente en ese escenario.

El siguiente paso natural es atacar esta perdida desmédida, es decir, intentar reducir el stoploss para que las pérdidas sean más pequeñas, y agregar filtros de entrada para no entrar en cualquier cruce sino solo en los que tengan más probabilidad de funcionar.

* Saldo inicial: 1,000 USDT
* Saldo final: 905.954 USDT
