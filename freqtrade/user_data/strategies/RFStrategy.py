# RFStrategy.py — Estrategia Random Forest para Freqtrade
#
# Carga un modelo .pkl entrenado en RandomForest.ipynb y genera señales
# de compra cuando la probabilidad predicha supera el umbral configurado.
#
# INSTRUCCIONES!!!! importante!!!
# Antes de usar:
#   1. Entrena el modelo con el notebook y copia el .pkl a:
#      freqtrade/user_data/strategies/models/ (o donde sea, aun ta dificil ubicar los modelos)
#   2. Ajusta MODEL_PATH y UMBRAL_COMPRA abajo.
#   3. Los features en _calcular_features() deben ser idénticos a los
#      del notebook (mismas columnas, mismo orden).

import logging
import os
import numpy as np
import pandas as pd
import joblib

from freqtrade.strategy import IStrategy
from pandas import DataFrame

logger = logging.getLogger(__name__)


class RFStrategy(IStrategy):

    # Configuración — ajusta estos valores

    MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        "rf_ETH_USDT_15m_20260516_1001.pkl"
    )

    UMBRAL_COMPRA = 0.55

    # Parámetros de Freqtrade

    timeframe = "15m"

    minimal_roi = {
        "0":   0.02,
        "60":  0.015,
        "120": 0.01,
        "240": 0.005,
    }

    stoploss = -0.025

    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    max_open_trades = 3

    # Velas de calentamiento para que los indicadores de periodo largo
    # (EMA 200, volatilidad 24) estén estables antes de la primera señal.
    startup_candle_count = 250

    process_only_new_candles = True

    # Inicialización — carga el modelo una sola vez al arrancar

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.modelo_rf = None
        self._cargar_modelo()

    def _cargar_modelo(self):
        if not os.path.exists(self.MODEL_PATH):
            logger.error(
                f"[RFStrategy] Modelo no encontrado en: {self.MODEL_PATH}\n"
                f"  Entrénalo con RandomForest.ipynb y copia el .pkl a:\n"
                f"  freqtrade/user_data/strategies/models/"
            )
            return

        try:
            self.modelo_rf = joblib.load(self.MODEL_PATH)
            logger.info(f"[RFStrategy] Modelo cargado: {self.MODEL_PATH}")
        except Exception as e:
            logger.error(f"[RFStrategy] Error al cargar modelo: {e}")

    # Features — deben ser idénticos a calcular_features() del notebook.
    # Si modificas algo aquí, modifícalo también allá.

    def _calcular_features(self, df: DataFrame) -> DataFrame:
        d = df.copy()

        close  = d['close'].values.astype(float)
        high   = d['high'].values.astype(float)
        low    = d['low'].values.astype(float)
        volume = d['volume'].values.astype(float)
        open_  = d['open'].values.astype(float)

        # Retornos y momentum
        d['ret_1']  = pd.Series(close).pct_change(1).values  * 100
        d['ret_3']  = pd.Series(close).pct_change(3).values  * 100
        d['ret_6']  = pd.Series(close).pct_change(6).values  * 100
        d['ret_12'] = pd.Series(close).pct_change(12).values * 100
        d['ret_24'] = pd.Series(close).pct_change(24).values * 100

        d['cuerpo_vela'] = (close - open_) / (open_ + 1e-9) * 100
        d['rango_vela']  = (high - low)    / (low   + 1e-9) * 100

        # EMAs — distancia del precio a cada media y cruces entre ellas
        precio_series = pd.Series(close)
        for periodo in [7, 14, 21, 50, 100, 200]:
            ema = precio_series.ewm(span=periodo, adjust=False).mean().values
            d[f'dist_ema_{periodo}'] = (close - ema) / (ema + 1e-9) * 100

        ema_7  = precio_series.ewm(span=7,  adjust=False).mean().values
        ema_21 = precio_series.ewm(span=21, adjust=False).mean().values
        ema_50 = precio_series.ewm(span=50, adjust=False).mean().values

        d['cruce_ema_7_21']  = (ema_7  - ema_21) / (ema_21 + 1e-9) * 100
        d['cruce_ema_7_50']  = (ema_7  - ema_50) / (ema_50 + 1e-9) * 100
        d['cruce_ema_21_50'] = (ema_21 - ema_50) / (ema_50 + 1e-9) * 100

        # RSI — normalizado a [-1, 1] para el modelo
        def _rsi(precios, periodo=14):
            delta        = pd.Series(precios).diff()
            ganancia     = delta.clip(lower=0)
            perdida      = -delta.clip(upper=0)
            avg_ganancia = ganancia.ewm(com=periodo - 1, adjust=False).mean()
            avg_perdida  = perdida.ewm(com=periodo - 1, adjust=False).mean()
            rs = avg_ganancia / (avg_perdida + 1e-9)
            return (100 - (100 / (1 + rs))).values

        try:
            import talib
            d['rsi_14'] = talib.RSI(close, timeperiod=14)
            d['rsi_7']  = talib.RSI(close, timeperiod=7)
        except ImportError:
            d['rsi_14'] = _rsi(close, 14)
            d['rsi_7']  = _rsi(close, 7)

        d['rsi_14_norm'] = (d['rsi_14'] - 50) / 50
        d['rsi_7_norm']  = (d['rsi_7']  - 50) / 50

        # Volatilidad
        retornos = precio_series.pct_change()
        d['volatilidad_6']  = retornos.rolling(6).std().values  * 100
        d['volatilidad_14'] = retornos.rolling(14).std().values * 100
        d['volatilidad_24'] = retornos.rolling(24).std().values * 100

        # Volumen
        vol_series = pd.Series(volume)
        vol_ma14   = vol_series.rolling(14).mean()
        d['ratio_volumen_14'] = volume / (vol_ma14.values + 1e-9)
        d['vol_ret_1']        = vol_series.pct_change(1).values

        # Bandas de Bollinger
        bb_media = precio_series.rolling(20).mean()
        bb_std   = precio_series.rolling(20).std()
        bb_upper = (bb_media + 2 * bb_std).values
        bb_lower = (bb_media - 2 * bb_std).values
        bb_width = bb_upper - bb_lower
        d['bb_posicion']   = (close - bb_lower) / (bb_width + 1e-9)
        d['bb_ancho_norm'] = bb_width / (bb_media.values + 1e-9)

        # MACD
        ema_12      = precio_series.ewm(span=12, adjust=False).mean().values
        ema_26      = precio_series.ewm(span=26, adjust=False).mean().values
        macd_line   = ema_12 - ema_26
        signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
        d['macd_diff'] = macd_line - signal_line
        d['macd_norm'] = macd_line / (close + 1e-9) * 100

        return d

    # Orden de columnas — debe ser idéntico al del entrenamiento
    COLUMNAS_FEATURES = [
        'ret_1', 'ret_3', 'ret_6', 'ret_12', 'ret_24',
        'cuerpo_vela', 'rango_vela',
        'dist_ema_7', 'dist_ema_14', 'dist_ema_21',
        'dist_ema_50', 'dist_ema_100', 'dist_ema_200',
        'cruce_ema_7_21', 'cruce_ema_7_50', 'cruce_ema_21_50',
        'rsi_14_norm', 'rsi_7_norm',
        'volatilidad_6', 'volatilidad_14', 'volatilidad_24',
        'ratio_volumen_14', 'vol_ret_1',
        'bb_posicion', 'bb_ancho_norm',
        'macd_diff', 'macd_norm',
    ]

    # Indicadores — Freqtrade llama esto en cada nueva vela

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.modelo_rf is None:
            logger.warning("[RFStrategy] Modelo no cargado. Sin señales.")
            dataframe['prob_sube'] = 0.0
            return dataframe

        dataframe = self._calcular_features(dataframe)

        mask_valido = dataframe[self.COLUMNAS_FEATURES].notna().all(axis=1)
        dataframe['prob_sube'] = 0.0

        if mask_valido.sum() > 0:
            X = dataframe.loc[mask_valido, self.COLUMNAS_FEATURES].values
            try:
                probabilidades = self.modelo_rf.predict_proba(X)[:, 1]
                dataframe.loc[mask_valido, 'prob_sube'] = probabilidades
            except Exception as e:
                logger.error(f"[RFStrategy] Error en predict_proba: {e}")

        return dataframe

    # Señal de entrada

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_tag']  = ''

        señal = (
            (dataframe['prob_sube'] >= self.UMBRAL_COMPRA) &
            (dataframe['volume'] > 0) &
            (dataframe['prob_sube'] > 0)
        )

        dataframe.loc[señal, 'enter_long'] = 1

        # Tags para análisis por enter_tag en Freqtrade
        dataframe.loc[señal & (dataframe['prob_sube'] >= 0.70), 'enter_tag'] = 'rf_alta_confianza'
        dataframe.loc[
            señal & (dataframe['prob_sube'] >= self.UMBRAL_COMPRA) & (dataframe['prob_sube'] < 0.70),
            'enter_tag'
        ] = 'rf_confianza_media'

        return dataframe

    # Señal de salida

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_tag']  = ''

        # Salida explícita cuando la confianza cae mucho.
        # La salida principal sigue siendo minimal_roi y trailing_stop.
        dataframe.loc[dataframe['prob_sube'] < 0.30, 'exit_long'] = 1
        dataframe.loc[dataframe['prob_sube'] < 0.30, 'exit_tag']  = 'rf_confianza_baja'

        return dataframe

    # Protecciones

    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 3,
                "stop_duration_candles": 12,
                "only_per_pair": False,
            },
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2,
            },
        ]
