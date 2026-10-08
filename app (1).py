"""App Streamlit: predicción de consumo de combustible (mpg) con el modelo de bagging optimizado."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from prediccion_utils import (cargar_artefactos, variables_de_entrada, opciones_categoricas, RANGOS,
                              preparar, predecir, predicciones_individuales)

st.set_page_config(page_title='Predicción de consumo (mpg)', page_icon='⛽', layout='centered')
st.title('⛽ Predicción del consumo de combustible')
st.caption('Modelo de bagging optimizado · entrada → One-Hot → MinMaxScaler → predicción de mpg')


@st.cache_resource
def artefactos():
    return cargar_artefactos(Path(__file__).parent)   # busca los .joblib junto a app.py


try:
    one_hot, scaler, modelo = artefactos()
except FileNotFoundError as e:
    st.error(f'No se encontró {e.filename}. Copia los 3 archivos .joblib en la misma carpeta que app.py.')
    st.stop()

nums, cats = variables_de_entrada(one_hot, scaler, modelo)

with st.form('vehiculo'):
    st.subheader('Características del vehículo')
    entrada = {}
    c1, c2 = st.columns(2)
    for i, var in enumerate(nums):
        r = RANGOS.get(var, dict(min=None, max=None, default=0.0, step=None, label=var))
        entero = isinstance(r['default'], int)
        entrada[var] = (c1 if i % 2 == 0 else c2).number_input(
            r['label'], min_value=r['min'], max_value=r['max'], value=r['default'], step=r['step'],
            format='%d' if entero else '%.1f')
    for var in cats:
        entrada[var] = st.selectbox('Origen' if var == 'origin' else var, opciones_categoricas(var, one_hot))
    enviar = st.form_submit_button('Predecir consumo', type='primary', use_container_width=True)

if enviar:
    X = preparar(entrada, one_hot, scaler, modelo)
    mpg = float(predecir(entrada, one_hot, scaler, modelo)[0])

    st.subheader('Resultado')
    m1, m2, m3 = st.columns(3)
    m1.metric('Rendimiento', f'{mpg:.1f} mpg')
    m2.metric('Consumo', f'{235.215 / mpg:.1f} L/100 km')
    m3.metric('vs. promedio (29,9 mpg)', f'{mpg - 29.9:+.1f} mpg')

    ind = predicciones_individuales(modelo, X, scaler)
    if ind is not None and len(ind) > 1:
        p10, p90 = np.percentile(ind, [10, 90])
        st.info(f'El 80 % de los {len(ind)} estimadores del ensamble predicen entre **{p10:.1f} y {p90:.1f} mpg**.')

    nivel = 'alto' if mpg > 35 else 'medio' if mpg >= 25 else 'bajo'
    st.write(f'Clasificación de eficiencia: **{nivel}** (bajo < 25 mpg ≤ medio ≤ 35 mpg < alto).')

    with st.expander('Ver datos transformados que recibe el modelo'):
        st.write('Entrada original:'); st.dataframe(pd.DataFrame([entrada]), hide_index=True)
        st.write('Después de One-Hot + MinMaxScaler:'); st.dataframe(X.round(4), hide_index=True)
