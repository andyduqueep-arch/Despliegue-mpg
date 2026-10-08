"""
App Streamlit: predicción del consumo de combustible (mpg) con el modelo de bagging optimizado.
Archivo único: no depende de otros .py. Debe estar junto a:
  one_hot_columns.joblib, min_max_scaler.joblib, bagging_optimizado.joblib y requirements.txt
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================== preprocesamiento y predicción
TARGET = 'mpg'
CATEGORICAS = {'origin': ['USA', 'Europe', 'Japan']}
ARCHIVOS = ('one_hot_columns.joblib', 'min_max_scaler.joblib', 'bagging_optimizado.joblib')

# Rango y valor por defecto de cada variable (Dataset_limpio.xlsx y reglas de calidad del punto 2.3)
RANGOS = {
    'cylinders':    dict(min=3,      max=8,      default=4,      step=1,    label='Cilindros'),
    'displacement': dict(min=50.0,   max=500.0,  default=148.8,  step=1.0,  label='Cilindrada (pulg³)'),
    'horsepower':   dict(min=40.0,   max=500.0,  default=102.8,  step=1.0,  label='Potencia (HP)'),
    'weight':       dict(min=1500.0, max=6000.0, default=2680.0, step=10.0, label='Peso (lb)'),
    'acceleration': dict(min=5.0,    max=25.0,   default=15.6,   step=0.1,  label='Aceleración 0-60 mph (s)'),
    'model_year':   dict(min=1970,   max=2026,   default=2010,   step=1,    label='Año del modelo'),
    'vehicle_age':  dict(min=0,      max=56,     default=16,     step=1,    label='Antigüedad (años)'),
}


def buscar_archivo(carpeta, nombre):
    """Encuentra el archivo aunque el navegador lo haya renombrado (p. ej. 'min_max_scaler (1).joblib')."""
    exacto = carpeta / nombre
    if exacto.exists():
        return exacto
    base = nombre.rsplit('.', 1)[0]
    candidatos = sorted(carpeta.glob(base + '*.joblib'))
    if candidatos:
        return candidatos[0]
    raise FileNotFoundError(2, 'No encontrado', nombre)


def cargar_artefactos(carpeta='.'):
    """Carga los 3 archivos .joblib: one-hot, scaler y modelo."""
    c = Path(carpeta)
    return tuple(joblib.load(buscar_archivo(c, f)) for f in ARCHIVOS)


def nombres(obj):
    """Columnas con las que se ajustó un objeto de sklearn (None si se ajustó con arrays)."""
    n = getattr(obj, 'feature_names_in_', None)
    return list(n) if n is not None else None


def es_encoder(obj):
    """True si one_hot_columns.joblib guarda un OneHotEncoder (y no una lista de columnas)."""
    return hasattr(obj, 'transform') and hasattr(obj, 'categories_')


def columnas_modelo(one_hot, modelo):
    """Orden final de columnas que espera el modelo."""
    if nombres(modelo):
        return nombres(modelo)
    return None if es_encoder(one_hot) else list(one_hot)


def variables_de_entrada(one_hot, scaler, modelo):
    """Deduce de los artefactos qué variables originales hay que pedirle al usuario."""
    cols = columnas_modelo(one_hot, modelo) or []
    if es_encoder(one_hot):
        cats = list(nombres(one_hot) or CATEGORICAS)
        base = nombres(scaler) or cols
    else:
        cats = [k for k in CATEGORICAS if any(c.startswith(k + '_') for c in cols)]
        base = cols
    nums = [c for c in base if c != TARGET and not any(c.startswith(k + '_') for k in cats)]
    return nums, cats


def opciones_categoricas(var, one_hot):
    if es_encoder(one_hot):
        j = (nombres(one_hot) or list(CATEGORICAS)).index(var)
        return [str(x) for x in one_hot.categories_[j]]
    return CATEGORICAS.get(var, [])


def aplicar_one_hot(df, one_hot):
    """Paso 1: One-Hot igual que en el entrenamiento (mismas columnas y mismo orden)."""
    if es_encoder(one_hot):
        cat_cols = nombres(one_hot) or list(CATEGORICAS)
        arr = one_hot.transform(df[cat_cols])
        arr = arr.toarray() if hasattr(arr, 'toarray') else arr
        dummies = pd.DataFrame(arr, columns=one_hot.get_feature_names_out(cat_cols), index=df.index)
        return pd.concat([df.drop(columns=cat_cols), dummies], axis=1)
    cat_cols = [c for c in CATEGORICAS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, dtype=int)
    return df.reindex(columns=list(one_hot), fill_value=0)   # agrega las dummies faltantes con 0


def aplicar_scaler(df, scaler):
    """Paso 2: MinMaxScaler sólo sobre las columnas con las que se ajustó."""
    df = df.copy()
    cols = nombres(scaler)
    if cols is None:  # ajustado con array sin nombres
        n = scaler.n_features_in_
        cols = list(df.columns) if n == df.shape[1] else [c for c in df.columns if c in RANGOS][:n]
    tmp = df.reindex(columns=cols, fill_value=0).astype(float)
    esc = pd.DataFrame(scaler.transform(tmp.values), columns=cols, index=df.index)
    for c in cols:
        if c in df.columns:
            df[c] = esc[c]
    return df


def preparar(entrada, one_hot, scaler, modelo):
    """Entrada original (dict o DataFrame) -> matriz lista para el modelo."""
    df = pd.DataFrame([entrada]) if isinstance(entrada, dict) else entrada.copy()
    df = aplicar_scaler(aplicar_one_hot(df, one_hot), scaler)
    orden = columnas_modelo(one_hot, modelo) or list(df.columns)
    return df.reindex(columns=orden, fill_value=0)


def desescalar_objetivo(pred, scaler):
    """Si el scaler incluyó mpg, la predicción sale entre 0 y 1: se devuelve a mpg."""
    cols = nombres(scaler)
    if cols and TARGET in cols:
        i = cols.index(TARGET)
        return np.asarray(pred) * scaler.data_range_[i] + scaler.data_min_[i]
    return np.asarray(pred)


def predecir(entrada, one_hot, scaler, modelo):
    """Devuelve las predicciones de mpg (array) para una o varias filas."""
    X = preparar(entrada, one_hot, scaler, modelo)
    pred = modelo.predict(X if nombres(modelo) else X.values)
    return desescalar_objetivo(pred, scaler)


def predicciones_individuales(modelo, X, scaler=None):
    """Predicción de cada estimador del ensamble para una fila (rango de incertidumbre)."""
    try:
        Xv = X.values
        feats = getattr(modelo, 'estimators_features_', None)
        if feats is not None:  # BaggingRegressor
            p = [e.predict(Xv[:, f])[0] for e, f in zip(modelo.estimators_, feats)]
        else:                  # RandomForest / ExtraTrees
            p = [e.predict(Xv)[0] for e in modelo.estimators_]
        return desescalar_objetivo(p, scaler) if scaler is not None else np.array(p)
    except Exception:
        return None


# =============================================================== interfaz
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
