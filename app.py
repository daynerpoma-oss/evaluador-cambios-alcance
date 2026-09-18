import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Evaluador Predictivo de Cambios de Alcance",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CARGA DE ARCHIVOS
# ============================================================

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_cambios_alcance.pkl")


@st.cache_data
def cargar_dataset():
    return pd.read_csv("dataset_cambios_alcance.csv")


@st.cache_data
def cargar_metricas():
    with open("metricas_modelo.json", "r", encoding="utf-8") as archivo:
        return json.load(archivo)


@st.cache_data
def cargar_importancia():
    return pd.read_csv("importancia_variables.csv")


modelo = cargar_modelo()
df = cargar_dataset()
metricas = cargar_metricas()
importancia = cargar_importancia()


# ============================================================
# TÍTULO
# ============================================================

st.title("📊 EVALUADOR PREDICTIVO DE CAMBIOS DE ALCANCE")

st.markdown(
    """
    **Herramienta de apoyo para estimar el impacto de cambios de alcance
    sobre el plazo de proyectos de edificación.**
    """
)

st.divider()


# ============================================================
# KPIs
# ============================================================

st.subheader("Indicadores del modelo")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Cambios analizados",
    metricas["n_casos"]
)

col2.metric(
    "MAE",
    f'{metricas["MAE_prueba"]:.2f} días'
)

col3.metric(
    "RMSE",
    f'{metricas["RMSE_prueba"]:.2f} días'
)

col4.metric(
    "R²",
    f'{metricas["R2_prueba"] * 100:.1f}%'
)

col5.metric(
    "Impacto promedio",
    f'{df["Dias_adicionales"].mean():.1f} días'
)


# ============================================================
# INTERPRETACIÓN DE MÉTRICAS
# ============================================================

with st.expander("¿Cómo interpretar las métricas?"):

    st.markdown(
        """
        **MAE — Error Absoluto Medio**

        Indica cuánto se desvía, en promedio, la predicción respecto
        al valor real. En nuestro modelo, el error medio es de
        aproximadamente 3 días.

        **RMSE — Raíz del Error Cuadrático Medio**

        Mide el error de predicción dando mayor peso a los errores
        grandes.

        **R² — Coeficiente de Determinación**

        Indica qué proporción de la variabilidad de los días
        adicionales es explicada por el modelo.
        """
    )


st.divider()


# ============================================================
# EVALUADOR
# ============================================================

st.header("🔎 Evaluador interactivo")

st.write(
    "Ingrese las características del cambio propuesto para estimar "
    "su posible impacto sobre el plazo del proyecto."
)

col1, col2 = st.columns(2)

with col1:

    etapa = st.selectbox(
        "Etapa",
        [
            "Diseño",
            "Construcción",
            "Cierre"
        ]
    )

    avance = st.slider(
        "Avance del proyecto (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=1
    )

    origen = st.selectbox(
        "Origen",
        [
            "Cliente",
            "Ingeniería/Diseño",
            "Normativo/Entidad",
            "Construcción/Obra"
        ]
    )


with col2:

    actividades = st.slider(
        "Actividades afectadas",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    actividad_critica = st.selectbox(
        "Actividad crítica",
        [
            "No",
            "Sí"
        ]
    )

    magnitud = st.selectbox(
        "Magnitud",
        [
            "Baja",
            "Media",
            "Alta"
        ]
    )


# ============================================================
# PREDICCIÓN
# ============================================================

if st.button(
    "🚀 PREDECIR IMPACTO",
    use_container_width=True
):

    nuevo_cambio = pd.DataFrame({
        "Etapa": [etapa],
        "Avance_%": [avance],
        "Origen": [origen],
        "Actividades_afectadas": [actividades],
        "Actividad_critica": [actividad_critica],
        "Magnitud": [magnitud]
    })

    prediccion = modelo.predict(nuevo_cambio)[0]

    # Evitar valores negativos
    prediccion = max(0, prediccion)

    # ========================================================
    # CLASIFICACIÓN DE IMPACTO
    # ========================================================

    if prediccion <= 5:

        nivel = "BAJO"

        accion = (
            "Registrar el cambio y continuar con el flujo normal "
            "de evaluación y aprobación."
        )

    elif prediccion <= 15:

        nivel = "MEDIO"

        accion = (
            "Revisar cronograma, recursos y actividades afectadas "
            "antes de aprobar el cambio."
        )

    else:

        nivel = "ALTO"

        accion = (
            "Realizar evaluación detallada de ruta crítica, "
            "reprocesos, recursos y plazo antes de aprobar."
        )


    # ========================================================
    # RESULTADO
    # ========================================================

    st.divider()

    st.subheader("Resultado de la evaluación")

    r1, r2 = st.columns(2)

    r1.metric(
        "Días adicionales estimados",
        f"{prediccion:.1f} días"
    )

    r2.metric(
        "Nivel de impacto",
        nivel
    )

    if nivel == "BAJO":

        st.success(
            f"🟢 IMPACTO BAJO\n\n{accion}"
        )

    elif nivel == "MEDIO":

        st.warning(
            f"🟡 IMPACTO MEDIO\n\n{accion}"
        )

    else:

        st.error(
            f"🔴 IMPACTO ALTO\n\n{accion}"
        )


    # ========================================================
    # CARACTERÍSTICAS DEL CAMBIO
    # ========================================================

    st.subheader("Características evaluadas")

    datos_mostrar = pd.DataFrame({
        "Característica": [
            "Etapa",
            "Avance",
            "Origen",
            "Actividades afectadas",
            "Actividad crítica",
            "Magnitud"
        ],
        "Valor": [
            etapa,
            f"{avance}%",
            origen,
            actividades,
            actividad_critica,
            magnitud
        ]
    })

    st.dataframe(
        datos_mostrar,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# IMPORTANCIA DE VARIABLES
# ============================================================

st.divider()

st.header("📈 Variables con mayor aporte predictivo")

st.info(
    "La importancia por permutación muestra cuánto cambia el "
    "desempeño del modelo cuando se altera cada variable. "
    "Un valor cercano a cero indica que, en esta muestra, "
    "alterar esa variable tuvo poco efecto sobre el error de predicción."
)

importancia_grafico = importancia.sort_values(
    "Importancia",
    ascending=True
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.barh(
    importancia_grafico["Variable"],
    importancia_grafico["Importancia"]
)

ax.set_xlabel("Importancia por permutación")
ax.set_ylabel("Variable")
ax.set_title("Importancia de las variables en el modelo")

st.pyplot(fig)


# ============================================================
# DISTRIBUCIÓN
# ============================================================

st.header("📊 Distribución de días adicionales")

st.caption(
    "Distribución observada en los 320 casos simulados utilizados "
    "para desarrollar el modelo."
)

fig2, ax2 = plt.subplots(figsize=(9, 5))

ax2.hist(
    df["Dias_adicionales"],
    bins=15
)

ax2.set_xlabel("Días adicionales")
ax2.set_ylabel("Número de cambios")
ax2.set_title("Distribución de días adicionales")

st.pyplot(fig2)


# ============================================================
# REGLA DE GESTIÓN
# ============================================================

st.header("📋 Regla de gestión para el PM")

tabla_gestion = pd.DataFrame({
    "Nivel": [
        "BAJO",
        "MEDIO",
        "ALTO"
    ],
    "Días estimados": [
        "0–5 días",
        "6–15 días",
        ">15 días"
    ],
    "Acción de gestión": [
        "Registrar el cambio y continuar el flujo normal de aprobación.",
        "Revisar cronograma, recursos y actividades afectadas.",
        "Evaluar ruta crítica, reprocesos, recursos y plazo antes de aprobar."
    ]
})

st.dataframe(
    tabla_gestion,
    hide_index=True,
    use_container_width=True
)


# ============================================================
# NOTA METODOLÓGICA
# ============================================================

st.divider()

st.caption(
    """
    Nota metodológica: el conjunto de datos utilizado en este proyecto
    es simulado y fue construido para representar escenarios plausibles
    de cambios de alcance en proyectos de edificación.

    La predicción constituye una herramienta de apoyo para la evaluación
    inicial del cambio. La decisión final corresponde al equipo de gestión
    del proyecto.
    """
)
