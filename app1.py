# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:53:38 2025

@author: jahop
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuración de la página ---
st.set_page_config(page_title="Dashboard Retail Avanzado", layout="wide")



# --- Introducción personalizada ---
st.markdown("""
# 👋 Hola, soy Javier Horacio Pérez Ricárdez

### 📌 Postulación: **Jefe Sr de Ingeniería de Datos** en El Puerto de Liverpool

Bienvenidos a esta aplicación creada especialmente para demostrar mis habilidades técnicas y analíticas aplicadas al puesto.  
Aquí podrán ver una simulación de un entorno real de trabajo en el área de datos para retail, con análisis interactivo, visualizaciones, alertas, y predicciones.

> Esta aplicación fue desarrollada con **Python**, **Streamlit**, **Plotly**, **scikit-learn** y herramientas comunes en la nube para procesamiento y visualización de datos.

---

""")


# --- Simulación de datos ---
@st.cache_data
def generar_datos():
    np.random.seed(42)
    fechas = pd.date_range("2023-01-01", "2025-06-01", freq="D")
    tiendas = ["Liverpool Perisur", "Liverpool Polanco", "Liverpool Satélite", "Liverpool Monterrey"]
    productos = ["Ropa", "Electrónica", "Juguetes", "Hogar", "Zapatos"]

    data = []
    for _ in range(3000):
        fecha = np.random.choice(fechas)
        tienda = np.random.choice(tiendas)
        producto = np.random.choice(productos)
        unidades = np.random.randint(1, 10)
        precio = np.random.uniform(200, 5000)
        venta = round(unidades * precio, 2)
        data.append([fecha, tienda, producto, unidades, venta])

    df = pd.DataFrame(data, columns=["Fecha", "Tienda", "Producto", "Unidades", "Venta"])
    df["Año"] = df["Fecha"].dt.year
    df["Mes"] = df["Fecha"].dt.to_period("M").astype(str)
    return df

df = generar_datos()

# --- Sidebar ---
st.sidebar.title("🔎 Filtros")
años = st.sidebar.multiselect("Selecciona Año(s):", sorted(df["Año"].unique()), default=sorted(df["Año"].unique()))
tiendas = st.sidebar.multiselect("Selecciona Tienda(s):", df["Tienda"].unique(), default=df["Tienda"].unique())
productos = st.sidebar.multiselect("Selecciona Producto(s):", df["Producto"].unique(), default=df["Producto"].unique())

df_filtrado = df[(df["Año"].isin(años)) & (df["Tienda"].isin(tiendas)) & (df["Producto"].isin(productos))]

# --- Alerta inteligente ---
ventas_totales_por_tienda = df_filtrado.groupby("Tienda")["Venta"].sum()
for tienda, venta in ventas_totales_por_tienda.items():
    if venta < 200000:
        st.warning(f"⚠️ Alerta: La tienda **{tienda}** tiene ventas bajas (${venta:,.2f})")

# --- KPIs ---
st.title("📊 Dashboard de Ventas - Área Digital")
col1, col2, col3 = st.columns(3)
col1.metric("🛒 Ventas Totales", f"${df_filtrado['Venta'].sum():,.2f}")
col2.metric("📦 Productos Vendidos", int(df_filtrado["Unidades"].sum()))
ticket = df_filtrado["Venta"].sum() / df_filtrado["Unidades"].sum() if df_filtrado["Unidades"].sum() > 0 else 0
col3.metric("💳 Ticket Promedio", f"${ticket:,.2f}")

# --- Gráficos ---
col4, col5 = st.columns(2)

# Ventas por tienda
ventas_por_tienda = df_filtrado.groupby("Tienda")["Venta"].sum().reset_index()
fig1 = px.bar(ventas_por_tienda, x="Tienda", y="Venta", title="Ventas por Tienda", color="Tienda", text_auto=True)
col4.plotly_chart(fig1, use_container_width=True)

# Ventas por mes
ventas_por_mes = df_filtrado.groupby("Mes")["Venta"].sum().reset_index()
fig2 = px.line(ventas_por_mes, x="Mes", y="Venta", title="Ventas Mensuales", markers=True)
col5.plotly_chart(fig2, use_container_width=True)

# --- Predicción de ventas para el siguiente mes ---
st.subheader("📈 Predicción de Ventas del Próximo Mes")
ventas_mes = df_filtrado.groupby("Mes")["Venta"].sum().reset_index()
ventas_mes["Mes_num"] = pd.to_datetime(ventas_mes["Mes"]).dt.to_period("M").astype(str)
ventas_mes["Mes_num"] = pd.to_datetime(ventas_mes["Mes_num"]).map(lambda x: x.toordinal())

X = ventas_mes[["Mes_num"]]
y = ventas_mes["Venta"]
modelo = LinearRegression().fit(X, y)
next_month = X["Mes_num"].max() + 30  # aproximación
prediccion = modelo.predict([[next_month]])[0]

fig_pred = px.scatter(ventas_mes, x="Mes", y="Venta", title="Predicción con Regresión Lineal")
fig_pred.add_scatter(x=["Próximo mes"], y=[prediccion], mode='markers+text', name="Predicción",
                     text=[f"${prediccion:,.2f}"], textposition="top center")
st.plotly_chart(fig_pred, use_container_width=True)

# --- Mapa de calor de correlaciones ---
st.subheader("📊 Mapa de Calor de Correlaciones")
corr = df_filtrado[["Unidades", "Venta"]].corr()
fig_heatmap, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig_heatmap)

# --- Descargar Excel ---
def convertir_a_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Datos')
    output.seek(0)
    return output

st.download_button(
    label="📥 Descargar Datos en Excel",
    data=convertir_a_excel(df_filtrado),
    file_name="reporte_ventas_avanzado.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# --- Footer ---
st.markdown("---")
st.markdown("📌 Aplicación creada por **Javier Horacio Pérez Ricárdez** para demostrar habilidades en ingeniería de datos con Python, Streamlit, Machine Learning y visualización avanzada.")
