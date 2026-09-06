import streamlit as st
from Analisis_Dinamico import Analisis_Modal, Analisis_Espectral

g = 9.81  # Aceleración de la gravedad (m/s2)

# BARRA LATERAL: datos de entrada
with st.sidebar:
    st.header('Análisis Dinámico Modal Espectral')
    st.subheader('Proyecto Final — Python Aplicado a la Ingeniería Estructural')
    st.subheader('Elaborado por: Alvaro Villa')

    st.markdown("### Datos del edificio")
    r1, r2 = st.columns([1, 2], gap="medium")
    with r1: "N° de pisos"
    with r2: n_pisos = st.number_input("n_pisos", value=5, min_value=1, step=1, label_visibility="collapsed")

    # --- Genera dinámicamente 1 fila de inputs (peso, rigidez, altura) por cada piso ---
    w = []; k = []; h = []
    for i in range(int(n_pisos)):
        st.markdown(f"**Piso {i+1}**")
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            wi = st.number_input(f"Peso (tnf) - piso {i+1}", value=60.0, min_value=0.1,
                                  key=f"w{i}", label_visibility="visible")
        with c2:
            ki = st.number_input(f"Rigidez (tnf/m) - piso {i+1}", value=25000.0, min_value=0.1,
                                  key=f"k{i}", label_visibility="visible")
        with c3:
            hi = st.number_input(f"Altura (m) - piso {i+1}", value=3.0, min_value=0.1,
                                  key=f"h{i}", label_visibility="visible")
        w.append(wi)
        k.append(ki)
        h.append(hi)

    st.markdown("### Parámetros sísmicos (Norma E.030)")
    Z = st.number_input("Factor de zona (Z)", value=0.45, min_value=0.0, step=0.05)
    U = st.number_input("Factor de uso (U)", value=1.00, min_value=0.0, step=0.05)
    S = st.number_input("Factor de suelo (S)", value=1.05, min_value=0.0, step=0.05)
    R = st.number_input("Coeficiente de reducción (R)", value=6.0, min_value=1.0, step=0.5)
    Tp = st.number_input("Periodo Tp (s)", value=0.6, min_value=0.01, step=0.1)
    Tl = st.number_input("Periodo TL (s)", value=2.0, min_value=0.01, step=0.1)

    S_coeff = Z*U*S*g/R

# Conversión de unidades (igual que en el notebook: peso -> masa, dividiendo entre g)
masas = [wi*(1000/g) for wi in w]      # tnf -> masa (con tnf=1000 según el sistema de unidades de la clase)
rigideces = [ki*1000 for ki in k]      # tnf/m -> mismo sistema de unidades (tnf=1000)

# ÁREA PRINCIPAL: resultados, todo en una sola página
st.title('Análisis Dinámico Modal Espectral')

st.header('1. Análisis Modal')

x1 = Analisis_Modal(masas, rigideces, h)

st.subheader('Formas de modo (normalizadas)')
st.dataframe(x1.modes)

st.subheader('Periodos, frecuencias y masas participativas')
st.dataframe(x1.results)

st.subheader('Gráficos de formas modales')
fig1 = x1.Graficos()
st.pyplot(fig1)

st.header('2. Análisis Espectral')

x2 = Analisis_Espectral(masas, rigideces, h, S_coeff, R, Tp, Tl)

st.subheader('Coeficiente sísmico, aceleración y desplazamiento espectral')
st.dataframe(x2.results)

st.subheader('Desplazamientos por nivel')
st.dataframe(x2.displacements)

st.subheader('Fuerzas de piso')
st.dataframe(x2.forces)

st.subheader('Derivas de entrepiso')
st.dataframe(x2.drifts)

st.subheader('Gráficos de resultados espectrales')
fig2 = x2.Graficos()
st.pyplot(fig2)
