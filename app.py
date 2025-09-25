import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Análisis de Sentimientos", page_icon="🗣️")

@st.cache_resource(show_spinner=False)
def load_classifier():
    # return_all_scores=True para obtener prob distribucional por estrellas (1 a 5)
    clf = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        return_all_scores=True
    )
    return clf

def aggregate_sentiment(star_scores):
    """
    star_scores: lista de dicts [{'label': '1 star', 'score': 0.1}, ...]
    retorna: dict con agregación a Negativo/Neutro/Positivo y etiqueta ganadora
    """
    scores = {d["label"]: d["score"] for d in star_scores}
    p_neg = scores.get("1 star", 0.0) + scores.get("2 stars", 0.0)
    p_neu = scores.get("3 stars", 0.0)
    p_pos = scores.get("4 stars", 0.0) + scores.get("5 stars", 0.0)
    agg = {"Negativo": p_neg, "Neutro": p_neu, "Positivo": p_pos}
    pred = max(agg, key=agg.get)
    conf = agg[pred]
    return agg, pred, conf, scores

st.title("Análisis de Sentimientos en Reseñas de Productos")
st.write("Escriba una reseña y observe cómo el modelo la clasifica (negativo, neutro o positivo).")
st.caption("Modelo base: nlptown/bert-base-multilingual-uncased-sentiment (1–5 estrellas).")

# ejemplos rápidos
ejemplos = [
    "El producto llegó tarde y en mal estado.",
    "Excelente servicio, lo recomiendo.",
    "Cumple lo que promete, aunque podría mejorar la batería.",
    "Mala calidad y pésima atención al cliente."
]
ejemplo = st.sidebar.selectbox("Ejemplos de reseñas", ["—"] + ejemplos, index=0)
texto = st.text_area("Ingrese su reseña aquí:", value="" if ejemplo=="—" else ejemplo, height=120)

if st.button("Analizar", type="primary"):
    if not texto.strip():
        st.warning("Por favor escriba una reseña.")
    else:
        with st.spinner("Cargando modelo y analizando…"):
            classifier = load_classifier()
            all_scores = classifier(texto)[0]  # lista de 5 dicts: '1 star' … '5 stars'
            agg, pred, conf, star_scores = aggregate_sentiment(all_scores)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Sentimiento predicho:** {pred}")
            st.markdown(f"**Confianza:** {conf*100:.2f} %")

        # tabla con las 5 estrellas
        df_stars = pd.DataFrame([star_scores]).T.reset_index()
        df_stars.columns = ["Etiqueta (estrellas)", "Probabilidad"]
        st.subheader("Distribución por estrellas")
        st.dataframe(df_stars, use_container_width=True)

        # grafico de barras simple usando st.bar_chart
        st.subheader("Probabilidades agregadas")
        df_agg = pd.DataFrame.from_dict(agg, orient="index", columns=["Probabilidad"])
        st.bar_chart(df_agg)

        with col2:
            st.info("Interpretación rápida:\n\n- 1–2 ⭐ → Negativo\n- 3 ⭐ → Neutro\n- 4–5 ⭐ → Positivo")

st.divider()
st.markdown(
    "Notas: este prototipo usa un modelo multilingüe entrenado para puntuar de 1 a 5 estrellas. "
    "Se agregan las probabilidades para mostrar una etiqueta en español."
)
st.caption("Advertencia: no usar en producción sin validar sesgos, desempeño por dominio y controles de privacidad.")
