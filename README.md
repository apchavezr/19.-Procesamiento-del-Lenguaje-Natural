# Demo: Streamlit + Sentiment Analysis (PLN)

## cómo ejecutar localmente
1. crear entorno virtual e instalar dependencias:
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     pip install -r requirements.txt
     ```
   - macOS / Linux (bash):
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```

2. ejecutar la app:
   ```bash
   streamlit run app.py
   ```

3. abrir el navegador en la URL indicada por Streamlit.

## despliegue rápido
- Streamlit Community Cloud:
  1) suba estos archivos a un repo en GitHub
  2) cree una nueva app en https://streamlit.io/cloud indicando `app.py` como archivo principal

- Hugging Face Spaces:
  1) cree un Space tipo *Streamlit*
  2) suba `app.py` y `requirements.txt` (opcional: `sample_reviews.csv`)

## notas
- el modelo `nlptown/bert-base-multilingual-uncased-sentiment` devuelve 1–5 estrellas; aquí se agregan a Negativo/Neutro/Positivo.
- valide desempeño con su propio dominio y monitoree sesgos antes de usarlo en producción.
