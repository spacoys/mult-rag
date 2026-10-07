"""Простой UI для загрузки PDF и поиска по нему."""
import sys
from pathlib import Path
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import tempfile
import streamlit as st

from pipeline import index_pdf
from retrieval.pipeline import RetrievalPipeline

st.set_page_config(page_title="RAG по документам", layout="wide")
st.title("📄 Поиск по документам")


@st.cache_resource
def get_pipeline():
    return RetrievalPipeline()


pipeline = get_pipeline()

uploaded = st.file_uploader("Загрузите PDF", type=["pdf"])
if uploaded and st.button("Проиндексировать"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded.read())
        tmp_path = f.name

    with st.spinner("Индексирую..."):
        index_pdf(tmp_path)
    st.success("Готово!")

query = st.text_input("Введите вопрос")
if query:
    with st.spinner("Ищу..."):
        hits = pipeline.retrieve(query)

    st.subheader("Результаты")
    for i, h in enumerate(hits, 1):
        with st.expander(f"{i}. Страница {h.payload.get('page', '?')} · score {h.score:.3f}"):
            st.write(h.payload.get("text", ""))