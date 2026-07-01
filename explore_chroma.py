#!/usr/bin/env python3
"""
ChromaDB Live Explorer — Auto‑detect collections
Run with: streamlit run explore_chroma.py
"""

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np

CHROMA_PATH = "data/chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Connect and list collections ────────────────────
@st.cache_resource
def get_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

@st.cache_resource
def get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

# ── Fetch all chunks from a selected collection ─────
def fetch_all_chunks(collection):
    all_ids, all_docs, all_metas = [], [], []
    offset = 0
    batch_size = 500
    while True:
        result = collection.get(
            offset=offset,
            limit=batch_size,
            include=["documents", "metadatas", "embeddings"]
        )
        if not result["ids"]:
            break
        all_ids.extend(result["ids"])
        all_docs.extend(result["documents"])
        all_metas.extend(result["metadatas"])
        offset += batch_size
    return all_ids, all_docs, all_metas

def build_df(ids, docs, metas):
    rows = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        rows.append({
            "id": ids[i],
            "document": doc[:500] + "..." if len(doc) > 500 else doc,
            "full_document": doc,
            "source_type": meta.get("source_type", ""),
            "branch": meta.get("branch", ""),
            "speciality": meta.get("speciality", ""),
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "chunk_index": meta.get("chunk_index", 0),
        })
    return pd.DataFrame(rows)

# ── Main UI ──────────────────────────────────────────
st.set_page_config(page_title="ChromaDB Explorer", layout="wide")
st.title("🔍 ChromaDB — Live Explorer")

client = get_client()
collections = client.list_collections()

if not collections:
    st.error("❌ No collections found in `data/chroma`.")
    st.info(
        "Run the vector loader first:\n\n"
        "```bash\npython -m ingestion.vec_loader --reset\n```"
    )
    st.stop()

# Let user choose collection
collection_names = [c.name for c in collections]
default_index = collection_names.index("healthhub_chunks") if "healthhub_chunks" in collection_names else 0
selected_name = st.sidebar.selectbox(
    "Select Collection",
    collection_names,
    index=default_index
)

embed_fn = get_embedding_fn()
collection = client.get_collection(selected_name, embedding_function=embed_fn)

# Load data
with st.spinner(f"Loading chunks from '{selected_name}'..."):
    ids, docs, metas = fetch_all_chunks(collection)
    df = build_df(ids, docs, metas)

st.success(f"✅ Loaded {len(df)} chunks from collection **{selected_name}**")

# ── Sidebar Filters ──────────────────────────────────
st.sidebar.header("🔎 Filter")
source_types = st.sidebar.multiselect(
    "Source Type",
    options=sorted(df["source_type"].unique()),
    default=[]
)
branches = st.sidebar.multiselect(
    "Branch",
    options=sorted(df["branch"].unique()),
    default=[]
)
specialities = st.sidebar.multiselect(
    "Speciality",
    options=sorted(df["speciality"].unique()),
    default=[]
)

filtered_df = df
if source_types:
    filtered_df = filtered_df[filtered_df["source_type"].isin(source_types)]
if branches:
    filtered_df = filtered_df[filtered_df["branch"].isin(branches)]
if specialities:
    filtered_df = filtered_df[filtered_df["speciality"].isin(specialities)]

st.sidebar.markdown(f"**Showing {len(filtered_df)} chunks**")

# ── Display Table ────────────────────────────────────
st.subheader("📋 Chunk List")
st.dataframe(
    filtered_df.drop(columns=["full_document"]),
    use_container_width=True,
    height=400
)

# ── Full document viewer ─────────────────────────────
expander = st.expander("📄 View full document of selected row")
with expander:
    selected_idx = st.number_input("Row index", 0, len(filtered_df)-1, 0)
    if not filtered_df.empty and selected_idx < len(filtered_df):
        st.text_area(
            "Full document text",
            filtered_df.iloc[selected_idx]["full_document"],
            height=200
        )

# ── PCA Scatter Plot ─────────────────────────────────
st.subheader("📊 Semantic Clusters (2D PCA)")
if st.checkbox("Show vector scatter plot"):
    if not filtered_df.empty:
        ids_to_plot = filtered_df["id"].tolist()
        with st.spinner("Fetching embeddings..."):
            result = collection.get(ids=ids_to_plot, include=["embeddings"])
            embeddings = result["embeddings"]
            if embeddings:
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(np.array(embeddings))
                plot_df = filtered_df.copy()
                plot_df["x"] = reduced[:, 0]
                plot_df["y"] = reduced[:, 1]
                fig = px.scatter(
                    plot_df,
                    x="x", y="y",
                    color="source_type",
                    hover_data=["title", "branch", "speciality"],
                    title=f"PCA projection – {selected_name}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No embeddings found for filtered items.")
    else:
        st.info("No chunks to plot – adjust filters.")