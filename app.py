"""Streamlit interface for the Research Paper Summarizer."""
from __future__ import annotations

import time
import hashlib

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modules.evaluation import calculate_metrics, calculate_rouge
from modules.pdf_extractor import PDFExtractionError, extract_pdf
from modules.preprocessing import split_sentences
from modules.tfidf_summarizer import summarize_tfidf
from modules.textrank_summarizer import summarize_textrank


st.set_page_config(page_title="Research Paper Summarizer", page_icon="📄", layout="wide")
st.title("Research Paper Summarizer")
st.write("Explore two classical extractive NLP methods for academic papers: TF-IDF salience and TextRank.")

upload = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
if upload:
    try:
        uploaded_bytes = upload.getvalue()
        document = extract_pdf(uploaded_bytes)
    except PDFExtractionError as exc:
        st.error(str(exc))
        st.stop()
    left, mid, right = st.columns(3)
    left.metric("Document", document.title)
    mid.metric("Pages", document.page_count)
    right.metric("Extracted words", document.word_count)
    with st.expander("Preview extracted text"):
        st.text(document.text[:12000] + ("…" if len(document.text) > 12000 else ""))

    source_id = hashlib.sha256(uploaded_bytes).hexdigest()
    if st.session_state.get("summary_source_id") != source_id:
        st.session_state.pop("summary_results", None)
    sentences_count = max(1, len(split_sentences(document.text)))
    requested_count = st.slider("Summary length (sentences)", 1, min(30, max(1, sentences_count)),
                                min(5, max(1, sentences_count)))
    method = st.radio("Summarization algorithm", ["TF-IDF", "TextRank", "Compare Both"], horizontal=True)
    reference = st.text_area("Reference summary (optional, for ROUGE evaluation)",
                             placeholder="Paste a human-written reference summary here.")
    if st.button("Generate Summary", type="primary"):
        methods = (["TF-IDF", "TextRank"] if method == "Compare Both" else [method])
        funcs = {"TF-IDF": summarize_tfidf, "TextRank": summarize_textrank}
        results = {}
        for name in methods:
            start = time.perf_counter()
            lines = funcs[name](document.text, requested_count)
            elapsed = time.perf_counter() - start
            summary = "\n\n".join(lines)
            metrics = calculate_metrics(document.text, summary, elapsed)
            results[name] = {"summary": summary, "metrics": metrics,
                             "rouge": calculate_rouge(summary, reference) if reference.strip() else {}}
        st.session_state["summary_results"] = results
        st.session_state["summary_doc_name"] = upload.name.rsplit(".", 1)[0]
        st.session_state["summary_source_id"] = source_id
        st.session_state["summary_has_reference"] = bool(reference.strip())

if upload and "summary_results" in st.session_state:
    results = st.session_state["summary_results"]
    columns = st.columns(len(results))
    for col, (name, result) in zip(columns, results.items()):
        metrics = result["metrics"]
        with col:
            st.subheader(name)
            st.write(result["summary"])
            st.caption(f"{metrics.word_count} words · {metrics.sentence_count} sentences")
            st.download_button("Download TXT", result["summary"],
                               file_name=f"{st.session_state['summary_doc_name']}_{name.lower().replace(' ', '_')}.txt",
                               mime="text/plain", key=f"download_{name}")
    rows = []
    for name, result in results.items():
        m = result["metrics"]
        row = {"Algorithm": name, "Processing time (s)": round(m.processing_seconds, 4),
               "Selected sentences": m.sentence_count, "Word count": m.word_count,
               "Compression ratio": round(m.compression_ratio, 4)}
        for key in ("rouge1", "rouge2", "rougeL"):
            row[f"{key.upper()} F1"] = round(result["rouge"][key]["f1"], 4) if result["rouge"] else None
        rows.append(row)
    st.subheader("Comparison and evaluation")
    st.dataframe(pd.DataFrame(rows).set_index("Algorithm"), use_container_width=True)
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar([n for n in results], [r["metrics"].processing_seconds for r in results], color=["#3674B5", "#578FCA"])
        ax.set_ylabel("Seconds")
        ax.set_title("Processing time")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    if not st.session_state.get("summary_has_reference"):
        st.info("ROUGE evaluation is unavailable until you supply a reference summary. ROUGE compares generated text with that reference.")
    else:
        st.caption("ROUGE-1, ROUGE-2 and ROUGE-L F1 scores compare each generated summary with your supplied reference.")
else:
    st.info("Upload a text-based PDF to begin. Scanned PDFs need OCR, which this version does not include.")
