# Research Paper Summarizer

A college-level NLP application that extracts text from academic PDFs and creates **extractive** summaries with two classical methods: sentence-level TF-IDF and TextRank. It can compare their size and runtime, calculate compression ratio, and calculate ROUGE only when the user supplies a reference summary.

## Objectives

- Extract page text and basic metadata from text-based PDF files.
- Demonstrate tokenization, normalization, stopword handling, vector-space sentence scoring, graph ranking, and reference-based evaluation.
- Preserve source sentence text and order in the generated extractive summaries.
- Make experiments reproducible without paid services, a database, authentication, or downloaded language models.

## Architecture

`app.py` provides the Streamlit interface. `modules/pdf_extractor.py` reads PDF bytes with PyMuPDF and reports actionable errors. `preprocessing.py` uses NLTK sentence and word tokenizers, then removes English stopwords. Both summarizers use scikit-learn TF-IDF representations; TF-IDF sums term weights per sentence while TextRank builds a cosine-similarity graph and runs weighted PageRank. `evaluation.py` calculates summary size/compression and optional ROUGE-1/2/L against the exact user-provided reference. pandas and Matplotlib render comparison data and a runtime chart.

## Algorithms and methodology

1. Extract text for each PDF page and join nonempty page text in document order.
2. Segment text into sentences with NLTK's Punkt tokenizer. Tokenize each sentence, lowercase tokens, remove English stopwords and one-character terms, and retain the original sentence alongside its normalized tokens.
3. **TF-IDF:** represent each sentence as a TF-IDF vector over the document vocabulary, sum its weights for a salience score, select the top requested sentences, and restore source order.
4. **TextRank:** create a weighted undirected graph whose nodes are sentences and whose edge weights are cosine similarity between sentence TF-IDF vectors. Run NetworkX PageRank, select top sentences, and restore source order. Zero-similarity nodes remain valid isolated nodes.
5. Optionally compare each generated summary to a human reference with ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1. ROUGE values are not produced without a reference.
6. Record wall-clock algorithm runtime, summary word and sentence count, and compression ratio (`summary words / extracted source words`).

The summary length control asks for a number of sentences. This is an extractive baseline, so sentences are copied rather than rewritten. Neither algorithm is presumed superior; results depend on document and reference quality.

## Folder structure

```text
Research_Paper_Summarizer_NLP/
├── app.py
├── modules/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── pdf_extractor.py
│   ├── preprocessing.py
│   ├── textrank_summarizer.py
│   └── tfidf_summarizer.py
├── tests/
│   ├── test_pdf_and_evaluation.py
│   └── test_summarizers.py
├── dataset/                  # Keep locally sourced experiment papers here
├── outputs/                  # Optional local outputs
├── results_template.csv
├── requirements.txt
└── .streamlit/config.toml
```

## Requirements and installation

Python 3.10–3.12 is recommended. From the project root:

```powershell
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux use `source .venv/bin/activate`. Then install and launch:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, upload a text-based PDF, choose a sentence count and method, then generate and download the result. No NLTK corpus download is required.

## Testing

Run the automated suite from the project root:

```powershell
python -m pytest -q
```

Tests use synthetic text and locally generated PDFs. They cover both summarizers, source ordering, empty and short inputs, zero-vector sentences, PDF metadata and page extraction, corrupt/empty/image-only PDFs, and metric calculations. They do not claim results for real research papers.

### Real-paper experiment procedure (10–20 papers)

1. Select 10–20 papers with permission to use them; record citation and PDF source. Include a range of lengths and topics. Prefer PDFs with selectable text and keep scanned papers out of the primary comparison unless they are separately identified.
2. Prepare one human reference summary per paper using a consistent guideline (for example, 100–200 words covering objective, method, findings, and conclusion). Do not use either generated summary as the reference.
3. Fix the extractive sentence count in advance (such as 5) and apply it equally to both algorithms for each paper. Run each algorithm at least three times if timing stability matters; record the mean runtime. The app displays one measured run per generation.
4. For every paper and algorithm, record source title, page/word count, chosen sentence count, summary word count, runtime, compression ratio, ROUGE-1/2/L F1, reference provenance, and notes in `results_template.csv`.
5. Inspect extraction errors and report excluded papers. Summarize means and ranges by algorithm, but discuss document-specific behavior and reference limitations. Do not infer statistical superiority from a small convenience sample or present undocumented results as measured results.

The app's comparison table and runtime chart describe only the currently uploaded document and current run. ROUGE uses `rouge-score` with stemming and measures lexical overlap with the supplied reference; it is not a complete measure of summary quality.

## Limitations and future enhancements

- **Scanned PDFs:** OCR is not included. Image-only files produce a clear message rather than an empty summary. OCR could be added as an optional preprocessing stage in future work.
- PDF extraction can lose layout, reading order, tables, equations, and figure content. Metadata titles may be absent.
- English stopwords and preprocessing are used; other languages require language-aware tokenization and stopword lists.
- Extractive summaries can be repetitive or omit context. TF-IDF and TextRank are educational baselines, not claims of state-of-the-art quality.
- Future work could add OCR, section-aware ranking, configurable language support, batch experiment export, and human evaluation.

## Final validation commands

```powershell
python -m pytest -q
python -c "import app; import modules.pdf_extractor, modules.preprocessing, modules.tfidf_summarizer, modules.textrank_summarizer, modules.evaluation"
python -m streamlit run app.py
```

The last command is the interactive smoke test; upload a PDF in the browser to check extraction and generation end to end.
