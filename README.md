# RAG-Based Document Q&A System — HCLTech Career Shaper Project 72

A Streamlit application for document-grounded question answering using LangChain, ChromaDB, sentence-transformer embeddings, and either OpenAI or Ollama. The supplied `dataset.csv` is retained for EDA and evaluation analytics; it contains recorded metrics and metadata, not source text. The live RAG index is therefore built from PDFs, DOCX, TXT, Markdown, or HTML files uploaded in the app.

GitHub repository: https://github.com/Triptichaturvedi/documind-rag-document-qa

## Requirements
- Windows 10/11, Python 3.12 recommended (also used by Streamlit Community Cloud)
- 8 GB RAM recommended for CPU embedding; first run downloads `all-MiniLM-L6-v2`
- Internet access for installing packages and downloading embeddings; OpenAI provider additionally needs an API key and may incur charges
- OpenAI is the cloud-compatible default and requires the `OPENAI_API_KEY`
  secret. Ollama remains available locally with `llama3.2:3b` pulled and
  running; it does not require OpenAI credits.

## Windows setup
Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud deployment

1. Push the source files, `requirements.txt`, tests, EDA script, dataset
   metadata, and the four PNG reports to a public GitHub repository.
2. In Streamlit Community Cloud, choose **Create app**, select the repository
   and branch, and set the main file to `app.py`.
3. Open **Advanced settings** and select **Python 3.12**. This is required
   for the pinned ONNX Runtime wheel used by the existing Chroma and
   sentence-transformer stack. Streamlit Community Cloud controls this
   setting in its deployment UI; `runtime.txt` is not the supported Python
   version configuration for Community Cloud.
4. In the app settings, add `OPENAI_API_KEY` under **Secrets** if using the
   OpenAI provider. Never commit that value.
5. OpenAI is the supported hosted deployment provider. Ollama requires a
   separately running local Ollama server and is not available inside a normal
   Streamlit Community Cloud deployment.
6. Uploaded documents and the local Chroma index are runtime data. They are
   not repository assets and may be lost when the Cloud app restarts or is
   redeployed; this project does not use a remote persistent vector database.

The deployment pins `chromadb==0.5.23` and `onnxruntime==1.20.1`. These are
the versions already validated in the local Python 3.12 environment. Do not
upgrade either package as part of this deployment fix without revalidating the
full RAG stack.

The public repository should include `data/raw/dataset.csv`,
`data/processed/dataset_cleaned.csv`,
`data/processed/data_dictionary_completed.csv`, `notebooks/eda_analysis.py`,
and the four images in `data/reports/`. The supplied CSVs contain metadata and
recorded scores only; they do not contain document text or personal records.
Do not publish uploaded documents, Chroma database files, secrets, or the
local `rag_test.txt` fixture.

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

## Use the application
1. Launch Streamlit and open the local URL shown in the terminal.
2. The app defaults to OpenAI so the same configuration works in Streamlit
   Community Cloud. Configure `OPENAI_API_KEY` through Streamlit Secrets or
   the environment; do not paste it into source code or commit it. For local
   use without an API key, select Ollama and verify the configured URL and
   `llama3.2:3b` service.
3. Upload one or more PDF, DOCX, TXT, MD, or HTML files.
4. Click **Build / replace document index** and wait for embedding to finish.
5. Enter a question and click **Get answer**. Expand evidence panels to inspect source chunks and relevance scores.
6. Uploading a new batch and rebuilding replaces the current index.

## Dataset and EDA
- `data/raw/dataset.csv`: unchanged supplied file
- `data/processed/dataset_cleaned.csv`: cleaned, validated copy
- `data/processed/data_dictionary_completed.csv`: completed dictionary based on observed columns
- `python notebooks/eda_analysis.py`: creates four PNG visualisations under `data/reports/`
- `pytest -q`: runs dataset, provider, retrieval, refusal, and temporary Chroma
  persistence tests

The verified dataset snapshot contains 500 rows and 7 columns. The three score/size
fields contain no missing values; `word_count` ranges from 50 to 499,
`retrieval_score` from 0.5008 to 0.9989, and `answer_quality_score` from 0.600
to 1.000. There are no duplicate rows, and all seven cleaned columns are
represented exactly once in `data_dictionary_completed.csv`. The raw and cleaned
files have the same schema and row count; the cleaned file is normalized rather
than byte-for-byte identical to the raw file.

The EDA run on 2026-10-26 generated these four files in `data/reports/`:
`01_document_type_distribution.png`, `02_score_distributions.png`,
`03_scores_by_query_type.png`, and `04_word_count_vs_quality.png`. It reported
mean retrieval score `0.746560`, mean answer quality `0.802908`, and correlation
`-0.0045` between the two recorded score columns. These are descriptive
statistics from the supplied CSV, not measurements of the live app.

### Verified runtime checks

- `pytest -q`: **6 passed** (including mocked provider, refusal,
  attribution, and temporary Chroma persistence tests)
- Python compilation of `app.py`, `src/rag_engine.py`, and
  `notebooks/eda_analysis.py`: **passed**
- Streamlit startup on port 8503: **HTTP 200**
- Document ingestion and splitting using `rag_test.txt`: **1 document, 1
  chunk**, with source and chunk metadata preserved
- Fresh Chroma persistence/reopen: **passed**; the relevant query retrieved
  `PF-204` with relevance score `0.754993` and source
  `rag_test.txt` / section `0` / `CHUNK-00001`
- An irrelevant question returned the explicit no-evidence response with no
  sources. Answer generation was exercised with a deterministic local test
  stub; Ollama is not installed in the current environment, so no live model
  answer is claimed.
- A real OpenAI API call was not performed because no credential was available;
  the automated OpenAI tests use a mock runnable and do not incur API charges.

The data dictionary describes the supplied score fields but does not claim how their scores were calculated. Mean scores in the app are descriptive statistics of the provided dataset, not measured live application performance.

## Architecture
Document upload → format-specific loader → recursive chunking with overlap → sentence-transformer embeddings → ChromaDB vector index → top-k similarity retrieval → OpenAI/Ollama answer generation using retrieved context → answer with source evidence.

## Important limitations
- This dataset does not contain `chunk_text`, `query`, `answer`, `source`, or `embedding_model`, despite these being listed as expected corpus fields in the brief. Do not treat the supplied CSV as the actual source corpus or fabricate missing text.
- Retrieval score threshold is a configurable baseline, not a calibrated guarantee of relevance. Validate it against mentor-approved question/answer pairs.
- PDF extraction quality depends on selectable text; scanned PDFs may require OCR, which is not included in this baseline.
- Do not use for legal/medical decisions. Verify generated answers against cited source passages.
- Add only documents you are authorized to process. Keep personal data out unless there is a valid basis and appropriate safeguards.
- If Ollama reports `llama-server.exe not found`, the Ollama installation is incomplete. Repair or reinstall Ollama from the official installer, then confirm `ollama list` still shows `llama3.2:3b`; the app and model store do not need to be changed.
- The original `chroma_db/`, `chroma_db_runtime/`, and their contents are
  preserved. Current indexes are written to `chroma_db_runtime_current/` so
  incompatible legacy native index files are not opened or overwritten. The
  older runtime directory was observed to fail Chroma initialization with
  `KeyError: '_type` under the installed Chroma version.
- The app resolves `chroma_db_runtime_current/` to an absolute path and passes
  an explicit Chroma persistent client configured for `default_tenant` and
  `default_database` to both index creation and reopening. Restart Streamlit
  after upgrading or changing the Chroma environment so the running process
  loads the current configuration.
- Streamlit Community Cloud does not provide the local Ollama service and
  should use OpenAI with a secret API key. The deployed app's Chroma index is
  ephemeral unless a remote persistence strategy is added.
- Provider errors are surfaced as actionable messages for missing
  credentials, authentication failures, rate limits, timeouts, and connection
  failures. Automated tests mock model calls; a real OpenAI API call requires
  your own key and may incur charges.

## Project deliverables checklist
- [x] Working Streamlit prototype code
- [x] Cleaned `dataset.csv` copy
- [x] Completed data dictionary
- [x] EDA script with 4 visualisations
- [x] Basic automated data tests
- [ ] Run and test on target laptop
- [ ] Add mentor-approved evaluation questions and record actual metrics
- [ ] Prepare 8+ page report, synopsis, 1–2 minute demo, and signed ethical declaration
