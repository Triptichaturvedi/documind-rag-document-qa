# System Architecture

## Block diagram
```mermaid
flowchart TD
    A[PDF / DOCX / TXT / MD / HTML] --> B[LangChain document loaders]
    B --> C[Recursive character text splitter]
    C --> D[Sentence Transformer embeddings]
    D --> E[(ChromaDB vector store)]
    F[User question] --> G[Question embedding / similarity search]
    E --> G
    G --> H[Top-k evidence chunks]
    H --> I[Grounded prompt]
    I --> J[OpenAI API or Ollama]
    J --> K[Answer + source evidence]
    K --> L[Streamlit UI]
    M[Provided evaluation CSV] --> N[EDA + charts + descriptive metrics]
```

## Trust and safety design
- The system prompt instructs the model to answer only from retrieved context and to admit when evidence is missing.
- Source filename, page/section and chunk ID are surfaced beside answers.
- A relevance threshold prevents generation when no retrieved chunk meets the baseline threshold; this threshold requires validation.
- Uploaded document contents are treated as data, not instructions.
- API keys are entered through Streamlit's password input and are not written to project files.

## Dataset boundary
The supplied CSV includes metadata and pre-recorded scores only. The actual content for live retrieval must come from uploaded documents. Evaluation metrics from the CSV must not be represented as newly measured performance of the live application.
