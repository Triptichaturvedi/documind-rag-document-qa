# Initial Dataset Inspection

- Rows: 500
- Columns: 7
- Missing values: 0
- Duplicate rows: 0
- Unique documents: 50
- Mean word count: 268.85
- Mean retrieval score: 0.7466
- Mean answer-quality score: 0.8029
- Retrieval-score/answer-quality Pearson correlation: -0.0045

Interpretation: the two score columns are nearly uncorrelated in this supplied sample; do not assume a high retrieval score predicts answer quality. The CSV contains no text chunks or actual query/answer pairs, so it cannot independently support end-to-end RAG testing.
