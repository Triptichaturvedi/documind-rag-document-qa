"""Reproducible EDA script. Run: python notebooks/eda_analysis.py"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'data/raw/dataset.csv'
OUT=ROOT/'data'/'reports'
OUT.mkdir(exist_ok=True)
df=pd.read_csv(IN)
for c in ['doc_id','doc_type','chunk_id','query_type']:
    df[c]=df[c].astype(str).str.strip()
for c in ['word_count','retrieval_score','answer_quality_score']:
    df[c]=pd.to_numeric(df[c],errors='coerce')
print('Shape:',df.shape)
print('Missing values:\n',df.isna().sum())
print('Duplicate rows:',df.duplicated().sum())
print('Numeric summary:\n',df[['word_count','retrieval_score','answer_quality_score']].describe())
# 1: document type distribution
ax=df['doc_type'].value_counts().plot(kind='bar',figsize=(8,5),title='Records by Document Type',xlabel='Document type',ylabel='Record count')
plt.tight_layout(); plt.savefig(OUT/'01_document_type_distribution.png',dpi=160); plt.close()
# 2: score distributions
fig,ax=plt.subplots(figsize=(8,5)); ax.hist(df['retrieval_score'].dropna(),bins=20,alpha=.75,label='Retrieval score'); ax.hist(df['answer_quality_score'].dropna(),bins=20,alpha=.65,label='Answer quality score'); ax.set(title='Distribution of Recorded Scores',xlabel='Score',ylabel='Frequency'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'02_score_distributions.png',dpi=160); plt.close(fig)
# 3: mean scores by query type
means=df.groupby('query_type')[['retrieval_score','answer_quality_score']].mean().sort_index()
ax=means.plot(kind='bar',figsize=(9,5),title='Mean Scores by Query Type',ylabel='Mean score',ylim=(0,1))
plt.xticks(rotation=25,ha='right'); plt.tight_layout(); plt.savefig(OUT/'03_scores_by_query_type.png',dpi=160); plt.close()
# 4: word count vs answer quality
fig,ax=plt.subplots(figsize=(8,5)); ax.scatter(df['word_count'],df['answer_quality_score'],alpha=.55); ax.set(title='Chunk Word Count vs Answer Quality',xlabel='Word count',ylabel='Answer quality score'); fig.tight_layout(); fig.savefig(OUT/'04_word_count_vs_quality.png',dpi=160); plt.close(fig)
print('Correlation between retrieval and answer-quality scores:',df[['retrieval_score','answer_quality_score']].corr().iloc[0,1].round(4))
print('Saved four plots in',OUT)
