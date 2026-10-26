from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def test_clean_dataset_exists_and_has_expected_columns():
    df=pd.read_csv(ROOT/'data/processed/dataset_cleaned.csv')
    expected={'doc_id','doc_type','chunk_id','word_count','retrieval_score','query_type','answer_quality_score'}
    assert expected.issubset(df.columns)
    assert len(df)==500

def test_dataset_scores_are_in_unit_interval():
    df=pd.read_csv(ROOT/'data/processed/dataset_cleaned.csv')
    assert df['retrieval_score'].between(0,1).all()
    assert df['answer_quality_score'].between(0,1).all()
