import re
import unicodedata
import nltk
import pandas as pd
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from tqdm import tqdm
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np
from functools import partial

STOPWORDS_PT = set(stopwords.words('portuguese'))

ABBREV_MAP = {
    r'\b(\d+)\s*ml\b': r'\1ml',
    r'\b(\d+)\s*l\b': r'\1l',
    r'\b(\d+)\s*litros?\b': r'\1l',
    r'\bml\b': 'ml',
    r'\bc/(\d+)\b': '',
    r'\bs/(\w+)\b': 'sem \1',
    r'\b(und|unid|unidade)s?\b': '',
    r'\bpet\b': 'pet',
    r'\blata\b': 'lata',
}

def normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    for pattern, repl in ABBREV_MAP.items():
        text = re.sub(pattern, repl, text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_PT and t != '']
    return ' '.join(tokens)

def normalize_batch(texts: list, desc: str = 'Normalizando') -> list:
    return [normalize(t) for t in tqdm(texts, desc=desc)]

def search_tfidf(query: str, vectorizer, catalog_matrix, product_ids, catalog, top_k: int = 5):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, catalog_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(product_ids[i], catalog.iloc[i]['product_name'], scores[i]) for i in top_indices]

def search_bm25(query: str, bm25, product_ids, catalog, top_k: int = 5):
    tokens = query.split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(product_ids[i], catalog.iloc[i]['product_name'], scores[i]) for i in top_indices]

def evaluate(queries_df: pd.DataFrame, search_fn, desc: str):
    p_at_1 = 0
    mrr = 0.0
    r_at_5 = 0

    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc=desc):
        results = search_fn(row['text'], top_k=5)
        correct_id = str(row['matched_id'])
        returned_ids = [str(r[0]) for r in results]

        if returned_ids[0] == correct_id:
            p_at_1 += 1

        for rank, pid in enumerate(returned_ids, start=1):
            if pid == correct_id:
                mrr += 1.0 / rank
                break

        if correct_id in returned_ids:
            r_at_5 += 1

    n = len(queries_df)
    print(f"\n{desc}")
    print(f"  P@1  = {p_at_1/n:.4f}")
    print(f"  MRR@5= {mrr/n:.4f}")
    print(f"  R@5  = {r_at_5/n:.4f}")

catalog = pd.read_csv('non_normalized/catalog.csv')
queries_val = pd.read_csv('non_normalized/queries_val.csv')
queries_test = pd.read_csv('non_normalized/queries_test.csv')
queries = pd.read_csv('non_normalized/queries.csv')

catalog['normalized'] = normalize_batch(catalog['product_name'].tolist(), desc='Catálogo')
queries_val['normalized'] = normalize_batch(queries_val['text'].tolist(), desc='Queries Val')
queries_test['normalized'] = normalize_batch(queries_test['text'].tolist(), desc='Queries Test')
queries['normalized'] = normalize_batch(queries['text'].tolist(), desc='Queries')

print(catalog[['product_name', 'normalized']].head())
print(queries_val[['text', 'normalized']].head())
print(queries_test[['text', 'normalized']].head())
print(queries[['text', 'normalized']].head())

catalog = catalog.drop(columns=['product_name']).rename(columns={'normalized': 'product_name'})
queries_val = queries_val.drop(columns=['text']).rename(columns={'normalized': 'text'})
queries_test = queries_test.drop(columns=['text']).rename(columns={'normalized': 'text'})
queries = queries.drop(columns=['text']).rename(columns={'normalized': 'text'})

os.makedirs('normalized', exist_ok=True)

catalog.to_csv('normalized/catalog.csv', index=False)
queries_val.to_csv('normalized/queries_val.csv', index=False)
queries_test.to_csv('normalized/queries_test.csv', index=False)
queries.to_csv('normalized/queries.csv', index=False)

corpus = catalog['product_name'].tolist()
product_ids = catalog['product_id'].tolist()

vectorizer = TfidfVectorizer()
catalog_matrix = vectorizer.fit_transform(corpus)  # (n_produtos, n_termos)

tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

tfidf_search = partial(search_tfidf, vectorizer=vectorizer, catalog_matrix=catalog_matrix, product_ids=product_ids, catalog=catalog)
bm25_search = partial(search_bm25, bm25=bm25, product_ids=product_ids, catalog=catalog)

evaluate(queries_val, tfidf_search, desc='TF-IDF')
evaluate(queries_val, bm25_search,  desc='BM25')
evaluate(queries_test, tfidf_search, desc='TF-IDF')
evaluate(queries_test, bm25_search,  desc='BM25')

queries['matched_id'] = [tfidf_search(text, top_k=1)[0][0] for text in tqdm(queries['text'], desc='Predições')]
queries.to_csv('normalized/queries_com_predicoes_tfidf.csv', index=False)
queries['matched_id'] = [bm25_search(text, top_k=1)[0][0] for text in tqdm(queries['text'], desc='Predições')]
queries.to_csv('normalized/queries_com_predicoes_bm25.csv', index=False)