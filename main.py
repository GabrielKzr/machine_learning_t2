import re
import unicodedata

import nltk
nltk.download('stopwords', quiet=True)

import pandas as pd
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

g_save_normalized = False
g_save_predictions = False

g_catalog = []
g_queries = []
g_queries_val = []
g_queries_test = []

g_normalized_queries = []
g_normalized_queries_val = []
g_normalized_queries_test = []

g_products_available_ids = []

tfidf_vectorizer = TfidfVectorizer()

g_similarity_vocabulary_matrix = []

g_tokenized_original_product_names = []
bm25 = []

def read_csvs() -> None:
    global g_catalog, g_queries_val, g_queries_test, g_queries
    
    g_catalog = pd.read_csv('non_normalized/catalog.csv')
    g_queries_val = pd.read_csv('non_normalized/queries_val.csv')
    g_queries_test = pd.read_csv('non_normalized/queries_test.csv')
    g_queries = pd.read_csv('non_normalized/queries.csv')    
        
def configure_attributes() -> None:
    global g_products_available_ids, g_normalized_original_product_names, bm25

    g_products_available_ids = g_catalog['product_id'].to_list()
    g_normalized_original_product_names = g_catalog['product_name'].to_list()
    
    tokenized_original_product_names = [doc.split() for doc in g_normalized_original_product_names]
    bm25 = BM25Okapi(tokenized_original_product_names)
    
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

def normalize_data() -> None:
    global g_normalized_queries, g_normalized_queries_val, g_normalized_queries_test
    
    g_normalized_queries = [normalize(t) for t in tqdm(g_queries['text'], desc='Normalizing Queries')]
    g_normalized_queries_val = [normalize(t) for t in tqdm(g_queries_val['text'], desc='Normalizing Validation Queries')]
    g_normalized_queries_test = [normalize(t) for t in tqdm(g_queries_test['text'], desc='Normalizing Test Queries')]
    
    if g_save_normalized:
        g_normalized_queries.to_csv('normalized/queries_val.csv', index=False)
        g_normalized_queries_val.to_csv('normalized/queries_test.csv', index=False)
        g_normalized_queries_test.to_csv('normalized/queries.csv', index=False)
    
def search_tfidf(query: str, vectorizer, catalog_matrix, product_ids, product_names, top_k: int = 5):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, catalog_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(product_ids[i], product_names[i], scores[i]) for i in top_indices]    

def search_bm25(query: str, bm25, product_ids, product_names, top_k: int = 5):
    tokens = query.split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(product_ids[i], product_names[i], scores[i]) for i in top_indices]

def evaluate(results: list, queries_df: pd.DataFrame, desc: str):
    p_at_1 = 0
    mrr = 0.0
    r_at_5 = 0

    for results_row, (_, df_row) in tqdm(zip(results, queries_df.iterrows()), total=len(queries_df), desc=desc):
        correct_id = str(df_row['matched_id'])
        returned_ids = [str(r[0]) for r in results_row]

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

def similarity_search() -> None: 
    global g_similarity_vocabulary_matrix, bm25
    
    # creating vocabulary matrix
    g_similarity_vocabulary_matrix = tfidf_vectorizer.fit_transform(g_normalized_original_product_names)  
    '''
        Example:
                      coca  cola  refrigerante  guarana  1l  ...
        produto 0  [  0.8   0.6      0.3         0       0.4  ]
        produto 1  [  0     0        0.3         0.9     0.4  ]
        produto 2  [  0     0        0           0       0    ]
        ...
    '''
    
    similarity_normalized_queries_tfidf = []
    similarity_normalized_queries_val_tfidf = []
    similarity_normalized_queries_test_tfidf = []
    
    similarity_normalized_queries_bm25 = []
    similarity_normalized_queries_val_bm25 = []
    similarity_normalized_queries_test_bm25 = []
    
    for row in tqdm(g_normalized_queries, desc='Similarity Searching Queries'):
        similarity_normalized_queries_tfidf.append(
            search_tfidf(
                row,
                tfidf_vectorizer,
                g_similarity_vocabulary_matrix,
                g_products_available_ids,
                g_normalized_original_product_names
            )
        )
        similarity_normalized_queries_bm25.append(
            search_bm25(
                row,
                bm25,
                g_products_available_ids,
                g_normalized_original_product_names
            )            
        )        

    for row in tqdm(g_normalized_queries_val, desc='Similarity Searching Validation Queries'):
        similarity_normalized_queries_val_tfidf.append(
            search_tfidf(
                row,
                tfidf_vectorizer,
                g_similarity_vocabulary_matrix,
                g_products_available_ids,
                g_normalized_original_product_names
            )
        )    
        similarity_normalized_queries_val_bm25.append(
            search_bm25(
                row,
                bm25,
                g_products_available_ids,
                g_normalized_original_product_names
            )            
        ) 
    
    for row in tqdm(g_normalized_queries_test, desc='Similarity Searching Test Queries'):
        similarity_normalized_queries_test_tfidf.append(
            search_tfidf(
                row,
                tfidf_vectorizer,
                g_similarity_vocabulary_matrix,
                g_products_available_ids,
                g_normalized_original_product_names
            )
        )
        similarity_normalized_queries_test_bm25.append(
            search_bm25(
                row,
                bm25,
                g_products_available_ids,
                g_normalized_original_product_names
            )            
        )  
    
    #evaluate(similarity_normalized_queries_tfidf, g_queries, desc='TF-IDF Val') # esse tu não pode rodar, porque ele não tem a coluna matched ID
    #evaluate(similarity_normalized_queries_bm25, g_queries, desc='TF-IDF Val') # esse tu não pode rodar, porque ele não tem a coluna matched ID
    evaluate(similarity_normalized_queries_val_tfidf, g_queries_val, desc='TF-IDF Val')
    evaluate(similarity_normalized_queries_test_tfidf, g_queries_test, desc='TF-IDF Test')
    evaluate(similarity_normalized_queries_val_bm25, g_queries_val, desc='BM25 Val')
    evaluate(similarity_normalized_queries_test_bm25, g_queries_test, desc='BM25 Test')
    

def deep_learning() -> None:
    pass

pipeline = [
    read_csvs,
    configure_attributes,
    normalize_data,
    similarity_search,
    deep_learning,    
]

for func in pipeline:
    func()