import re
import unicodedata
import argparse

import nltk
nltk.download('stopwords', quiet=True)

import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np
from functools import partial
from google import genai
from anthropic import Anthropic
import time
from sentence_transformers import SentenceTransformer
import sys

class Tee:
    def __init__(self, file):
        self.file = file
        self.terminal = sys.__stdout__
    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
    def flush(self):
        self.terminal.flush()
        self.file.flush()

# -------------
# PARAMETERS
# -------------

g_save_normalized = False
g_save_predictions = False

embeddings_catalog_cache_path = 'embeddings/emb_catalog_cache.npy'
embeddings_queries_val_cache_path = 'embeddings/emb_queries_val_cache.npy'
embeddings_queries_test_cache_path = 'embeddings/emb_queries_test_cache.npy'

def make_zero_shot_prompt(query: str, candidate_list: str) -> str:
    return f"""Você é um sistema de busca de produtos de bebidas.
Dada a query do usuário, identifique qual dos candidatos abaixo é o produto mais adequado.
Responda APENAS com o número do candidato escolhido (1-10), sem explicação.

Query: "{query}"

Candidatos:
{candidate_list}

Número do candidato mais adequado:"""

def build_few_shot_examples(n_examples: int = 5, n_candidates: int = 10) -> str:
    examples = ""
    sample = g_queries_val.head(n_examples)
    
    for _, row in sample.iterrows():
        query = normalize(row['text'])
        correct_id = row['matched_id']
        
        candidates = search_bm25(query, bm25, g_products_available_ids, g_normalized_original_product_names, top_k=n_candidates)
        
        candidate_list = "\n".join(
            [f"{i+1}. (ID: {pid}) {name}" for i, (pid, name, _) in enumerate(candidates)]
        )
        
        correct_number = next(
            (i+1 for i, (pid, _, _) in enumerate(candidates) if str(pid) == str(correct_id)),
            1
        )
        
        examples += f"""Query: "{query}"
Candidatos:
{candidate_list}
Número do candidato mais adequado: {correct_number}

"""
    return examples

def make_few_shot_prompt(query: str, candidate_list: str, n_examples_build: int = 5, n_candidates_build: int = 10) -> str:
    few_shot_examples = build_few_shot_examples(n_examples=n_examples_build, n_candidates=n_candidates_build)

    return f"""Você é um sistema de busca de produtos de bebidas.
Dada a query do usuário, identifique qual dos candidatos abaixo é o produto mais adequado.
Responda APENAS com o número do candidato escolhido (1-10), sem explicação.

Exemplos:
{few_shot_examples}
Agora responda:
Query: "{query}"

Candidatos:
{candidate_list}

Número do candidato mais adequado:"""

# -------------

client = None

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

g_catalog = []
g_queries = []
g_queries_val = []
g_queries_test = []

g_normalized_queries = []
g_normalized_queries_val = []
g_normalized_queries_test = []

g_products_available_ids = []
g_normalized_original_product_names = [] 
g_normalized_catalog_names = []         

TFIDF_NO_MATCH_THRESHOLD = 0.30 # None para desativar

tfidf_vectorizer = TfidfVectorizer()

g_similarity_vocabulary_matrix = []

g_tokenized_original_product_names = []
bm25 = []

dlm_api_key = ''

def read_csvs() -> None:
    global g_catalog, g_queries_val, g_queries_test, g_queries
    
    g_catalog = pd.read_csv('non_normalized/catalog.csv')
    g_queries_val = pd.read_csv('non_normalized/queries_val.csv')
    g_queries_test = pd.read_csv('non_normalized/queries_test.csv')
    g_queries = pd.read_csv('non_normalized/queries.csv')    
        
def configure_attributes() -> None:
    global g_products_available_ids, g_normalized_original_product_names
    global g_normalized_catalog_names, bm25

    g_products_available_ids = g_catalog['product_id'].to_list()
    g_normalized_original_product_names = g_catalog['product_name'].to_list()

    g_normalized_catalog_names = [normalize(name) for name in tqdm(g_normalized_original_product_names, desc='Normalizing Catalog')]

    tokenized_catalog_names = [doc.split() for doc in g_normalized_catalog_names]
    bm25 = BM25Okapi(tokenized_catalog_names)
    
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
        pd.Series(g_normalized_queries).to_csv('normalized/queries.csv', index=False)
        pd.Series(g_normalized_queries_val).to_csv('normalized/queries_val.csv', index=False)
        pd.Series(g_normalized_queries_test).to_csv('normalized/queries_test.csv', index=False)
    
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

def apply_no_match(results: list, threshold) -> list:
    if threshold is None or not results:
        return results
    top_score = results[0][2]
    if top_score < threshold:
        return [("NO_MATCH", "NO_MATCH", top_score)] + results
    return results

def evaluate(results: list, queries_df: pd.DataFrame, desc: str):
    p_at_1 = 0
    mrr = 0.0
    r_at_5 = 0

    for results_row, (_, df_row) in zip(results, queries_df.iterrows()):
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
    
    g_similarity_vocabulary_matrix = tfidf_vectorizer.fit_transform(g_normalized_catalog_names)
    
    similarity_normalized_queries_val_tfidf = []
    similarity_normalized_queries_test_tfidf = []
    similarity_normalized_queries_val_bm25 = []
    similarity_normalized_queries_test_bm25 = []

    for row in tqdm(g_normalized_queries_val, desc='Similarity Searching Validation Queries'):
        similarity_normalized_queries_val_tfidf.append(
            apply_no_match(
                search_tfidf(row, tfidf_vectorizer, g_similarity_vocabulary_matrix, g_products_available_ids, g_normalized_original_product_names),
                TFIDF_NO_MATCH_THRESHOLD)
        )    
        similarity_normalized_queries_val_bm25.append(
            search_bm25(row, bm25, g_products_available_ids, g_normalized_original_product_names)
        ) 
    
    for row in tqdm(g_normalized_queries_test, desc='Similarity Searching Test Queries'):
        similarity_normalized_queries_test_tfidf.append(
            apply_no_match(
                search_tfidf(row, tfidf_vectorizer, g_similarity_vocabulary_matrix, g_products_available_ids, g_normalized_original_product_names),
                TFIDF_NO_MATCH_THRESHOLD)
        )
        similarity_normalized_queries_test_bm25.append(
            search_bm25(row, bm25, g_products_available_ids, g_normalized_original_product_names)
        )  
    
    evaluate(similarity_normalized_queries_val_tfidf, g_queries_val, desc='TF-IDF Val')
    evaluate(similarity_normalized_queries_test_tfidf, g_queries_test, desc='TF-IDF Test')
    evaluate(similarity_normalized_queries_val_bm25, g_queries_val, desc='BM25 Val')
    evaluate(similarity_normalized_queries_test_bm25, g_queries_test, desc='BM25 Test')

#########################################################################
# ------------------ Deep Learning
#########################################################################

def search_embeddings(query: np.ndarray, catalog_matrix, product_ids, product_names, top_k: int = 5):
    scores = cosine_similarity(query.reshape(1, -1), catalog_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(product_ids[i], product_names[i], scores[i]) for i in top_indices]  

def build_embeddings(text_vector: list, cache_path: str) -> np.ndarray:
    if os.path.exists(cache_path):
        print(f"Carregando embeddings do cache ({cache_path})...")
        return np.load(cache_path)
    
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    print("Encoding documents (this may take a few seconds)...")
    embeddings = model.encode(text_vector, show_progress_bar=True, batch_size=64)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings

def semantic_embedding() -> None:
    embedding_catalog = build_embeddings(g_normalized_original_product_names, embeddings_catalog_cache_path)
    embedding_queries_val = build_embeddings(g_normalized_queries_val, embeddings_queries_val_cache_path)
    embedding_queries_test = build_embeddings(g_normalized_queries_test, embeddings_queries_test_cache_path)
    
    results_val = []
    results_test = []
    
    for query_emb in tqdm(embedding_queries_val, desc="Searching Embedding Validation Queries"):
        results_val.append(
            search_embeddings(query_emb, embedding_catalog, g_products_available_ids, g_normalized_original_product_names)
        )
        
    for query_emb in tqdm(embedding_queries_test, desc="Searching Embedding Test Queries"):
        results_test.append(
            search_embeddings(query_emb, embedding_catalog, g_products_available_ids, g_normalized_original_product_names)
        )
    
    evaluate(results_val, g_queries_val, desc="Semantic Embedding Val")
    if 'matched_id' in g_queries_test.columns:
        evaluate(results_test, g_queries_test, desc="Semantic Embedding Test")

def _call_llm(prompt: str) -> str:
    while True:
        try:
            if g_provider == 'claude':
                response = client.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
            else:  # gemini
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=prompt
                )
                return response.text.strip()
        except Exception as e:
            if '429' in str(e) or 'rate' in str(e).lower():
                print("Rate limit atingido, aguardando 60s...")
                time.sleep(60)
            elif '503' in str(e) or 'overload' in str(e).lower():
                print("Servidor indisponível, aguardando 30s...")
                time.sleep(30)
            else:
                raise


def _llm_rerank(query: str, candidates: list, prompt_fn) -> list:
    candidate_list = "\n".join(
        [f"{i+1}. (ID: {pid}) {name}" for i, (pid, name, _) in enumerate(candidates)]
    )
    prompt = prompt_fn(query, candidate_list)
        
    text = _call_llm(prompt)    
    
    print(f"Query: '{query}'")
    print(f"Resposta do modelo: '{text}'")
    print(f"Candidatos:")
    for i, (pid, name, score) in enumerate(candidates):
        print(f"  {i+1}. (ID: {pid}) {name} [score: {score:.4f}]")

    try:
        chosen = int(text) - 1
    except ValueError:
        print(f"[WARN] Resposta inválida, usando fallback (candidato 1)")
        chosen = 0

    print(f"Escolhido: {chosen + 1}")
    reranked = [candidates[chosen]] + [c for i, c in enumerate(candidates) if i != chosen]
    return reranked

def _run_llm(desc_val: str, desc_test: str, prompt_fn) -> None:
    results_val = []
    results_test = []

    for query in tqdm(g_normalized_queries_val, desc=desc_val):
        candidates = search_bm25(query, bm25, g_products_available_ids, g_normalized_original_product_names, top_k=10)
        reranked = _llm_rerank(query, candidates, prompt_fn)
        results_val.append(reranked[:5])
        time.sleep(4)

    for query in tqdm(g_normalized_queries_test, desc=desc_test):
        candidates = search_bm25(query, bm25, g_products_available_ids, g_normalized_original_product_names, top_k=10)
        reranked = _llm_rerank(query, candidates, prompt_fn)
        results_test.append(reranked[:5])
        time.sleep(4)

    evaluate(results_val, g_queries_val, desc=f"{desc_val} Results")
    if 'matched_id' in g_queries_test.columns:
        evaluate(results_test, g_queries_test, desc=f"{desc_test} Results")

def zero_shot() -> None:
    _run_llm("Zero-shot Val", "Zero-shot Test", make_zero_shot_prompt)

def few_shot() -> None:
    _run_llm("Few-shot Val", "Few-shot Test", make_few_shot_prompt)

#########################################################################
# ------------------ Main
#########################################################################

def main() -> None:
    global dlm_api_key, client, g_provider

    parser = argparse.ArgumentParser(description="ML T2 Pipeline")
    parser.add_argument('--run_all',   action='store_true', help='Roda tudo')
    parser.add_argument('--similarity', action='store_true', help='Roda TF-IDF e BM25')
    parser.add_argument('--embedding',  action='store_true', help='Roda Semantic Embedding')
    parser.add_argument('--zero_shot',  action='store_true', help='Roda Zero-Shot LLM')
    parser.add_argument('--few_shot',   action='store_true', help='Roda Few-Shot LLM')
    parser.add_argument('--provider', choices=['gemini', 'claude'], default='gemini')
    args = parser.parse_args()
    
    g_provider = args.provider

    any_specific = args.similarity or args.embedding or args.zero_shot or args.few_shot
    run_all = args.run_all or not any_specific

    run_similarity = run_all or args.similarity
    run_embedding  = run_all or args.embedding
    run_zero_shot  = run_all or args.zero_shot
    run_few_shot   = run_all or args.few_shot

    if g_provider == 'claude':
        load_dotenv(Path(__file__).parent / ".env" / "claude.env")
        dlm_api_key = os.getenv("ANTHROPIC_API_KEY")
        client = Anthropic(api_key=dlm_api_key)
        print(f"Provedor: Claude  |  Api Key: {dlm_api_key}")
    else:
        load_dotenv(Path(__file__).parent / ".env" / "gemini.env")
        dlm_api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=dlm_api_key)
        print(f"Provedor: Gemini  |  Api Key: {dlm_api_key}")

    print(f"Flags: similarity={run_similarity}, embedding={run_embedding}, zero_shot={run_zero_shot}, few_shot={run_few_shot}")

    read_csvs()
    normalize_data()
    configure_attributes()

    if run_similarity:
        print("\n-- Running Similarity Search (TF-IDF + BM25)")
        similarity_search()

    if run_embedding:
        print("\n-- Running Semantic Embedding")
        semantic_embedding()

    if run_zero_shot:
        print("\n-- Running Zero-Shot")
        zero_shot()

    if run_few_shot:
        print("\n-- Running Few-Shot")
        few_shot()


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    log_file = open("outputs/runtime_log.txt", "w")
    sys.stdout = Tee(log_file)
    sys.stderr = Tee(log_file)

    main()

    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__