import re
import unicodedata
import nltk
import pandas as pd
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from tqdm import tqdm
import os

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