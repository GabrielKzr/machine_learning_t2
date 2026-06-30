# T2 - Matching de Produtos | Aprendizado de Máquina

## Integrantes

- Gabriel Kaizer — Matrícula: 000000
- João Pedro Antunes — Matrícula: 22107504
- Vinicius Mibielli — Matrícula: 22107795

---

## Visão Geral

A partir de um texto de consulta heterogêneo (ex.: **"COCA COLA 1L C/6"**), o sistema identifica o produto correspondente em um catálogo normalizado com aproximadamente **14 mil produtos**.

Foram implementadas duas famílias de abordagens:

- **Abordagem 1 (NLP clássico)**
  - TF-IDF + Similaridade do Cosseno
  - BM25

- **Abordagem 2 (Deep Learning)**
  - Embeddings semânticos (*Sentence-Transformers*)
  - Re-ranqueamento por LLM (Gemini e Claude, em modos zero-shot e few-shot)

As métricas reportadas são:

- **P@1 (Precision@1)**
- **MRR@5 (Mean Reciprocal Rank@5)**
- **R@5 (Recall@5)**

avaliadas sobre os conjuntos de **validação** e **teste**.

---

## Estrutura do Projeto

```text
.
├── main.py                    # Pipeline completo (todas as abordagens)
├── parse_log.py               # Recalcula métricas do Gemini/Claude a partir do log
├── relatorio_T2.pdf           # Relatório final
├── requirements.txt           # Dependências
├── non_normalized/            # Dados (catalog, queries, queries_val, queries_test)
├── embeddings/                # Cache de embeddings (.npy)
└── deep_learning_results/     # Logs das execuções do Gemini/Claude
```

---

## Instalação

### 1. Crie um ambiente virtual (recomendado)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

No Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

> **Observação**
>
> Na primeira execução serão baixados modelos utilizados pelas abordagens de embeddings.
> Um dos modelos possui aproximadamente **1 GB**, portanto é necessário haver espaço disponível em disco.

---

## Configuração da API (Gemini / Claude)

As abordagens **zero-shot** e **few-shot** utilizam as APIs do **Google Gemini** e do **Anthropic Claude**.

As chaves são lidas das variáveis de ambiente:

- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`

### Linux / macOS

```bash
export GEMINI_API_KEY="sua_chave_aqui"
export ANTHROPIC_API_KEY="sua_chave_aqui"
```

### Windows (CMD)

```cmd
set GEMINI_API_KEY=sua_chave_aqui
set ANTHROPIC_API_KEY=sua_chave_aqui
```

### Windows (PowerShell)

```powershell
$env:GEMINI_API_KEY="sua_chave_aqui"
$env:ANTHROPIC_API_KEY="sua_chave_aqui"
```

> **Importante:** As abordagens clássicas (TF-IDF e BM25) e de embeddings **não necessitam de API Key**.

---

## Como Reproduzir os Resultados

Execute os comandos a partir da raiz do projeto.

### Executar todas as abordagens

```bash
python3 main.py
```

### Apenas TF-IDF e BM25

```bash
python3 main.py --similarity
```

### Apenas embeddings semânticos

```bash
python3 main.py --embedding
```

### Apenas re-ranqueamento por LLM

```bash
python3 main.py --zero_shot
python3 main.py --few_shot
```

A saída apresenta as métricas **P@1**, **MRR@5** e **R@5** para validação e teste.

Além disso, um log completo é salvo em:

```text
outputs/runtime_log.txt
```

### Recalcular as métricas do Gemini/Claude

Sem repetir as chamadas à API:

```bash
python3 parse_log.py
```

---

## Resultados

### Validação (250 consultas)

| Método | P@1 | MRR@5 | R@5 |
|--------|----:|------:|----:|
| TF-IDF + Cosseno | **0.960** | 0.979 | 1.000 |
| BM25 | 0.956 | 0.976 | 1.000 |
| Embeddings (mpnet) | 0.788 | 0.841 | 0.924 |
| BM25 + Gemini (zero-shot) | **0.968** | **0.983** | **1.000** |
| BM25 + Claude (few-shot) | **0.968** | **0.983** | **1.000** |

### Teste (250 consultas)

| Método | P@1 | MRR@5 | R@5 |
|--------|----:|------:|----:|
| TF-IDF + Cosseno | **0.976** | **0.986** | **1.000** |
| BM25 | 0.972 | 0.984 | 1.000 |
| Embeddings (mpnet) | 0.844 | 0.878 | 0.928 |
| BM25 + Gemini (zero-shot) | 0.972 | 0.985 | 1.000 |
| BM25 + Claude (few-shot) | 0.972 | **0.986** | **1.000** |

---

## Observações

- **TF-IDF** apresentou o melhor desempenho no conjunto de **teste**, além de baixa latência (~2 s) e não depender de serviços externos.
- O re-ranqueamento por **LLMs** (Gemini/Claude) obteve a melhor precisão na **validação**, porém com custo de API e maior tempo de execução (~24 min).
- Os embeddings semânticos (*Sentence-Transformers*) tiveram desempenho inferior às abordagens baseadas em recuperação lexical para este conjunto de dados.