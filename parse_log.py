import re
import sys
import pandas as pd
from tqdm import tqdm


def parse_zero_shot_log(log_text: str):
    results = []
    chosen_counts = {}

    lines = log_text.splitlines()
    i = 0
    while i < len(lines):
        # Procura início de bloco: linha que começa com "Query: '"
        if "Query: '" not in lines[i]:
            i += 1
            continue

        # Lê query
        query_start = lines[i].index("Query: '")
        query_line = lines[i][query_start:]
        i += 1

        # Lê resposta do modelo
        if i >= len(lines) or not lines[i].startswith("Resposta do modelo: '"):
            continue
        i += 1  # pula linha da resposta

        # Pula "Candidatos:"
        if i >= len(lines) or not lines[i].startswith("Candidatos:"):
            continue
        i += 1

        # Lê candidatos
        candidates = []
        candidate_pattern = re.compile(r"^\s+(\d+)\. \(ID: ([^\)]+)\) (.+) \[score: ([\d\.]+)\]$")
        while i < len(lines):
            m = candidate_pattern.match(lines[i])
            if m:
                rank = int(m.group(1))
                pid = m.group(2)
                name = m.group(3)
                score = float(m.group(4))
                candidates.append((pid, name, score))
                i += 1
            else:
                break

        # Pula linha de WARN se existir
        if i < len(lines) and lines[i].startswith("[WARN]"):
            i += 1

        # Lê "Escolhido: N"
        if i >= len(lines) or not lines[i].startswith("Escolhido: "):
            continue
        chosen_rank = int(lines[i].split(": ")[1].strip())
        i += 1

        if not candidates:
            continue

        chosen_idx = chosen_rank - 1
        if chosen_idx >= len(candidates):
            chosen_idx = 0

        chosen = candidates[chosen_idx]
        reranked = [chosen] + [c for j, c in enumerate(candidates) if j != chosen_idx]
        results.append(reranked[:5])

        chosen_counts[chosen_rank] = chosen_counts.get(chosen_rank, 0) + 1

    print(f"Blocos extraídos: {len(results)}")
    print(f"Distribuição de escolhas: {dict(sorted(chosen_counts.items()))}")
    return results


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


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "deep_learning_results/log_zeroshot_results.txt"
    queries_val_path = sys.argv[2] if len(sys.argv) > 2 else "non_normalized/queries_val.csv"

    print(f"Lendo log: {log_path}")
    with open(log_path, "r") as f:
        log_text = f.read()

    results = parse_zero_shot_log(log_text)

    queries_val = pd.read_csv(queries_val_path)

    if len(results) != len(queries_val):
        print(f"[WARN] {len(results)} resultados vs {len(queries_val)} queries — usando os primeiros {len(results)}")
        queries_val = queries_val.iloc[:len(results)]

    evaluate(results, queries_val, desc="Zero-shot Val (do log)")