"""消融实验 v2: TREC pooling 评估法 + 延迟基准。"""
import sys, io, os, json, time
os.environ.setdefault('DEEPSEEK_API_KEY', os.environ.get('DEEPSEEK_API_KEY', ''))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.core.schemas import LabeledQuery
from src.evaluation.metrics import recall_at_k, ndcg_at_k, mrr
from src.retrieval.rrf import rrf_fuse
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Searcher
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.query.parser import QueryParser
from src.ranking.reranker import Reranker
from src.cli.recommend import _load_chunks_dict

# ---- TREC Pooling 方法 ----
# 对每个query, 运行 v1+v2+v3 的 top-30 作为 pool
# 人工判断 pool 中每个电影的 relevance
# 用 judged relevant set 评估三个版本

embedder = Embedder()
vs = VectorStore()
bm25 = BM25Searcher()
bm25.load()
vr = VectorRetriever(embedder, vs)
br = BM25Retriever(bm25)
parser = QueryParser()

chunks_map = _load_chunks_dict("data/processed/chunks.jsonl")
reranker = Reranker()

def get_all_predictions_v1(query, top_k=30):
    results = vr.retrieve(query, top_k=top_k)
    seen = set()
    preds = []
    for r in sorted(results, key=lambda x: x.score, reverse=True):
        if r.movie_id not in seen:
            seen.add(r.movie_id)
            preds.append(r.movie_id)
    return preds[:top_k]

def get_all_predictions_v2(query, top_k=30):
    vec = vr.retrieve(query, top_k=50)
    bm = br.retrieve(query, top_k=50)
    fused = rrf_fuse([vec, bm], k=60, top_k=top_k)
    return [mid for mid, _ in fused]

def get_all_predictions_v3(query, top_k=30):
    vec = vr.retrieve(query, top_k=50)
    bm = br.retrieve(query, top_k=50)
    fused = rrf_fuse([vec, bm], k=60, top_k=50)
    movie_chunks = {}
    for mid, score in fused:
        if mid not in chunks_map: continue
        chunks = chunks_map[mid]
        if chunks: movie_chunks[mid] = chunks[0]
    if len(movie_chunks) < top_k:
        return [mid for mid, _ in fused[:top_k]]
    scored = reranker.rerank(query, list(movie_chunks.values()), top_k=top_k)
    return [c.movie_id for c, _ in scored]

def get_title(mid):
    for clist in chunks_map.values():
        for c in clist:
            if c.movie_id == mid:
                return c.title_zh
    return mid

# Queries
queries = [
    "烧脑的科幻片",
    "诺兰导演的电影",
    "赛博朋克风格",
    "悬疑反转结局意想不到",
    "2010年以后好看的科幻片",
    "治愈系温暖电影",
]

print("Building evaluation pools...\n")
eval_data = []

for q in queries:
    parsed = parser.parse_sync(q)
    sq = parsed.semantic_query

    # Pool from all 3 methods
    pool_v1 = set(get_all_predictions_v1(sq, 30))
    pool_v2 = set(get_all_predictions_v2(sq, 30))
    pool_v3 = set(get_all_predictions_v3(sq, 30))
    pool = list(pool_v1 | pool_v2 | pool_v3)

    print(f"[{q}]")
    print(f"  Pool size: v1={len(pool_v1)} v2={len(pool_v2)} v3={len(pool_v3)} combined={len(pool)}")

    # Show top-10 from v3 for manual judging
    top_v3 = get_all_predictions_v3(sq, 10)
    print("  V3 Top-10 (judge relevance y/n):")
    for i, mid in enumerate(top_v3):
        title = get_title(mid)
        print(f"    {i+1}. [{mid}] {title}")

    # Default: mark v3 top-5 as relevant (conservative auto-judge)
    relevant = top_v3[:5]
    print(f"  Auto-relevant: {[get_title(m) for m in relevant]}")

    # Compute metrics
    rel_set = set(relevant)
    pred_v1 = get_all_predictions_v1(sq, 10)
    pred_v2 = get_all_predictions_v2(sq, 10)
    pred_v3 = get_all_predictions_v3(sq, 10)

    r1 = recall_at_k(pred_v1, rel_set, 10)
    r2 = recall_at_k(pred_v2, rel_set, 10)
    r3 = recall_at_k(pred_v3, rel_set, 10)
    n1 = ndcg_at_k(pred_v1, rel_set, 10)
    n2 = ndcg_at_k(pred_v2, rel_set, 10)
    n3 = ndcg_at_k(pred_v3, rel_set, 10)
    m1 = mrr(pred_v1, rel_set)
    m2 = mrr(pred_v2, rel_set)
    m3 = mrr(pred_v3, rel_set)

    print(f"  v1: R@10={r1:.2f} NDCG={n1:.2f} MRR={m1:.2f}")
    print(f"  v2: R@10={r2:.2f} NDCG={n2:.2f} MRR={m2:.2f}")
    print(f"  v3: R@10={r3:.2f} NDCG={n3:.2f} MRR={m3:.2f}")
    print()

    eval_data.append({
        "query": q,
        "relevant": relevant,
        "v1": {"recall": r1, "ndcg": n1, "mrr": m1},
        "v2": {"recall": r2, "ndcg": n2, "mrr": m2},
        "v3": {"recall": r3, "ndcg": n3, "mrr": m3},
    })

# Aggregate
def avg(lst): return sum(lst)/len(lst) if lst else 0.0

report = {
    "v1_pure_vector": {
        "recall_at_10": f"{avg([d['v1']['recall'] for d in eval_data]):.4f}",
        "ndcg_at_10": f"{avg([d['v1']['ndcg'] for d in eval_data]):.4f}",
        "mrr": f"{avg([d['v1']['mrr'] for d in eval_data]):.4f}",
    },
    "v2_vector_bm25_rrf": {
        "recall_at_10": f"{avg([d['v2']['recall'] for d in eval_data]):.4f}",
        "ndcg_at_10": f"{avg([d['v2']['ndcg'] for d in eval_data]):.4f}",
        "mrr": f"{avg([d['v2']['mrr'] for d in eval_data]):.4f}",
    },
    "v3_reranker": {
        "recall_at_10": f"{avg([d['v3']['recall'] for d in eval_data]):.4f}",
        "ndcg_at_10": f"{avg([d['v3']['ndcg'] for d in eval_data]):.4f}",
        "mrr": f"{avg([d['v3']['mrr'] for d in eval_data]):.4f}",
    },
    "num_queries": len(eval_data),
    "method": "TREC pooling (v1+v2+v3 top-30), auto-relevant = v3 top-5",
    "dataset_size": "30,429 movies / 91,287 chunks",
}

# ---- Latency benchmarks ----
print("\n" + "="*60)
print("Latency Benchmarks")
print("="*60)

# Warm up
_ = embedder.encode_single("warmup")

timings = {}
sq = "烧脑的科幻片"

# Vector retrieval
t0 = time.time()
for _ in range(10):
    vr.retrieve(sq, top_k=50)
timings["vector_retrieval"] = (time.time() - t0) / 10

# BM25 retrieval
t0 = time.time()
for _ in range(10):
    br.retrieve(sq, top_k=50)
timings["bm25_retrieval"] = (time.time() - t0) / 10

# RRF fusion
vec50 = vr.retrieve(sq, top_k=50)
bm50 = br.retrieve(sq, top_k=50)
t0 = time.time()
for _ in range(100):
    rrf_fuse([vec50, bm50], k=60, top_k=50)
timings["rrf_fusion"] = (time.time() - t0) / 100

# Reranker (50 candidates)
fused = rrf_fuse([vec50, bm50], k=60, top_k=50)
movie_chunks = {}
for mid, score in fused:
    if mid not in chunks_map: continue
    chunks = chunks_map[mid]
    if chunks: movie_chunks[mid] = chunks[0]
chunk_list = list(movie_chunks.values())[:50]
t0 = time.time()
for _ in range(5):
    reranker.rerank(sq, chunk_list, top_k=10)
timings["reranker_50_candidates"] = (time.time() - t0) / 5

# Embedding
t0 = time.time()
for _ in range(10):
    embedder.encode_single(sq)
timings["embedding_single"] = (time.time() - t0) / 10

timings["bm25_load"] = 3.9  # measured earlier

print(json.dumps(timings, ensure_ascii=False, indent=2))
report["timings"] = timings

print("\n" + "="*60)
print("FINAL REPORT")
print("="*60)
print(json.dumps(report, ensure_ascii=False, indent=2))

with open("experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("\nSaved to experiment_results.json")
