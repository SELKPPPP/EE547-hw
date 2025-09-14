import sys, os, re, time, json,datetime
import urllib.request, urllib.error
import xml.etree.ElementTree as ET
from collections import Counter
from itertools import combinations
from datetime import datetime, timezone



def jaccard_similarity(doc1_words, doc2_words):
    """Calculate Jaccard similarity between two documents."""
    set1 = set(doc1_words)
    set2 = set(doc2_words)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0


def UTC_time():
    # always UTC with trailing Z
    return datetime.now(timezone.utc).isoformat()

def log(msg: str):
    print(f"[{UTC_time()}] {msg}", flush=True)



WORD_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
SENT_SPLIT_RE = re.compile(r"[.!?]+")

def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))

def sentence_count(text: str) -> int:
    return len([s for s in SENT_SPLIT_RE.split(text) if s.strip()])

def avg_word_length(words) -> float:
    return round(sum(len(w) for w in words) / len(words), 4) if words else 0.0

def ngrams(words, n):
    return zip(*[words[i:] for i in range(n)])

#lower words
def tokenize(text):
    return [t.lower() for t in WORD_RE.findall(text)]


def main():
    shared = "/shared"
    proc_dir = os.path.join(shared, "processed")
    analysis_dir = os.path.join(shared, "analysis")
    status_dir = os.path.join(shared, "status")

    os.makedirs(analysis_dir, exist_ok=True)

    # Wait for processor completion marker
    proc_marker = os.path.join(status_dir, "process_complete.json")
    log(f"Analyzer waiting for {proc_marker}")
    while not os.path.exists(proc_marker):
        time.sleep(1)

    # Read processed/*.json
    if not os.path.isdir(proc_dir):
        log(f"ERROR: processed directory not found: {proc_dir}")
        minimal = {
            "processing_timestamp": UTC_time(),
            "documents_processed": 0,
            "total_words": 0,
            "unique_words": 0,
            "top_100_words": [],
            "document_similarity": [],
            "top_bigrams": [],
            "top_trigrams": [],
            "readability": {"avg_sentence_length": 0.0, "avg_word_length": 0.0, "complexity_score": 0.0},
        }
        with open(os.path.join(analysis_dir, "final_report.json"), "w", encoding="utf-8") as f:
            json.dump(minimal, f, ensure_ascii=False, indent=2)
        return

    files = sorted([f for f in os.listdir(proc_dir) if f.lower().endswith(".json")])
    docs = []
    for fname in files:
        path = os.path.join(proc_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            text = obj.get("text", "")
            words = tokenize(text)
            sents = sentence_count(text)
            docs.append({"name": fname, "words": words, "sentences": sents})
        except Exception as e:
            log(f"ERROR reading {fname}: {e}")

    # 
    if not docs:
        minimal = {
            "processing_timestamp": UTC_time(),
            "documents_processed": 0,
            "total_words": 0,
            "unique_words": 0,
            "top_100_words": [],
            "document_similarity": [],
            "top_bigrams": [],
            "top_trigrams": [],
            "readability": {"avg_sentence_length": 0.0, "avg_word_length": 0.0, "complexity_score": 0.0},
        }
        with open(os.path.join(analysis_dir, "final_report.json"), "w", encoding="utf-8") as f:
            json.dump(minimal, f, ensure_ascii=False, indent=2)
        return

    #  word frequency (top 100) no doc
    global_words = []
    total_sentences = 0
    for d in docs:
        global_words.extend(d["words"])
        total_sentences += d["sentences"]

    total_words = len(global_words)
    unique_words = len(set(global_words))
    word_ctr = Counter(global_words)
    top100 = word_ctr.most_common(100)
    top_100_words = [
        {"word": w, "count": c, "frequency": round(c / total_words, 3) if total_words else 0.0}
        for w, c in top100
    ]

    # document similarity (Jaccard)
    doc_sim = []
    for d1, d2 in combinations(docs, 2):
        sim = jaccard_similarity(d1["words"], d2["words"])
        doc_sim.append({"doc1": d1["name"], "doc2": d2["name"], "similarity": round(sim, 3)})

    #n-grams (bigrams & trigrams) 
    bi_ctr, tri_ctr = Counter(), Counter()
    for d in docs:
        ws = d["words"]
        bi_ctr.update([" ".join(t) for t in ngrams(ws, 2)])
        tri_ctr.update([" ".join(t) for t in ngrams(ws, 3)])
    top_bigrams = [{"bigram": bg, "count": c} for bg, c in bi_ctr.most_common(50)]
    top_trigrams = [{"trigram": tg, "count": c} for tg, c in tri_ctr.most_common(50)]

    #readability metrics
    avg_sentence_len = round(total_words / total_sentences, 4) if total_sentences else 0.0
    avg_word_len = avg_word_length(global_words)
    complexity_score = round(avg_sentence_len * avg_word_len, 4)

    # Final report
    report = {
        "processing_timestamp": UTC_time(),
        "documents_processed": len(docs),
        "total_words": total_words,
        "unique_words": unique_words,
        "top_100_words": top_100_words,
        "document_similarity": doc_sim,
        "top_bigrams": top_bigrams,
        "top_trigrams": top_trigrams,
        "readability": {
            "avg_sentence_length": avg_sentence_len,
            "avg_word_length": avg_word_len,
            "complexity_score": complexity_score,
        },
    }

    os.makedirs(analysis_dir, exist_ok=True)
    out_path = os.path.join(analysis_dir, "final_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        print(report,"have report")
        json.dump(report, f, ensure_ascii=False, indent=2)

    log(f"Analyzer completed")

if __name__ == "__main__":
    main()