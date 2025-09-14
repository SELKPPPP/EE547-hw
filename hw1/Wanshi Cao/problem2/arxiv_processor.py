# arxiv_processor.py
# Standard library only: sys, json, urllib.request, xml.etree.ElementTree, datetime, time, re, os
import sys, os, re, time, json,datetime
import urllib.request, urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timezone


ARXIV_ENDPOINT = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
             'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
             'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
             'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
             'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
             'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
             'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
             'such', 'as', 'also', 'very', 'too', 'only', 'so', 'than', 'not'}



def UTC_time():
    # always UTC with trailing Z
    return datetime.now(timezone.utc).isoformat()

def log_line(log_path, msg):
    line = f"[{UTC_time()}] {msg}"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)

def build_url(query, max_results):
    # max_results given by para
    params = urllib.parse.urlencode({
        "search_query": query,
        "start":0,
        "max_results": max_results
    })
    return f"{ARXIV_ENDPOINT}?{params}"

def fetch_with_retry(url, log_path, max_attempts=3, wait_seconds=3):
    headers = {"User-Agent": "arxiv-processor/1.0 (mailto:example@example.com)"}
    attempt = 0
    while attempt< max_attempts:
        attempt += 1
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                #No error
                data = resp.read()
                return data.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            #429 error
            if e.code == 429:
                log_line(log_path, f"Rate limited (HTTP 429). Waiting {wait_seconds}s then retry {attempt}/{max_attempts}...")
                time.sleep(wait_seconds)
                continue
            else:
                log_line(log_path,f"NETWORK ERROR: HTTP {e.code} {e.reason}")
                return None
        except Exception as e:
            log_line(log_path, f"NETWORK ERROR: {e}")
            return None
    log_line(log_path, f"Rate limit retries exceeded after {max_attempts} attempts.")
    return None

# Text
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
UPPER_RE = re.compile(r"\b(?=\w*[A-Z])[\w\-]+\b")
NUMERIC_RE = re.compile(r"\b(?=\w*\d)[\w\-]+\b")
HYPHEN_RE = re.compile(r"\b\w+(?:-\w+)+\b")
SENT_SPLIT_RE = re.compile(r"[.!?]+")

#lower words
def tokenize(text):
    return [t.lower() for t in WORD_RE.findall(text)]

#text = "This is a test. Does it work? Yes!"
# ["This is a test", "Does it work", "Yes"] Three sentences
def sentence_tokens(text):
    parts =[s.strip() for s in SENT_SPLIT_RE.split(text)]
    return [p for p in parts if p] 

def abstract_stats(abstract):
    tokens = tokenize(abstract)
    total_words = len(tokens)
    unique_words = len(set(tokens))
    avg_word_length = (sum(len(t) for t in tokens) / total_words) if total_words else 0.0

    sents = sentence_tokens(abstract)
    sent_word_counts = [len(tokenize(s)) for s in sents]
    total_sentences = len(sent_word_counts)
    avg_words_per_sentence = (sum(sent_word_counts) / total_sentences) if total_sentences else 0.0

    return {
        "total_words": int(total_words),
        "unique_words": int(unique_words),
        "total_sentences": int(total_sentences),
        "avg_words_per_sentence": float(round(avg_words_per_sentence, 4)),
        "avg_word_length": float(round(avg_word_length, 4)),
    }

#corpus-level
#return freq without stopwords and every words in the abstract
def per_doc_word_counts_for_corpus(abstract):
    tokens = tokenize(abstract)
    seen_in_doc = set()
    freq = {}
    for w in tokens:
        if w in STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
        seen_in_doc.add(w)
    return freq, seen_in_doc

#Find uppercase terms CNN, LSTM, BERT-base
# numeric terms， ResNet50, GPT-3, 2D-CNN
#hyphenated terms self-attention, cross-entropy, state-of-the-art
def extract_terms(abstract):
    uppercase = set(UPPER_RE.findall(abstract))
    numeric = set(NUMERIC_RE.findall(abstract))
    hyphen = set(HYPHEN_RE.findall(abstract))
    return uppercase, numeric, hyphen

# XML parsing
#ArXiv API  Atom feed  <entry>  to  Python parse
def parse_entries(xml_text, log_path):
    #Invalid XML: If the API returns malformed XML, log the error and continue with other papers
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log_line(log_path, f"INVALID XML: {e}")
        return []

    entries = []
    for entry in root.findall("atom:entry", ATOM_NS):
        full_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
        arxiv_id = full_id.rsplit("/", 1)[-1] if "/" in full_id else full_id

        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        abstract = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ATOM_NS) or "").strip()
        updated = (entry.findtext("atom:updated", default="", namespaces=ATOM_NS) or "").strip()

        authors = []
        for a in entry.findall("atom:author", ATOM_NS):
            nm = a.findtext("atom:name", default="", namespaces=ATOM_NS) or ""
            nm = nm.strip()
            if nm:
                authors.append(nm)

        categories = []
        for c in entry.findall("atom:category", ATOM_NS):
            term = c.attrib.get("term", "").strip()
            if term:
                categories.append(term)

        # id、title、abstract
        #Missing fields: If a paper lacks required fields, skip it and log a warning
        if not (arxiv_id and title and abstract):
            log_line(log_path, f"WARNING: missing required fields; skipping one entry id='{arxiv_id}'")
            continue

        entries.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "categories": categories,
            "published": published,
            "updated": updated
        })
    return entries

#   main 
def main():
    if len(sys.argv) != 4:
        print("Usage: python arxiv_processor.py <search_query> <max_results(1-100)> <output_dir>", file=sys.stderr)
        sys.exit(2)

    search_query = sys.argv[1]
    max_results = int(sys.argv[2])
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "processing.log")

    t0 = time.time()
    log_line(log_path, f"Starting ArXiv query: {search_query}")

    url = build_url(search_query, max_results)
    xml_text = fetch_with_retry(url, log_path, max_attempts=3, wait_seconds=3)
    if xml_text is None:
        # Network errors: If the ArXiv API is unreachable, write error to log and exit with code 1
        log_line(log_path, "Network error. Exiting with code 1.")
        sys.exit(1)


    entries = parse_entries(xml_text, log_path)
    log_line(log_path, f"Fetched {len(entries)} results from ArXiv API")

    
    global_freq = {}         # word
    global_docfreq = {}      # word in doc
    seen_words_docs = []     # set
    uppercase_terms_all = set()
    numeric_terms_all = set()
    hyphen_terms_all = set()
    category_dist = {}       

    papers_out = []
    abs_lengths = []         #

    #every paper to paper_out  to paper.json
    for e in entries:
        log_line(log_path, f"Processing paper: {e['arxiv_id']}")

        # 1) abstract stats
        stats = abstract_stats(e["abstract"])
        papers_out.append({
            "arxiv_id": e["arxiv_id"],
            "title": e["title"],
            "authors": e["authors"],
            "abstract": e["abstract"],
            "categories": e["categories"],
            "published": e["published"],
            "updated": e["updated"],
            "abstract_stats": stats
        })

        # 2) corpus
        freq, docset = per_doc_word_counts_for_corpus(e["abstract"])
        for w, c in freq.items():
            global_freq[w] = global_freq.get(w, 0) + c
        for w in docset:
            global_docfreq[w] = global_docfreq.get(w, 0) + 1
        seen_words_docs.append(docset)

        # 3) 3 
        up, num, hyp = extract_terms(e["abstract"])
        uppercase_terms_all.update(up)
        numeric_terms_all.update(num)
        hyphen_terms_all.update(hyp)

        # 4) catagory
        for cat in e["categories"]:
            category_dist[cat] = category_dist.get(cat, 0) + 1

        abs_lengths.append(stats["total_words"])

    # File 1: papers.json
    papers_json_path = os.path.join(out_dir, "papers.json")
    with open(papers_json_path, "w", encoding="utf-8") as f:
        json.dump(papers_out, f, ensure_ascii=False, indent=2)

    # File 2: corpus_analysis.json
    total_abstracts = len(papers_out)
    total_words = sum(abs_lengths)
    unique_words_global = len(global_freq)
    avg_abstract_len = (total_words / total_abstracts) if total_abstracts else 0.0
    longest_abs = max(abs_lengths) if abs_lengths else 0
    shortest_abs = min(abs_lengths) if abs_lengths else 0

    # top-50 words up to down
    top_items = sorted(global_freq.items(), key=lambda x: (-x[1], x[0]))[:50]
    top_words = [
        {"word": w, "frequency": int(freq), "documents": int(global_docfreq.get(w, 0))}
        for w, freq in top_items
    ]

    corpus_payload = {
        "query": search_query,
        "papers_processed": total_abstracts,
        "processing_timestamp": UTC_time(),
        "corpus_stats": {
            "total_abstracts": total_abstracts,
            "total_words": int(total_words),
            "unique_words_global": int(unique_words_global),
            "avg_abstract_length": float(round(avg_abstract_len, 4)),
            "longest_abstract_words": int(longest_abs),
            "shortest_abstract_words": int(shortest_abs)
        },
        "top_50_words": top_words,
        "technical_terms": {
            "uppercase_terms": sorted(uppercase_terms_all),
            "numeric_terms": sorted(numeric_terms_all),
            "hyphenated_terms": sorted(hyphen_terms_all)
        },
        "category_distribution": dict(sorted(category_dist.items(), key=lambda x: (-x[1], x[0])))
    }

    corpus_json_path = os.path.join(out_dir, "corpus_analysis.json")
    with open(corpus_json_path, "w", encoding="utf-8") as f:
        json.dump(corpus_payload, f, ensure_ascii=False, indent=2)

    # Complete
    elapsed = time.time() - t0
    log_line(log_path, f"Completed processing: {total_abstracts} papers in [{elapsed:.2f}] seconds")

if __name__ == "__main__":
    main()
