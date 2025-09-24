
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, unquote
import json, re, sys, os
from datetime import datetime

# ------------ Load Data ------------
DATA_DIR = os.environ.get("ARXIV_DATA_DIR", ".")
PAPERS_PATH = os.path.join(DATA_DIR, "sample_data/papers.json")
CORPUS_PATH = os.path.join(DATA_DIR, "sample_data/corpus_analysis.json")

def load_json_safe(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[WARN] Missing file: {path}. Using default.", flush=True)
        return default
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}. Using default.", flush=True)
        return default

PAPERS = load_json_safe(PAPERS_PATH, [])
CORPUS = load_json_safe(CORPUS_PATH, {})

# index by arxiv_id for O(1) lookup
PAPER_BY_ID = {}
for p in PAPERS:
    if "arxiv_id" in p:
        PAPER_BY_ID[p["arxiv_id"]] = p

# ------------ Utilities ------------
def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(method, path, status, extra=""):
    # [2025-09-16 14:30:22] GET /papers - 200 OK (15 results)
    msg = f"[{now_ts()}] {method} {path} - {status}"
    if extra:
        msg += f" {extra}"
    print(msg, flush=True)


def write_json(handler, obj, status=200):
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
    except Exception as e:
        print(f"[WARN] Failed to send response: {e}", flush=True)

def write_error(handler, status, message):
    write_json(handler, {"error": message}, status=status)

# word-boundary regex counter (case-insensitive)
def count_term(text, term):
    if not text or not term:
        return 0
    # \bterm\b, escape special chars in term
    pattern = r"\b" + re.escape(term) + r"\b"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


# ------------ Search Logic ------------
#Case-insensitive search in titles and abstracts
#Count term frequency as match score
#Support multi-word queries (search for all terms)
def search_papers(query):

    raw = query.strip()
    # split on whitespace, remove empty
    terms = [t for t in re.split(r"\s+", raw) if t]
    #"machine learning" -> ["machine", "learning"]
    if not terms:
        return None  # signal malformed

    results = []
    for p in PAPERS:
        title = p.get("title", "") or ""
        abstract = p.get("abstract", "") or ""
        # AND semantics: every term must appear at least once in title or abstract
        per_paper_score = 0
        matched_fields = set()

        scored = True
        for t in terms:
            ct_title = count_term(title, t)
            ct_abs = count_term(abstract, t)
            if ct_title + ct_abs == 0:
                scored = False
                break
            per_paper_score += (ct_title + ct_abs)
            if ct_title > 0:
                matched_fields.add("title")
            if ct_abs > 0:
                matched_fields.add("abstract")

        if scored and per_paper_score > 0:
            results.append({
                "arxiv_id": p.get("arxiv_id"),
                "title": title,
                "match_score": per_paper_score,
                "matches_in": sorted(list(matched_fields))
            })

    # Descending by match_score, then ascending by title
    results.sort(key=lambda x: (-x["match_score"], x["title"]))
    return {"query": query, "results": results}

# ------------ HTTP Handler ------------
class ArxivHandler(BaseHTTPRequestHandler):
    # Override server version string
    server_version = "ArxivStdlibHTTP.problem1/1.0"

    def do_GET(self):
        try:
            parsed_url = urlparse(self.path)
            #ParseResult(
            # scheme='http',
            # netloc='localhost:8080',
            # path='/search',
            # params='',
            # query='q=machine%20learning&sort=desc',
            # fragment=''
            #)
            
            path = parsed_url.path


            query_params = parse_qs(parsed_url.query)

            #machine%20learning -> machine learning
            #sort=desc
            #limit=10

            # /papers  (list)
            # Returns a list of papers with basic info
            if path == "/papers":
                #http://localhost:8080/papers
                items = [
                    {
                        "arxiv_id": p.get("arxiv_id"),
                        "title": p.get("title"),
                        "authors": p.get("authors", []),
                        "categories": p.get("categories", []),
                    }
                    for p in PAPERS
                ]
                write_json(self, items, status=200)
                log_line("GET", path, "200 OK", f"({len(items)} results)")
                # [2025-09-16 14:30:22] GET /papers - 200 OK (15 results)
                return

            # /papers/{id}
            if path.startswith("/papers/"):
                #http://localhost:8080/papers/1234.56789
                # extract id
                arxiv_id = unquote(path[len("/papers/"):])
                # Get paper by id
                paper = PAPER_BY_ID.get(arxiv_id)
                if not paper:
                    write_error(self, 404, "Paper not found")
                    log_line("GET", path, "404 Not Found")
                    return
                write_json(self, paper, status=200)
                log_line("GET", path, "200 OK")
                return
             

            # /search?q=...
            if path == "/search":
                #http://localhost:8080/search?q=machine%20learning

                # Get Maachine Learning
                # from query param "q"
                query = query_params.get("q", [""])[0]
                
                result = search_papers(query)
                if result is None:
                    write_error(self, 400, "Malformed or empty search query")
                    log_line("GET", f"{path}?{parsed_url.query}", "400 Bad Request")
                    return
                write_json(self, result, status=200)
                log_line("GET", f"{path}?{parsed_url.query}", "200 OK",
                         f"({len(result['results'])} results)")
                return

            # /stats
            if path == "/stats":
                # http://localhost:8080/stats
                # Returns corpus statistics as a JSON object.
                # Example structure:
                # {
                #   "total_papers": int,
                #   "category_counts": { "cs.AI": int, ... },
                #   "author_counts": { "John Doe": int, ... },
                #   ...
                # }
                write_json(self, CORPUS if isinstance(CORPUS, dict) else {}, status=200)
                log_line("GET", path, "200 OK")
                return

            # invalid endpoint
            write_error(self, 404, "Endpoint not found")
            log_line("GET", path, "404 Not Found")

        except Exception as e:
            # 500 with JSON error message
            write_error(self, 500, f"Server error: {repr(e)}")
            log_line("GET", self.path, "500 Internal Server Error")

    # slient log to avoid default stdout logging
    def log_message(self, format, *args):
        return

# ------------ Main Server Loop ------------
def main():
    # Default port 8080, can override with command line argument
    port = 8080
    if len(sys.argv) >= 2:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"[WARN] Invalid port '{sys.argv[1]}', fallback to 8080.", flush=True)

    server = HTTPServer(("0.0.0.0", port), ArxivHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("[INFO] Server stopped.", flush=True)

if __name__ == "__main__":
    main()
