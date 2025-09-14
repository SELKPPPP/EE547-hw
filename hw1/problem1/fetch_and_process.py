import sys, os, json, time, datetime, re
from urllib import request, error
from datetime import datetime, timezone


def UTC_time():
    # always UTC with trailing Z
    return datetime.now(timezone.utc).isoformat()

def is_text_content(content_type):
    if (content_type or "").lower().find("text") != -1: #Find text
        return True
    else:
        return False

#extract_charset("text/plain; Charset=iso-8859-1") 
#return iso-8859-1
def extract_charset(content_type):
    if not content_type:
        return None
    m = re.search(r"charset=([^;]+)",content_type, re.I) 
    if m:
        return m.group(1).strip() 
    else:
        return None

def count_words_from_text(text):
    return len(re.findall(r"[A-Za-z0-9]+",text))

def fetch(url, timeout_sec=10):
    started = time.time()
    ts = UTC_time()
    try:
        req = request.Request(url, method="GET") #Get the url
        with request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read()  # bytes
            response_ms = (time.time() - started)*1000.0
            ctype = resp.headers.get("Content-Type", "")
            status = resp.getcode()

            content_len = len(body)

            word_cnt = None
            if is_text_content(ctype):
                charset = extract_charset(ctype) or "utf-8" #decode from byte to string 
                try:
                    text = body.decode(charset, errors="ignore")
                except Exception:
                    text = body.decode("utf-8", errors="ignore")
                word_cnt = count_words_from_text(text)

            
            #return response.json and error line

            return {
                "url": url,
                "status_code": int(status) if status is not None else None,
                "response_time_ms": response_ms,
                "content_length": content_len,
                "word_count": word_cnt,
                "timestamp": ts,
                "error": None,
            }, None 

    except error.HTTPError as e:
        # 4xx/5xx error
        response_ms = (time.time() - started)* 1000.0
        try:
            body = e.read() or b""
        except Exception:
            body = b""
        ctype = e.headers.get("Content-Type", "") if hasattr(e, "headers") else ""
        content_len = len(body)

        word_cnt = None

        if is_text_content(ctype):
            charset = extract_charset(ctype) or "utf-8"
            try:
                text = body.decode(charset, errors="ignore")
            except Exception:
                text = body.decode("utf-8", errors="ignore")
            word_cnt = count_words_from_text(text)

        return {
            "url": url,
            "status_code": int(e.getcode())if e.getcode() is not None else None,
            "response_time_ms": response_ms,
            "content_length": content_len,
            "word_count": word_cnt,
            "timestamp": ts,
            "error": None, 
        }, None

    except Exception as e:
        response_ms = (time.time() - started)*1000.0
        msg = str(e) or e.__class__.__name__ #incase the e is None
        return {
            "url": url,
            "status_code": None,
            "response_time_ms": response_ms,
            "content_length": 0,
            "word_count": None,
            "timestamp": ts,
            "error": msg,
        }, f"[{ts}] [{url}]: {msg}"

def main():
    #only two commands
    if len(sys.argv) != 3:
        print("Usage: python fetch_and_process.py <input_urls_file> <output_dir>", file=sys.stderr)
        sys.exit(2)

    in_file = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    with open(in_file, "r", encoding="utf-8") as f:
        urls = [ln.strip() for ln in f if ln.strip()]

    processing_start = UTC_time()

    responses =[]
    errors_lines = []
    status_counts = {}
    total_bytes = 0
    success_times =[]
    success_cnt = 0
    fail_cnt = 0

    for url in urls:
        rec, err_line = fetch(url, timeout_sec=10)
        responses.append(rec)

        # No error
        if rec["error"] is None:
            success_cnt += 1
            total_bytes += int(rec["content_length"] or 0)
            success_times.append(float(rec["response_time_ms"]))
            code = str(rec["status_code"])if rec["status_code"] is not None else "unknown"
            status_counts[code] = status_counts.get(code, 0) + 1
        #Error
        else:
            fail_cnt += 1
            if err_line:
                errors_lines.append(err_line)

    processing_end = UTC_time()
    avg_ms = (sum(success_times) / len(success_times)) if success_times else 0.0

    # Write files
    with open(os.path.join(out_dir, "responses.json"), "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    summary = {
        "total_urls": len(urls),
        "successful_requests": success_cnt,
        "failed_requests": fail_cnt,
        "average_response_time_ms": avg_ms,
        "total_bytes_downloaded": total_bytes,
        "status_code_distribution": {k: int(v) for k, v in sorted(status_counts.items())},
        "processing_start": processing_start,
        "processing_end": processing_end,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "errors.log"), "w", encoding="utf-8") as f:
        for line in errors_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    main()
