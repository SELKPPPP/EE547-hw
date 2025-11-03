# Create api_server.py that exposes query functionality via HTTP endpoints.

# Required Endpoints:

# GET /papers/recent?category={category}&limit={limit}
# Returns recent papers in category
# Default limit: 20
# GET /papers/author/{author_name}
# Returns all papers by author
# GET /papers/{arxiv_id}
# Returns full paper details by ID
# GET /papers/search?category={category}&start={date}&end={date}
# Returns papers in date range
# GET /papers/keyword/{keyword}?limit={limit}
# Returns papers matching keyword
# Default limit: 20
# Implementation Requirements:

# Use only Python standard library http.server (no Flask/FastAPI)
# Accept port number as command line argument (default 8080)
# Return JSON responses with proper HTTP status codes
# Handle errors gracefully (404 for not found, 500 for server errors)
# Log requests to stdout


import sys, json, argparse


import boto3
from botocore.exceptions import ClientError 
from boto3.dynamodb.conditions import Key, Attr

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import urllib.parse
import time

from query_papers import (
    query_recent_in_category,
    query_papers_by_author,
    get_paper_by_id,
    query_papers_in_date_range,
    query_papers_by_keyword
)

DYNAMODB = boto3.resource('dynamodb', region_name='us-east-2')
TABLE_NAME = 'arxiv_papers' 

class PaperQueryHandler(BaseHTTPRequestHandler):
    def _send_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

    def do_GET(self):
        try:
            parsed_path = urlparse(self.path)
            path_components = parsed_path.path.strip('/').split('/')
            query_params = parse_qs(parsed_path.query)
            
            # GET /papers/recent?category={category}&limit={limit}
            if path_components == ['papers', 'recent']:
                category = query_params.get('category', [None])[0]
                if not category:
                    self._send_response(400, {"error": "Missing 'category' query parameter"})
                    return
                limit = int(query_params.get('limit', [20])[0])
                results = query_recent_in_category(TABLE_NAME, category, limit)
                self._send_response(200, {"category": category, "papers": results, "count": len(results)})

            # GET /papers/author/{author_name}
            elif len(path_components) == 3 and path_components[:2] == ['papers', 'author']:
                author_name = path_components[2]
                results = query_papers_by_author(TABLE_NAME, author_name)
                self._send_response(200, {"author": urllib.parse.unquote(author_name), "papers": results, "count": len(results)})

            # GET /papers/keyword/{keyword}?limit={limit}
            elif len(path_components) == 3 and path_components[:2] == ['papers', 'keyword']:
                keyword = path_components[2]
                limit = int(query_params.get('limit', [20])[0])
                results = query_papers_by_keyword(TABLE_NAME, keyword, limit)
                self._send_response(200, {"keyword": keyword, "papers": results, "count": len(results)})

            # GET /papers/search?category={category}&start={date}&end={date}
            elif path_components == ['papers', 'search']:
                category = query_params.get('category', [None])[0]
                start_date = query_params.get('start', [None])[0]
                end_date = query_params.get('end', [None])[0]
                if not all([category, start_date, end_date]):
                    self._send_response(400, {"error": "Missing 'category', 'start', or 'end' query parameter"})
                    return
                results = query_papers_in_date_range(TABLE_NAME, category, start_date, end_date)
                self._send_response(200, {"category": category, "start": start_date, "end": end_date, "papers": results, "count": len(results)})

            # GET /papers/{arxiv_id}
            elif len(path_components) == 2 and path_components[0] == 'papers':
                arxiv_id = path_components[1]
                result = get_paper_by_id(TABLE_NAME, arxiv_id)
                if result:
                    self._send_response(200, result)
                else:
                    self._send_response(404, {"error": f"Paper with ID '{arxiv_id}' not found"})
            
            else:
                self._send_response(404, {"error": "Endpoint not found"})

        except Exception as e:
            print(f"Server Error: {e}", file=sys.stderr)
            self._send_response(500, {"error": "An internal server error occurred."})

def run_server(port):
    server_address = ('', port)
    httpd = HTTPServer(server_address, PaperQueryHandler)
    print(f"Starting server on http://localhost:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="API server for querying ArXiv papers from DynamoDB.")
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on.')
    parser.add_argument('--table', type=str, default='arxiv_papers', help='DynamoDB table name.')
    args = parser.parse_args()
    
    TABLE_NAME = args.table
    run_server(args.port)