# Create query_papers.py that implements queries for all five access patterns.

# Your script must support these commands:

# # Query 1: Recent papers in category
# python query_papers.py recent <category> [--limit 20] [--table TABLE]

# # Query 2: Papers by author
# python query_papers.py author <author_name> [--table TABLE]

# # Query 3: Get paper by ID
# python query_papers.py get <arxiv_id> [--table TABLE]

# # Query 4: Papers in date range
# python query_papers.py daterange <category> <start_date> <end_date> [--table TABLE]

# # Query 5: Papers by keyword
# python query_papers.py keyword <keyword> [--limit 20] [--table TABLE]


import sys, os, json, re, argparse
from datetime import datetime
from collections import Counter, defaultdict

import boto3
from botocore.exceptions import ClientError 
dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
from boto3.dynamodb.conditions import Key, Attr
import time

# Recent papers in category
def query_recent_in_category(table_name, category, limit=20):
    """
    Query 1: Browse recent papers in category.
    Uses: Main table partition key query with sort key descending.
    """
    response = dynamodb.Table(table_name).query(
        KeyConditionExpression=Key('PK').eq(f'CATEGORY#{category}'),
        ScanIndexForward=False,
        Limit=limit
    )
    return response['Items']

def query_papers_by_author(table_name, author_name):
    """
    Query 2: Find all papers by author.
    Uses: GSI1 (AuthorIndex) partition key query.
    """
    response = dynamodb.Table(table_name).query(
        IndexName='AuthorIndex',
        KeyConditionExpression=Key('GSI1PK').eq(f'AUTHOR#{author_name}')
    )
    return response['Items']

def get_paper_by_id(table_name, arxiv_id):
    """
    Query 3: Get specific paper by ID.
    Uses: GSI2 (PaperIdIndex) for direct lookup.
    """
    response = dynamodb.Table(table_name).query(
        IndexName='PaperIdIndex',
        KeyConditionExpression=Key('GSI2PK').eq(f'PAPER#{arxiv_id}')
    )
    return response['Items'][0] if response['Items'] else None

def query_papers_in_date_range(table_name, category, start_date, end_date):
    """
    Query 4: Papers in category within date range.
    Uses: Main table with composite sort key range query.
    """
    response = dynamodb.Table(table_name).query(
        KeyConditionExpression=(
            Key('PK').eq(f'CATEGORY#{category}') &
            Key('SK').between(f'{start_date}#', f'{end_date}#zzzzzzz')
        )
    )
    return response['Items']

def query_papers_by_keyword(table_name, keyword, limit=20):
    """
    Query 5: Papers containing keyword.
    Uses: GSI3 (KeywordIndex) partition key query.
    """
    response = dynamodb.Table(table_name).query(
        IndexName='KeywordIndex',
        KeyConditionExpression=Key('GSI3PK').eq(f'KEYWORD#{keyword.lower()}'),
        ScanIndexForward=False,
        Limit=limit
    )
    return response['Items']

def format_paper_output(item):
    """Formats a DynamoDB item into a clean dictionary for output."""
    if not item:
        return None
    return {
        "arxiv_id": item.get("arxiv_id"),
        "title": item.get("title"),
        "authors": item.get("authors"),
        "published": item.get("published"),
        "categories": item.get("categories")
    }

def main():
    parser = argparse.ArgumentParser(description="Query ArXiv paper data from DynamoDB.")
    subparsers = parser.add_subparsers(dest='command')

    # Query 1: Recent papers in category
    parser_recent = subparsers.add_parser('recent', help='Browse recent papers in category.')
    parser_recent.add_argument('category', type=str, help='Category to query.')
    parser_recent.add_argument('--limit', type=int, default=20, help='Number of papers to return.')
    parser_recent.add_argument('--table', type=str, default='arxiv_papers', help='DynamoDB table name.')

    # Query 2: Papers by author
    parser_author = subparsers.add_parser('author', help='Find all papers by author.')
    parser_author.add_argument('author_name', type=str, help='Author name to query.')
    parser_author.add_argument('--table', type=str, default='arxiv_papers', help='DynamoDB table name.')

    # Query 3: Get paper by ID
    parser_get = subparsers.add_parser('get', help='Get specific paper by ID.')
    parser_get.add_argument('arxiv_id', type=str, help='ArXiv ID of the paper.')
    parser_get.add_argument('--table', type=str, default='arxiv_papers', help='DynamoDB table name.')

    # Query 4: Papers in date range
    parser_daterange = subparsers.add_parser('daterange', help='Papers in category within date range.')
    parser_daterange.add_argument('category', type=str, help='Category to query.')
    parser_daterange.add_argument('start_date', type=str, help='Start date (YYYY-MM-DD).')
    parser_daterange.add_argument('end_date', type=str, help='End date (YYYY-MM-DD).')
    parser_daterange.add_argument('--table', type=str, default='arxiv_papers', help='DynamoDB table name.')

    # Query 5: Papers by keyword
    parser_keyword = subparsers.add_parser('keyword', help='Papers containing keyword.')
    parser_keyword.add_argument('keyword', type=str, help='Keyword to query.')
    parser_keyword.add_argument('--limit', type=int, default=20, help='Number of papers to return.')
    parser_keyword.add_argument('--table', type=str, default='arxiv_papers', help='DynamoDB table name.')

    args = parser.parse_args() 

    start_time = time.time()
    items = []
    output = {
        "query_type": None,
        "parameters": {},
    }

    if args.command == 'recent':
        output["query_type"] = "recent_in_category"
        output["parameters"] = {"category": args.category, "limit": args.limit}
        items = query_recent_in_category(args.table, args.category, args.limit)
    elif args.command == 'author':
        output["query_type"] = "papers_by_author"
        output["parameters"] = {"author_name": args.author_name}
        items = query_papers_by_author(args.table, args.author_name)
    elif args.command == 'get':
        output["query_type"] = "get_paper_by_id"
        output["parameters"] = {"arxiv_id": args.arxiv_id}
        item = get_paper_by_id(args.table, args.arxiv_id)
        items = [item] if item else []
    elif args.command == 'daterange':
        output["query_type"] = "papers_in_date_range"
        output["parameters"] = {"category": args.category, "start_date": args.start_date, "end_date": args.end_date}
        items = query_papers_in_date_range(args.table, args.category, args.start_date, args.end_date)
    elif args.command == 'keyword':
        output["query_type"] = "papers_by_keyword"
        output["parameters"] = {"keyword": args.keyword, "limit": args.limit}
        items = query_papers_by_keyword(args.table, args.keyword, args.limit)
    
    end_time = time.time()

    # Populate the rest of the output object
    output["results"] = [format_paper_output(item) for item in items]
    output["count"] = len(items)
    output["execution_time_ms"] = int((end_time - start_time) * 1000)

    # Print the final JSON object to stdout
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()      