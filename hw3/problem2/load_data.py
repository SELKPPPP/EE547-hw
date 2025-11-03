import sys, os, json, re, argparse
from datetime import datetime
from collections import Counter, defaultdict

import boto3
from botocore.exceptions import ClientError 

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'can', 'this', 'that', 'these', 'those', 'we', 'our', 'use', 'using',
    'based', 'approach', 'method', 'paper', 'propose', 'proposed', 'show'
}

# Design a DynamoDB table schema that efficiently supports these required query patterns:
# Browse recent papers by category (e.g., “Show me latest ML papers”)
# Find all papers by a specific author
# Get full paper details by arxiv_id
# List papers published in a date range within a category
# Search papers by keyword (extracted from abstract)

# Design Requirements:

# Define partition key and sort key for main table
# Design Global Secondary Indexes (GSIs) to support all access patterns
# Implement denormalization strategy for efficient queries
# Document trade-offs in your schema design

# Example Schema Structure:

# # Main Table Item
# {
#   "PK": "CATEGORY#cs.LG",
#   "SK": "2023-01-15#2301.12345",
#   "arxiv_id": "2301.12345",
#   "title": "Paper Title",
#   "authors": ["Author1", "Author2"],
#   "abstract": "Full abstract text...",
#   "categories": ["cs.LG", "cs.AI"],
#   "keywords": ["keyword1", "keyword2"],
#   "published": "2023-01-15T10:30:00Z"
# }

# # GSI1: Author access
# {
#   "GSI1PK": "AUTHOR#Author1",
#   "GSI1SK": "2023-01-15",
#   # ... rest of paper data
# }

# # Additional GSIs as needed for other access patterns



#  Create DynamoDB table with appropriate partition/sort keys
def create_dynamodb_table(table_name, region, dynamodb=None):

    if not dynamodb:
        dynamodb = boto3.resource('dynamodb', region_name=region)

    try:
        print(f"Creating DynamoDB table: {table_name}")
        print("Creating GSIs: AuthorIndex, KeywordIndex, PaperIdIndex")
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {'AttributeName': 'PK', 'KeyType': 'HASH'},  # Partition key
                {'AttributeName': 'SK', 'KeyType': 'RANGE'}   # Sort key
            ],
            AttributeDefinitions=[
                {'AttributeName': 'PK', 'AttributeType': 'S'},
                {'AttributeName': 'SK', 'AttributeType': 'S'},
                {'AttributeName': 'GSI1PK', 'AttributeType': 'S'},
                {'AttributeName': 'GSI1SK', 'AttributeType': 'S'},
                {'AttributeName': 'GSI2PK', 'AttributeType': 'S'},
                {'AttributeName': 'GSI2SK', 'AttributeType': 'S'},
                {'AttributeName': 'GSI3PK', 'AttributeType': 'S'},
            ],
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'AuthorIndex',
                    'KeySchema': [
                        {'AttributeName': 'GSI1PK', 'KeyType': 'HASH'},
                        {'AttributeName': 'GSI1SK', 'KeyType': 'RANGE'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                },
                {
                    'IndexName': 'KeywordIndex',
                    'KeySchema': [
                        {'AttributeName': 'GSI2PK', 'KeyType': 'HASH'},
                        {'AttributeName': 'GSI2SK', 'KeyType': 'RANGE'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                },
                {
                    'IndexName': 'PaperIdIndex',
                    'KeySchema': [
                        {'AttributeName': 'GSI3PK', 'KeyType': 'HASH'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                }
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        table.wait_until_exists()
        print(f"Table {table_name} and GSIs created successfully.")
        return table
    except dynamodb.meta.client.exceptions.ResourceInUseException:
        print(f"Table {table_name} already exists. Skipping creation.")
        return dynamodb.Table(table_name)

# Extract keywords from abstracts (top 10 most frequent words, excluding stopwords)
def extract_keywords(abstract):
    words = re.findall(r'\b\w+\b', abstract.lower())
    filtered_words = [word for word in words if word not in STOPWORDS]
    word_counts = Counter(filtered_words)
    most_common = word_counts.most_common(10)
    return [word for word, count in most_common]



# Transform paper data from HW#1 format to DynamoDB items
# Implement denormalization as needed
def transform_paper_to_item(paper):
    arxiv_id = paper['arxiv_id']
    published_date = paper['published'][:10]  # YYYY-MM-DD
    keywords = extract_keywords(paper['abstract'])

    # Create a single base item with all paper details.
    # This avoids duplicating data in memory.
    base_item = {
        'arxiv_id': paper['arxiv_id'],
        'title': paper['title'],
        'authors': paper['authors'],
        'abstract': paper['abstract'],
        'categories': paper['categories'],
        'keywords': keywords,
        'published': paper['published']
    }

    items = []
    stats = defaultdict(int)

    # 1. Main items: One per category for category-based queries
    for category in paper['categories']:
        item = base_item.copy()
        item['PK'] = f"CATEGORY#{category}"
        item['SK'] = f"{published_date}#{arxiv_id}"
        items.append(item)
        stats['category_items'] += 1

    # 2. GSI1 items: One per author for author-based queries
    for author in paper['authors']:
        item = base_item.copy()
        # Each author item needs a UNIQUE primary key
        item['PK'] = f"PAPER#{arxiv_id}#AUTHOR#{author}"
        item['SK'] = "AUTHOR"
        # Add GSI keys for the author index
        item['GSI1PK'] = f"AUTHOR#{author}"
        item['GSI1SK'] = f"{published_date}#{arxiv_id}"
        items.append(item)
        stats['author_items'] += 1

    # 3. GSI2 items: One per keyword for keyword-based queries
    for keyword in keywords:
        item = base_item.copy()
        # Each keyword item needs a UNIQUE primary key
        item['PK'] = f"PAPER#{arxiv_id}#KEYWORD#{keyword}"
        item['SK'] = "KEYWORD"
        # Add GSI keys for the keyword index
        item['GSI2PK'] = f"KEYWORD#{keyword}"
        item['GSI2SK'] = f"{published_date}#{arxiv_id}"
        items.append(item)
        stats['keyword_items'] += 1

    # 4. GSI3 item: One item for direct paper lookup by ID
    item = base_item.copy()
    # This item is for direct lookup via GSI3
    item['PK'] = f"PAPER#{arxiv_id}"
    item['SK'] = "METADATA"
    # Add GSI key for the paper ID index
    item['GSI3PK'] = f"PAPER#{arxiv_id}"
    items.append(item)
    stats['paper_id_items'] += 1

    return items, stats


# Batch write items to DynamoDB (use batch_write_item for efficiency)
def batch_write_items(table, items):
    with table.batch_writer() as batch:
        for item in items:
            batch.put_item(Item=item)
    print(f"Inserted {len(items)} items into table {table.name}.")

def main():    
    parser = argparse.ArgumentParser(description="Load ArXiv paper data into DynamoDB.")
    parser.add_argument("--papers_json_path", type=str, help="Path to the papers.json file.")
    parser.add_argument("--table_name", type=str, help="Name of the DynamoDB table.")
    parser.add_argument("--region", type=str, default="us-east-2", help="AWS region for DynamoDB.")
    args = parser.parse_args()

    # Create DynamoDB table
    table = create_dynamodb_table(args.table_name, args.region)

    # Load papers from JSON file
    print(f"Loading papers from {args.papers_json_path}")
    with open(args.papers_json_path, 'r') as f:
        papers = json.load(f)

    all_items = []
    total_stats = defaultdict(int)

    print(f"Transforming and preparing {len(papers)} papers for insertion.")
    for paper in papers:
        items, stats = transform_paper_to_item(paper)
        all_items.extend(items)
        for key, value in stats.items():
            total_stats[key] += value

    # Batch write items to DynamoDB
    batch_write_items(table, all_items)

    num_papers = len(papers)
    num_items = len(all_items)  
    denormalization_factor = num_items / num_papers if num_papers > 0 else 0         


    print(f"Loaded {num_papers} papers")
    print(f"Created {num_items} DynamoDB items (denormalized)")
    print(f"Denormalization factor: {denormalization_factor:.1f}x")


    print("\nStorage breakdown:")
    cat_avg = total_stats['category_items'] / num_papers if num_papers > 0 else 0
    auth_avg = total_stats['author_items'] / num_papers if num_papers > 0 else 0
    key_avg = total_stats['keyword_items'] / num_papers if num_papers > 0 else 0
    id_avg = total_stats['paper_id_items'] / num_papers if num_papers > 0 else 0

    print(f"  - Category items: {total_stats['category_items']} ({cat_avg:.1f} per paper avg)")
    print(f"  - Author items:   {total_stats['author_items']} ({auth_avg:.1f} per paper avg)")
    print(f"  - Keyword items:  {total_stats['keyword_items']} ({key_avg:.1f} per paper avg)")
    print(f"  - Paper ID items: {total_stats['paper_id_items']} ({id_avg:.1f} per paper)")

if __name__ == "__main__":
    main()    
    