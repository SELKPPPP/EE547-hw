# EE547 - Homework 3, Problem 2: DynamoDB Data Modeling


## Schema Design Decisions

### 1. Why did you choose your partition key structure?

Because it allows us store different data to one table while also efficiently querying them. 

### 2. How many GSIs did you create and why?

3 

1.  **`AuthorIndex` (GSI1)**:
    *   **Reason**: To efficiently search for all papers by author name.

2.  **`KeywordIndex` (GSI2)**:
    *   **Reason**: To enable efficient keyword-based searches for relevant papers.

3.  **`PaperIdIndex` (GSI3)**:
    *   **Reason**: To enable direct and efficient retrieval of complete information for individual papers using their unique `arxiv_id`.

### 3. What denormalization trade-offs did you make?

The core trade-off we made is to sacrifice storage space and write operation complexity in exchange for extremely high read performance and predictable low latency.

## Denormalization Analysis

*   **Average number of DynamoDB items per paper**: **14.9x**
*   **Storage multiplication factor**:  **15**。
*   **Which access patterns caused the most duplication?**:
    1.  **Keywords**: We extracted 10 keywords for each paper, resulting in a tenfold increase in data duplication from this process alone.
    2.  **Authors**: On average, each paper has five authors, resulting in five times the replication effort.

## Query Limitations

### 1. What queries are NOT efficiently supported by your schema?

*   **Aggregate Query**:
    *   Count the total number of papers published by a specific author.
    *   Identify the most frequently cited papers (assuming citation data is available)

### 2. Why are these difficult in DynamoDB?

DynamoDB does not have aggregate functions like `COUNT()`, `SUM()`, or `AVG()` found in SQL.

## When to Use DynamoDB

### 1. Based on this exercise, when would you choose DynamoDB over PostgreSQL?


*   **Large-scale, high-throughput OLTP (Online Transaction Processing) systems**: For example, shopping carts, user session management, game leaderboards, IoT data reception, and so on.
*   **The access mode is very clear and fixed.**: When you can precisely list all the queries an application requires.


### 2. What are the key trade-offs?

PostgreSQL offers exceptional query flexibility, though performance may vary depending on query complexity. DynamoDB sacrifices flexibility in exchange for ultimate performance on predefined queries.
PostgreSQL enforces a normalized data model to minimize data redundancy. DynamoDB encourages denormalization, using data redundancy to optimize read performance.

## EC2 Deployment

*   **EC2 Instance Public IP**: `18.223.166.67`
*   **IAM Role ARN Used**: `arn:aws:iam::679345828708:role/EC2-DynamoDB-Access-Role`