# Snowflake Ingestion Guide

This guide explains how to use the Snowflake ingestor to extract data from Snowflake data warehouses into Semantica.

## Table of Contents

- [Installation](#installation)
- [Authentication Methods](#authentication-methods)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Installation

Install Semantica with Snowflake support:

```bash
# Install with Snowflake support only
pip install semantica[db-snowflake]

# Or install with all database connectors
pip install semantica[db-all]
```

This will install:
- `snowflake-connector-python>=3.0.0`
- `cryptography>=3.4.0` (required for key-pair authentication)

## Authentication Methods

### 1. Password Authentication (Default)

The simplest authentication method using username and password:

```python
from semantica.ingest import SnowflakeIngestor

ingestor = SnowflakeIngestor(
    account="myaccount",  # Your Snowflake account identifier
    user="myuser",
    password="mypassword",
    warehouse="COMPUTE_WH",
    database="MYDB",
    schema="PUBLIC"
)
```

**Environment Variables:**

```bash
export SNOWFLAKE_ACCOUNT="myaccount"
export SNOWFLAKE_USER="myuser"
export SNOWFLAKE_PASSWORD="mypassword"
export SNOWFLAKE_WAREHOUSE="COMPUTE_WH"
export SNOWFLAKE_DATABASE="MYDB"
export SNOWFLAKE_SCHEMA="PUBLIC"
```

**Note:** For key-pair authentication, you must provide the `private_key_path` parameter directly in code. Environment variables are supported for account, user, warehouse, database, schema, role, authenticator, and token parameters.

```python
# Now you can omit supported parameters - they'll be read from environment
ingestor = SnowflakeIngestor()
```

### 2. Key-Pair Authentication

More secure authentication using RSA key pairs:

**Generate Key Pair:**

```bash
# Generate private key
openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out rsa_key.p8 -nocrypt

# Generate public key
openssl rsa -in rsa_key.p8 -pubout -out rsa_key.pub
```

**Add public key to Snowflake:**

```sql
ALTER USER myuser SET RSA_PUBLIC_KEY='MIIBIjANBgkqh...';
```

**Use in Python:**

```python
from semantica.ingest import SnowflakeIngestor

ingestor = SnowflakeIngestor(
    account="myaccount",
    user="myuser",
    private_key_path="/path/to/rsa_key.p8",
    warehouse="COMPUTE_WH",
    database="MYDB"
)
```

**With encrypted private key:**

```python
ingestor = SnowflakeIngestor(
    account="myaccount",
    user="myuser",
    private_key_path="/path/to/rsa_key.p8",
    private_key_passphrase="my_passphrase",
    warehouse="COMPUTE_WH"
)
```

### 3. OAuth Authentication

Use OAuth tokens for authentication:

```python
ingestor = SnowflakeIngestor(
    account="myaccount",
    user="myuser",
    authenticator="oauth",
    token="your_oauth_token",
    warehouse="COMPUTE_WH"
)
```

### 4. SSO Authentication (External Browser)

For organizations using SSO:

```python
ingestor = SnowflakeIngestor(
    account="myaccount",
    user="myuser",
    authenticator="externalbrowser",
    warehouse="COMPUTE_WH"
)
# This will open a browser for SSO login
```

## Basic Usage

### Ingest a Table

```python
from semantica.ingest import SnowflakeIngestor

ingestor = SnowflakeIngestor(
    account="myaccount",
    user="myuser",
    password="mypassword",
    warehouse="COMPUTE_WH",
    database="MYDB",
    schema="PUBLIC"
)

# Ingest entire table
data = ingestor.ingest_table("CUSTOMERS")

print(f"Retrieved {data.row_count} rows")
print(f"Columns: {data.columns}")
print(f"First row: {data.data[0]}")
```

### Ingest with Filters

```python
# With WHERE clause
data = ingestor.ingest_table(
    "CUSTOMERS",
    where="COUNTRY = 'USA' AND CREATED_DATE > '2024-01-01'"
)

# With limit and offset for pagination
data = ingestor.ingest_table(
    "CUSTOMERS",
    limit=10000,
    offset=0,
    order_by="CREATED_DATE DESC"
)
```

### Execute Custom Queries

```python
# Simple query
data = ingestor.ingest_query("""
    SELECT 
        CUSTOMER_ID,
        SUM(AMOUNT) AS TOTAL_AMOUNT
    FROM SALES
    WHERE DATE >= '2024-01-01'
    GROUP BY CUSTOMER_ID
    HAVING SUM(AMOUNT) > 1000
""")

# Parameterized query
data = ingestor.ingest_query(
    "SELECT * FROM SALES WHERE DATE > %(start_date)s AND REGION = %(region)s",
    params={
        "start_date": "2024-01-01",
        "region": "WEST"
    }
)
```

### Large Result Sets with Batching

```python
# Fetch in batches to manage memory
data = ingestor.ingest_query(
    "SELECT * FROM LARGE_TABLE",
    batch_size=5000  # Fetch 5000 rows at a time
)

print(f"Total rows: {data.row_count}")
```

## Advanced Features

### Schema Introspection

```python
# Get table schema
schema = ingestor.get_table_schema("CUSTOMERS")

for column in schema["columns"]:
    print(f"{column['name']}: {column['type']} (nullable: {column['nullable']})")

print(f"Primary keys: {schema['primary_keys']}")
```

### List All Tables

```python
# List tables in current schema
tables = ingestor.list_tables()
print(f"Found {len(tables)} tables: {tables}")

# List tables in specific database/schema
tables = ingestor.list_tables(database="OTHER_DB", schema="OTHER_SCHEMA")
```

### Export as Documents

Convert Snowflake data to Semantica document format:

```python
# Ingest data
data = ingestor.ingest_table("ARTICLES")

# Convert to documents
documents = ingestor.export_as_documents(
    data,
    id_field="ARTICLE_ID",
    text_fields=["TITLE", "CONTENT", "SUMMARY"]
)

# Now use in Semantica pipeline
from semantica.pipeline import Pipeline

pipeline = Pipeline()
for doc in documents:
    pipeline.process_document(doc)
```

### Context Manager

Use as a context manager for automatic connection cleanup:

```python
with SnowflakeIngestor(
    account="myaccount",
    user="myuser",
    password="mypassword"
) as ingestor:
    data = ingestor.ingest_table("CUSTOMERS")
    # Connection automatically closed on exit
```

### Multiple Schemas/Databases

```python
# Ingest from different databases and schemas
data1 = ingestor.ingest_table(
    "CUSTOMERS",
    database="SALES_DB",
    schema="PROD"
)

data2 = ingestor.ingest_table(
    "ORDERS",
    database="SALES_DB",
    schema="STAGING"
)
```

## Error Handling

```python
from semantica.utils.exceptions import ProcessingError, ValidationError

try:
    ingestor = SnowflakeIngestor(
        account="myaccount",
        user="myuser",
        password="wrong_password"
    )
    data = ingestor.ingest_table("CUSTOMERS")
    
except ValidationError as e:
    print(f"Configuration error: {e}")
    
except ProcessingError as e:
    print(f"Processing failed: {e}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Test Connection

```python
# Test connection before using
connector = SnowflakeConnector(
    account="myaccount",
    user="myuser",
    password="mypassword"
)

if connector.test_connection():
    print("Connection successful!")
else:
    print("Connection failed")
```

## Best Practices

### 1. Use Environment Variables for Credentials

**Don't:**
```python
# Hard-coded credentials (bad!)
ingestor = SnowflakeIngestor(
    account="myaccount",
    user="myuser",
    password="mypassword123"
)
```

**Do:**
```python
# Use environment variables
import os
from dotenv import load_dotenv

load_dotenv()

ingestor = SnowflakeIngestor()  # Reads from environment
```

### 2. Use Key-Pair Authentication for Production

Key-pair authentication is more secure than passwords:

```python
ingestor = SnowflakeIngestor(
    account="myaccount",
    user="myuser",
    private_key_path=os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH"),
    warehouse="COMPUTE_WH"
)
```

### 3. Paginate Large Result Sets

```python
# Don't fetch millions of rows at once
PAGE_SIZE = 10000

for page in range(total_pages):
    data = ingestor.ingest_table(
        "LARGE_TABLE",
        limit=PAGE_SIZE,
        offset=page * PAGE_SIZE
    )
    process_batch(data)
```

### 4. Use Query Filters

Push filtering to Snowflake instead of filtering in Python:

**Don't:**
```python
# Fetching all rows and filtering in Python (inefficient)
all_data = ingestor.ingest_table("CUSTOMERS")
filtered = [row for row in all_data.data if row["COUNTRY"] == "USA"]
```

**Do:**
```python
# Filter in Snowflake (efficient)
data = ingestor.ingest_table("CUSTOMERS", where="COUNTRY = 'USA'")
```

### 5. Set Appropriate Warehouse Size

```python
# Use appropriate warehouse for workload
ingestor = SnowflakeIngestor(
    account="myaccount",
    user="myuser",
    password="mypassword",
    warehouse="LARGE_WH"  # For heavy workloads
)
```

### 6. Close Connections

```python
# Always close connections when done
try:
    ingestor = SnowflakeIngestor(...)
    data = ingestor.ingest_table("CUSTOMERS")
finally:
    ingestor.close()

# Or use context manager
with SnowflakeIngestor(...) as ingestor:
    data = ingestor.ingest_table("CUSTOMERS")
```

## Examples

### Example 1: ETL Pipeline

```python
from semantica.ingest import SnowflakeIngestor
from semantica.pipeline import Pipeline

# Initialize ingestor
ingestor = SnowflakeIngestor()

# Extract data
sales_data = ingestor.ingest_query("""
    SELECT 
        s.ORDER_ID,
        s.CUSTOMER_ID,
        c.CUSTOMER_NAME,
        s.PRODUCT_ID,
        p.PRODUCT_NAME,
        s.AMOUNT,
        s.ORDER_DATE
    FROM SALES s
    JOIN CUSTOMERS c ON s.CUSTOMER_ID = c.ID
    JOIN PRODUCTS p ON s.PRODUCT_ID = p.ID
    WHERE s.ORDER_DATE >= CURRENT_DATE - 30
""")

# Transform to documents
documents = ingestor.export_as_documents(
    sales_data,
    id_field="ORDER_ID",
    text_fields=["CUSTOMER_NAME", "PRODUCT_NAME"]
)

# Load into Semantica
pipeline = Pipeline()
pipeline.process_documents(documents)
```

### Example 2: Multi-Table Ingestion

```python
from semantica.ingest import SnowflakeIngestor

ingestor = SnowflakeIngestor()

# Get all tables
tables = ingestor.list_tables()

# Ingest each table
for table_name in tables:
    print(f"Ingesting {table_name}...")
    
    # Get schema first
    schema = ingestor.get_table_schema(table_name)
    
    # Ingest with limit
    data = ingestor.ingest_table(table_name, limit=1000)
    
    # Process data
    process_table(table_name, data, schema)
```

### Example 3: Incremental Loading

```python
import datetime
from semantica.ingest import SnowflakeIngestor

ingestor = SnowflakeIngestor()

# Get last load timestamp
last_load = get_last_load_timestamp()  # Your function

# Ingest only new/updated records
data = ingestor.ingest_table(
    "CUSTOMERS",
    where=f"UPDATED_AT > '{last_load}'",
    order_by="UPDATED_AT ASC"
)

print(f"Loaded {data.row_count} new/updated records")

# Update last load timestamp
set_last_load_timestamp(datetime.datetime.now())
```

## Troubleshooting

### Connection Issues

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test connection
connector = SnowflakeConnector(
    account="myaccount",
    user="myuser",
    password="mypassword"
)

if not connector.test_connection():
    print("Connection failed - check credentials and network")
```

### Large Query Timeouts

```python
# Use batching for large queries
data = ingestor.ingest_query(
    "SELECT * FROM VERY_LARGE_TABLE",
    batch_size=10000  # Fetch in batches
)
```

### Memory Issues with Large Results

```python
# Process in chunks instead of loading all at once
CHUNK_SIZE = 10000
offset = 0

while True:
    chunk = ingestor.ingest_table(
        "LARGE_TABLE",
        limit=CHUNK_SIZE,
        offset=offset
    )
    
    if chunk.row_count == 0:
        break
        
    process_chunk(chunk)
    offset += CHUNK_SIZE
```
