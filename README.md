# Building a High-Performance RAG Solution with Pgvectorscale, Python, and Ollama

This tutorial will guide you through setting up and using `pgvectorscale` with Docker and Python, leveraging the open source `mxbai-embed-large` model for embeddings. You'll learn to build a cutting-edge RAG (Retrieval-Augmented Generation) solution, combining advanced retrieval techniques (including hybrid search) with intelligent answer generation based on the retrieved context. Perfect for AI engineers looking to enhance their projects with state-of-the-art vector search and generation capabilities with the power of PostgreSQL.

## YouTube Tutorial
You can watch the full tutorial here on [YouTube](https://youtu.be/hAdEuDBN57g).

## Pgvectorscale Documentation

For more information about using PostgreSQL as a vector database in AI applications with Timescale, check out these resources:

- [GitHub Repository: pgvectorscale](https://github.com/timescale/pgvectorscale)
- [Blog Post: PostgreSQL and Pgvector: Now Faster Than Pinecone, 75% Cheaper, and 100% Open Source](https://www.timescale.com/blog/pgvector-is-now-as-fast-as-pinecone-at-75-less-cost/)
- [Blog Post: RAG Is More Than Just Vector Search](https://www.timescale.com/blog/rag-is-more-than-just-vector-search/)
- [Blog Post: A Python Library for Using PostgreSQL as a Vector Database in AI Applications](https://www.timescale.com/blog/a-python-library-for-using-postgresql-as-a-vector-database-in-ai-applications/)

## Why PostgreSQL?

Using PostgreSQL with pgvectorscale as your vector database offers several key advantages over dedicated vector databases:

- PostgreSQL is a robust, open-source database with a rich ecosystem of tools, drivers, and connectors. This ensures transparency, community support, and continuous improvements.

- By using PostgreSQL, you can manage both your relational and vector data within a single database. This reduces operational complexity, as there's no need to maintain and synchronize multiple databases.

- Pgvectorscale enhances pgvector with faster search capabilities, higher recall, and efficient time-based filtering. It leverages advanced indexing techniques, such as the DiskANN-inspired index, to significantly speed up Approximate Nearest Neighbor (ANN) searches.

Pgvectorscale Vector builds on top of [pgvector](https://github.com/pgvector/pgvector), offering improved performance and additional features, making PostgreSQL a powerful and versatile choice for AI applications.

## Prerequisites

- Docker
- Python 3.7+
- Ollama
- PostgreSQL GUI client

## Steps

1. Set up Docker environment
2. Connect to the database using a PostgreSQL GUI client (I use TablePlus)
3. Create a Python script to insert document chunks as vectors using mxbai-embed-large embedding model (using Ollama).
4. Create a Python function to perform similarity search

## Detailed Instructions

### 1. Set up Docker environment

Create a `docker-compose.yml` file with the following content:

```yaml
services:
  timescaledb:
    image: timescale/timescaledb-ha:pg16
    container_name: timescaledb
    environment:
      - POSTGRES_DB=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - timescaledb_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  timescaledb_data:
```

Run the Docker container:

```bash
docker compose up -d
```

### 2. Connect to the database using a PostgreSQL GUI client

- Open client
- Create a new connection with the following details:
  - Host: localhost
  - Port: 5432
  - User: postgres
  - Password: password
  - Database: postgres

### 3. Set up Ollama

Install ollama locally or point to a managed ollama instance by setting OLLAMA_BASE_URL in the [app/.env](app/.env).

1. Optional if ollama is different than http://127.0.0.1:11434/:

```bash
cp example.env .env
```

2. For linux based on the env config docs described [here](https://github.com/ollama/ollama/blob/main/docs/faq.md#setting-environment-variables-on-linux
) edit /etc/systemd/system/ollama.service.d/override.conf or make sure the env vars are set, OLLAMA_HOST=0.0.0.0 and OLLAMA_ORIGINS=*:

```
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"

systemctl daemon-reload
systemctl restart ollama
```

3. Make sure ollama is running on http://127.0.0.1:11434/ (or by loading OLLAMA_BASE_URL), should return:

```
Ollama is running
```

4. Finally load the following models:

```bash
ollama pull llama3.1
ollama pull mxbai-embed-large
```

### 4. Create a Python script to insert document chunks as vectors

#### 4.1 Installing and running Python

Download and install python from https://www.python.org/downloads/.

Once Python is installed from the root directory of this project, run the following commands:

```
python3 -m venv ./venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

#### 4.2 Running insert_vectors.py

See `insert_vectors.py` for the implementation. This script uses the open source `mxbai-embed-large` model to generate embeddings.

### 5. Create a Python function to perform similarity search

See `similarity_search.py` for the implementation. This script also uses the open source `mxbai-embed-large` model for query embedding.

## Usage

1. Create a copy of `example.env` and rename it to `.env`
2. Open `.env` and fill in your TIMESCALE_SERVICE_URL if your postgres database is different than the default configs.
3. Run the Docker container
4. Install the required Python packages using `pip install -r requirements.txt`
5. Execute `insert_vectors.py` to populate the database
6. Play with `similarity_search.py` to perform similarity searches

## Using ANN search indexes to speed up queries

Timescale Vector offers indexing options to accelerate similarity queries, particularly beneficial for large vector datasets (10k+ vectors):

1. Supported indexes:
   - timescale_vector_index (default): A DiskANN-inspired graph index
   - pgvector's HNSW: Hierarchical Navigable Small World graph index
   - pgvector's IVFFLAT: Inverted file index

2. The DiskANN-inspired index is Timescale's latest offering, providing improved performance. Refer to the [Timescale Vector explainer blog](https://www.timescale.com/blog/pgvector-is-now-as-fast-as-pinecone-at-75-less-cost/) for detailed information and benchmarks.

For optimal query performance, creating an index on the embedding column is recommended, especially for large vector datasets.

## Cosine Similarity in Vector Search

### What is Cosine Similarity?

Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space. It's a measure of orientation rather than magnitude.

- Range: -1 to 1 (for normalized vectors, which is typical in text embeddings)
- 1: Vectors point in the same direction (most similar)
- 0: Vectors are orthogonal (unrelated)
- -1: Vectors point in opposite directions (most dissimilar)

### Cosine Distance

In pgvector, the `<=>` operator computes cosine distance, which is 1 - cosine similarity.

- Range: 0 to 2
- 0: Identical vectors (most similar)
- 1: Orthogonal vectors
- 2: Opposite vectors (most dissimilar)

### Interpreting Results

When you get results from similarity_search:

- Lower distance values indicate higher similarity.
- A distance of 0 would mean exact match (rarely happens with embeddings).
- Distances closer to 0 indicate high similarity.
- Distances around 1 suggest little to no similarity.
- Distances approaching 2 indicate opposite meanings (rare in practice).


## Troubleshooting

### Test Ollama

Run the following curl command to see if ollama/llama3.1 is returning actual content:

```bash
curl -X POST "http://127.0.0.1:11434/api/chat" \
-H "Content-Type: application/json" \
-d '{
  "model": "llama3.1",
  "stream": false,
  "messages": [
    {
      "role": "system",
      "content": "# Role and Purpose\nYou are an AI assistant for an e-commerce FAQ system. Your task is to synthesize a coherent and helpful answer \nbased on the given question and relevant context retrieved from a knowledge database.\n\n# Guidelines:\n1. Provide a clear and concise answer to the question.\n2. Use only the information from the relevant context to support your answer.\n3. The context is retrieved based on cosine similarity, so some information might be missing or irrelevant.\n4. Be transparent when there is insufficient information to fully answer the question.\n5. Do not make up or infer information not present in the provided context.\n6. If you cannot answer the question based on the given context, clearly state that.\n7. Maintain a helpful and professional tone appropriate for customer service.\n8. Adhere strictly to company guidelines and policies by using only the provided knowledge base.\n\nReview the question from the user:\n"
    },
    {
      "role": "user",
      "content": "# User question:\nWhat are your shipping options?"
    },
    {
      "role": "assistant",
      "content": "# Retrieved information:\n[\n  {\n    \"content\": \"Question: What are your shipping options?\\nAnswer: We offer standard (3-5 business days) and express (1-2 business days) shipping options.\",\n    \"category\": \"Shipping\"\n  },\n  {\n    \"content\": \"Question: Do you offer international shipping?\\nAnswer: Yes, we ship to most countries worldwide. Shipping costs and delivery times vary by location.\",\n    \"category\": \"Shipping\"\n  },\n  {\n    \"content\": \"Question: How can I track my order?\\nAnswer: You can track your order by logging into your account and viewing the order status or using the tracking number sent to your email.\",\n    \"category\": \"Order Management\"\n  }\n]"
    },
    {
      "role": "assistant",
      "content": ""
    }
  ]
}'
```

Should return something like:

```json
{
  "model": "llama3.1",
  "created_at": "2024-12-22T23:12:26.040267689Z",
  "message": {
    "role": "assistant",
    "content": "# Answer:\nWe offer two shipping options:\n\n*   Standard shipping, which takes approximately 3-5 business days\n*   Express shipping, which takes approximately 1-2 business days\n\nIf you have any questions about shipping or would like to estimate delivery times for a specific location, please don't hesitate to contact us. We're here to help!"
  },
  "done_reason": "stop",
  "done": true,
  "total_duration": 63889551074,
  "load_duration": 6111508585,
  "prompt_eval_count": 359,
  "prompt_eval_duration": 42337000000,
  "eval_count": 73,
  "eval_duration": 14540000000
}
```