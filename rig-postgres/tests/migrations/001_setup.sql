-- ensure extension is installed
CREATE EXTENSION IF NOT EXISTS vector;

-- create table with embeddings using 1536 dimensions (text-embedding-3-small)
CREATE TABLE documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document jsonb NOT NULL,
  embeddings vector(1536)
);

-- create index on embeddings
CREATE INDEX IF NOT EXISTS document_embeddings_idx ON documents 
USING hnsw(embeddings vector_cosine_ops);
