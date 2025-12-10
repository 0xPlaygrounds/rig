-- ensure extension is installed
CREATE EXTENSION IF NOT EXISTS vector;

-- create table with embeddings using 1536 dimensions (text-embedding-3-small)
CREATE TABLE documents (
  id uuid DEFAULT gen_random_uuid(), -- we can have repeated entries
  document jsonb NOT NULL,
  embedded_text text NOT NULL,
  embedding vector(1536)
);

-- create index on embeddings
CREATE INDEX IF NOT EXISTS document_embeddings_idx ON documents 
USING hnsw(embedding vector_cosine_ops);
