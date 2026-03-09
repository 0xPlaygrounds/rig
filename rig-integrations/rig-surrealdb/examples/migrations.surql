-- define table & fields
DEFINE TABLE documents SCHEMAFULL;

DEFINE FIELD document ON documents
	TYPE object;

DEFINE field embedding ON documents
	TYPE array<float>;

DEFINE field embedded_text ON documents
	TYPE string;

-- define index on embedding field
DEFINE INDEX IF NOT EXISTS words_embedding_vector_index ON documents
     FIELDS embedding
     HNSW DIMENSION 1536
     DIST COSINE;
