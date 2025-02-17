-- define table & fields
DEFINE TABLE documents SCHEMAFULL;
DEFINE field document on table documents type object;
DEFINE field embedding on table documents type array<float>;
DEFINE field embedded_text on table documents type string;

-- define index on embedding field
DEFINE INDEX IF NOT EXISTS words_embedding_vector_index ON documents
     FIELDS embedding
     MTREE DIMENSION 1536
     DIST COSINE;
