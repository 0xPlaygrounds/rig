use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{VectorStoreError, VectorStoreIndex},
};
use scylla::{
    client::{Compression, session::Session, session_builder::SessionBuilder},
    statement::prepared::PreparedStatement,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Represents a vector store implementation using ScyllaDB as the backend.
///
/// ScyllaDB is a high-performance NoSQL database that's compatible with Apache Cassandra
/// and provides excellent performance for vector storage and similarity search operations.
pub struct ScyllaDbVectorStore<M: EmbeddingModel> {
    /// Model used to generate embeddings for the vector store
    model: M,
    /// Session instance for ScyllaDB communication
    session: Arc<Session>,
    /// Keyspace and table name for vector storage
    keyspace: String,
    table: String,
    /// The number of dimensions for vectors
    dimensions: usize,
    /// Prepared statements for optimized queries
    insert_stmt: PreparedStatement,
    search_stmt: PreparedStatement,
    get_by_id_stmt: PreparedStatement,
}

#[derive(Debug, Serialize, Deserialize)]
struct VectorRecord {
    id: Uuid,
    vector: Vec<f32>,
    metadata: String, // JSON serialized metadata
    created_at: i64,  // Unix timestamp
}

impl<M: EmbeddingModel> ScyllaDbVectorStore<M> {
    /// Creates a new instance of `ScyllaDbVectorStore`.
    ///
    /// # Arguments
    /// * `model` - Embedding model instance
    /// * `session` - ScyllaDB session
    /// * `keyspace` - Keyspace name (will be created if it doesn't exist)
    /// * `table` - Table name for storing vectors
    /// * `dimensions` - Number of dimensions for the vectors
    pub async fn new(
        model: M,
        session: Session,
        keyspace: &str,
        table: &str,
        dimensions: usize,
    ) -> Result<Self, VectorStoreError> {
        let session = Arc::new(session);

        // Create keyspace if it doesn't exist
        let create_keyspace_cql = format!(
            "CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH REPLICATION = {{
                'class': 'SimpleStrategy',
                'replication_factor': 1
            }}"
        );
        session
            .query_unpaged(create_keyspace_cql, &[])
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        // Create table for storing vectors
        // Note: Once ScyllaDB vector search is fully available, we'll use VECTOR type
        // For now, we use a list of floats and implement similarity search in application code
        let create_table_cql = format!(
            "CREATE TABLE IF NOT EXISTS {keyspace}.{table} (
                id UUID PRIMARY KEY,
                vector LIST<FLOAT>,
                metadata TEXT,
                created_at BIGINT
            )"
        );
        session
            .query_unpaged(create_table_cql, &[])
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        // Prepare statements for better performance
        let insert_stmt = session
            .prepare(format!(
                "INSERT INTO {keyspace}.{table} (id, vector, metadata, created_at) VALUES (?, ?, ?, ?)"
            ))
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let search_stmt = session
            .prepare(format!(
                "SELECT id, vector, metadata, created_at FROM {keyspace}.{table}"
            ))
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let get_by_id_stmt = session
            .prepare(format!(
                "SELECT id, vector, metadata, created_at FROM {keyspace}.{table} WHERE id = ?"
            ))
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        Ok(Self {
            model,
            session,
            keyspace: keyspace.to_string(),
            table: table.to_string(),
            dimensions,
            insert_stmt,
            search_stmt,
            get_by_id_stmt,
        })
    }

    /// Get the session reference
    pub fn session(&self) -> &Arc<Session> {
        &self.session
    }

    /// Get the keyspace name
    pub fn keyspace(&self) -> &str {
        &self.keyspace
    }

    /// Get the table name
    pub fn table(&self) -> &str {
        &self.table
    }

    /// Get a document by its ID
    pub async fn get_by_id<T: for<'a> Deserialize<'a> + Send>(
        &self,
        id: &str,
    ) -> Result<Option<T>, VectorStoreError> {
        let uuid =
            Uuid::parse_str(id).map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let result = self
            .session
            .execute_unpaged(&self.get_by_id_stmt, (uuid,))
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows_result = result
            .into_rows_result()
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        if let Some(first_row) = rows_result
            .rows::<(Uuid, Vec<f32>, String, i64)>()
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?
            .next()
        {
            let (_, _, metadata, _) =
                first_row.map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

            let payload: T = serde_json::from_str(&metadata)?;
            return Ok(Some(payload));
        }

        Ok(None)
    }

    /// Insert documents with their embeddings into the vector store
    pub async fn insert_documents<Doc: Serialize + Embed + Send>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        for (document, embeddings) in documents {
            let metadata = serde_json::to_string(&document)?;
            let now = chrono::Utc::now().timestamp();

            for embedding in embeddings.into_iter() {
                let vector: Vec<f32> = embedding.vec.into_iter().map(|x| x as f32).collect();

                if vector.len() != self.dimensions {
                    return Err(VectorStoreError::DatastoreError(
                        format!(
                            "Vector dimension mismatch: expected {}, got {}",
                            self.dimensions,
                            vector.len()
                        )
                        .into(),
                    ));
                }

                let id = Uuid::new_v4();

                self.session
                    .execute_unpaged(&self.insert_stmt, (id, vector, &metadata, now))
                    .await
                    .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
            }
        }

        Ok(())
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
        let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    /// Generate query vector from text
    async fn generate_query_vector(&self, query: &str) -> Result<Vec<f32>, VectorStoreError> {
        let embedding = self.model.embed_text(query).await?;
        Ok(embedding.vec.iter().map(|&x| x as f32).collect())
    }
}

impl<M: EmbeddingModel + std::marker::Sync + Send> VectorStoreIndex for ScyllaDbVectorStore<M> {
    /// Search for the top `n` nearest neighbors to the given query.
    /// Returns a vector of tuples containing the score, ID, and payload of the nearest neighbors.
    ///
    /// Note: This implementation performs a brute-force search since ScyllaDB's native vector
    /// search is still in development. Once available, this will be optimized to use native
    /// vector search capabilities with ANN (Approximate Nearest Neighbor) algorithms.
    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let query_vector = self.generate_query_vector(query).await?;

        // Fetch all vectors (this will be optimized once ScyllaDB vector search is available)
        let results = self
            .session
            .execute_unpaged(&self.search_stmt, &[])
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows_result = results
            .into_rows_result()
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let mut candidates = Vec::new();

        for row_result in rows_result
            .rows::<(Uuid, Vec<f32>, String, i64)>()
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?
        {
            let (id, vector, metadata, _) =
                row_result.map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

            let similarity = Self::cosine_similarity(&query_vector, &vector);
            let score = similarity as f64;

            let payload: T = serde_json::from_str(&metadata)?;

            candidates.push((score, id.to_string(), payload));
        }

        // Sort by similarity score (descending) and take top n
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        candidates.truncate(n);

        Ok(candidates)
    }

    /// Search for the top `n` nearest neighbors to the given query.
    /// Returns a vector of tuples containing the score and ID of the nearest neighbors.
    async fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let query_vector = self.generate_query_vector(query).await?;

        let results = self
            .session
            .execute_unpaged(&self.search_stmt, &[])
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows_result = results
            .into_rows_result()
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let mut candidates = Vec::new();

        for row_result in rows_result
            .rows::<(Uuid, Vec<f32>, String, i64)>()
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?
        {
            let (id, vector, _, _) =
                row_result.map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

            let similarity = Self::cosine_similarity(&query_vector, &vector);
            let score = similarity as f64;

            candidates.push((score, id.to_string()));
        }

        // Sort by similarity score (descending) and take top n
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        candidates.truncate(n);

        Ok(candidates)
    }
}

/// Convenience function to create a ScyllaDB session
pub async fn create_session(uri: &str) -> Result<Session, VectorStoreError> {
    SessionBuilder::new()
        .known_node(uri)
        .compression(Some(Compression::Lz4))
        .build()
        .await
        .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
}
