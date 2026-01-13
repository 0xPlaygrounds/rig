use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{SearchFilter, VectorSearchRequest},
    },
};
use scylla::{
    client::{Compression, session::Session, session_builder::SessionBuilder},
    statement::prepared::PreparedStatement,
    value::CqlValue,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, RwLock},
};
use uuid::Uuid;

/// Represents a vector store implementation using ScyllaDB as the backend.
///
/// ScyllaDB is a high-performance NoSQL database that's compatible with Apache Cassandra
/// and provides excellent performance for vector storage and similarity search operations.
pub struct ScyllaDbVectorStore<M: EmbeddingModel> {
    /// Model used to generate embeddings for the vector store
    model: M,
    /// Session instance for ScyllaDB communication
    pub session: Arc<Session>,
    /// Keyspace and table name for vector storage
    keyspace: String,
    table: String,
    /// The number of dimensions for vectors
    dimensions: usize,
    /// Prepared statements for optimized queries
    insert_stmt: PreparedStatement,
    search_stmt: PreparedStatement,
    get_by_id_stmt: PreparedStatement,
    /// Cache for statements which cannot be prepared AOT
    cache: Arc<RwLock<HashMap<u64, PreparedStatement>>>,
}

// NOTE: Cannot be used as a dynamic store due to CqlValue not impl'ing Serialize or Deserialize
/// TODO: Write tests for this !
#[derive(Clone, Debug)]
pub struct ScyllaSearchFilter {
    condition: String,
    params: Vec<CqlValue>,
}

impl std::hash::Hash for ScyllaSearchFilter {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.condition.hash(state)
    }
}

impl SearchFilter for ScyllaSearchFilter {
    type Value = CqlValue;

    fn eq(key: String, value: Self::Value) -> Self {
        Self {
            condition: format!("{key} = ?"),
            params: vec![value],
        }
    }

    fn gt(key: String, value: Self::Value) -> Self {
        Self {
            condition: format!("{key} > ?"),
            params: vec![value],
        }
    }

    fn lt(key: String, value: Self::Value) -> Self {
        Self {
            condition: format!("{key} < ?"),
            params: vec![value],
        }
    }

    fn and(self, rhs: Self) -> Self {
        Self {
            condition: format!("({}) AND ({})", self.condition, rhs.condition),
            params: self.params.into_iter().chain(rhs.params).collect(),
        }
    }

    fn or(self, rhs: Self) -> Self {
        Self {
            condition: format!("({}) OR ({})", self.condition, rhs.condition),
            params: self.params.into_iter().chain(rhs.params).collect(),
        }
    }
}

impl ScyllaSearchFilter {
    fn params(&self) -> &[CqlValue] {
        self.params.as_slice()
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self {
            condition: format!("NOT ({})", self.condition),
            ..self
        }
    }

    pub fn gte(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self {
            condition: format!("{key} >= ?"),
            params: vec![value],
        }
    }

    pub fn lte(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self {
            condition: format!("{key} <= ?"),
            params: vec![value],
        }
    }

    pub fn ne(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self {
            condition: format!("{key} != ?"),
            params: vec![value],
        }
    }

    pub fn member(key: String, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        let placeholders = vec!["?"; values.len()].join(", ");

        Self {
            condition: format!("{key} IN ({placeholders})"),
            params: values,
        }
    }
}

impl<M> ScyllaDbVectorStore<M>
where
    M: EmbeddingModel,
{
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
            cache: Default::default(),
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

    async fn get_filter_statement_or_default(
        &self,
        req: &VectorSearchRequest<ScyllaSearchFilter>,
    ) -> Result<PreparedStatement, VectorStoreError> {
        if let Some(filter) = req.filter() {
            let mut hasher = DefaultHasher::new();
            filter.hash(&mut hasher);
            let filter_hash = hasher.finish();

            let statement = if let Some(cached) = self
                .cache
                .read()
                .ok()
                .and_then(|cache| cache.get(&filter_hash).cloned())
            {
                cached
            } else {
                let query = format!(
                    "SELECT id, vector, metadata, created_at FROM {}.{} WHERE {} ALLOW FILTERING",
                    self.keyspace, self.table, filter.condition
                );

                let prepared = self
                    .session
                    .prepare(query)
                    .await
                    .map_err(|e| VectorStoreError::DatastoreError(e.into()))?;

                let mut cache = self.cache.write().map_err(|e| {
                    VectorStoreError::DatastoreError(
                        format!("Error writing statement cache: {e}").into(),
                    )
                })?;
                cache.insert(filter_hash, prepared.clone());
                prepared
            };

            Ok(statement)
        } else {
            Ok(self.search_stmt.clone())
        }
    }
}

impl<Model> InsertDocuments for ScyllaDbVectorStore<Model>
where
    Model: EmbeddingModel + Send + Sync,
{
    async fn insert_documents<Doc: Serialize + Embed + Send>(
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
}

impl<M> VectorStoreIndex for ScyllaDbVectorStore<M>
where
    M: EmbeddingModel + std::marker::Sync + Send,
{
    type Filter = ScyllaSearchFilter;

    /// Search for the top `n` nearest neighbors to the given query.
    /// Returns a vector of tuples containing the score, ID, and payload of the nearest neighbors.
    ///
    /// Note: This implementation performs a brute-force search since ScyllaDB's native vector
    /// search is still in development. Once available, this will be optimized to use native
    /// vector search capabilities with ANN (Approximate Nearest Neighbor) algorithms.
    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<ScyllaSearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let query_vector = self.generate_query_vector(req.query()).await?;

        let statement = self.get_filter_statement_or_default(&req).await?;
        let params = req
            .filter()
            .as_ref()
            .map(ScyllaSearchFilter::params)
            .unwrap_or([].as_slice());

        // Fetch all vectors (this will be optimized once ScyllaDB vector search is available)
        let results = self
            .session
            .execute_unpaged(&statement, params)
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

            if req.threshold().is_some_and(|threshold| score < threshold) {
                continue;
            }

            let payload: T = serde_json::from_str(&metadata)?;

            candidates.push((score, id.to_string(), payload));
        }

        // Sort by similarity score (descending) and take top n
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        candidates.truncate(req.samples() as usize);

        Ok(candidates)
    }

    /// Search for the top `n` nearest neighbors to the given query.
    /// Returns a vector of tuples containing the score and ID of the nearest neighbors.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<ScyllaSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let query_vector = self.generate_query_vector(req.query()).await?;

        let statement = self.get_filter_statement_or_default(&req).await?;
        let params = req
            .filter()
            .as_ref()
            .map(ScyllaSearchFilter::params)
            .unwrap_or_default();

        let results = self
            .session
            .execute_unpaged(&statement, params)
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

            if req.threshold().is_some_and(|threshold| score < threshold) {
                continue;
            }

            candidates.push((score, id.to_string()));
        }

        // Sort by similarity score (descending) and take top n
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        candidates.truncate(req.samples() as usize);

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
