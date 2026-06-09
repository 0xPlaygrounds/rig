//! Redis vector store integration for Rig.
//!
//! Provides a [`RedisVectorStore`] that implements Rig's [`VectorStoreIndex`] and
//! [`InsertDocuments`] traits using RediSearch's vector similarity search (`FT.SEARCH`).
//!
//! # Prerequisites
//!
//! The RediSearch index must be created before using this store. The expected schema is:
//! - A HASH-based index with the specified prefix
//! - A `document` field of type TEXT (stores serialized JSON)
//! - An `embedded_text` field of type TEXT (stores the source text)
//! - A vector field (configurable name) of type VECTOR with FLOAT32 elements
//! - Optionally, additional fields for metadata filtering (TAG, NUMERIC, etc.)
//!
//! # Distance Metric
//!
//! This implementation assumes the RediSearch index uses **COSINE** distance.
//! Redis returns cosine distance (0 = identical, 2 = opposite), which is converted
//! to cosine similarity (1 = identical, -1 = opposite) via `1.0 - distance`.
//! Using a different distance metric (L2, IP) will produce incorrect similarity scores.
//!
//! # Metadata Filtering
//!
//! To enable filtering on document fields during search, configure metadata fields
//! via [`RedisVectorStore::with_metadata_fields`]. These fields are extracted from
//! the serialized document JSON during insertion and written as separate hash fields,
//! making them available for RediSearch filter queries.
//!
//! Your RediSearch index schema must declare these fields with appropriate types
//! (TAG, NUMERIC, TEXT) for filters to work.
//!
//! # Example
//! ```ignore
//! use rig_redis::RedisVectorStore;
//!
//! let store = RedisVectorStore::new(
//!     embedding_model,
//!     redis_client,
//!     "my_index".into(),
//!     "embedding".into(),
//! )
//! .await?
//! .with_key_prefix("doc:".to_string())
//! .with_metadata_fields(vec!["category".to_string(), "price".to_string()]);
//! ```

pub mod filter;

pub use filter::Filter;
use redis::aio::ConnectionManager;
use rig_core::{
    Embed, OneOrMany,
    embeddings::embedding::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, TopNResults, VectorStoreError, VectorStoreIndex, VectorStoreIndexDyn,
        request::{Filter as CoreFilter, VectorSearchRequest},
    },
    wasm_compat::WasmBoxedFuture,
};
use serde::{Deserialize, Serialize};

/// Redis vector store implementation using RediSearch vector similarity search.
///
/// Uses Redis's `FT.SEARCH` command with KNN vector queries for similarity search.
/// Internally holds a [`ConnectionManager`] for automatic reconnection on transient failures.
///
/// # Key Prefix
///
/// If your RediSearch index uses a `PREFIX` configuration (e.g., `PREFIX 1 doc:`),
/// you **must** call [`RedisVectorStore::with_key_prefix`] with the matching prefix
/// so that inserted documents are discoverable by the index.
///
/// # Metadata Fields
///
/// Configure metadata fields via [`RedisVectorStore::with_metadata_fields`] to enable
/// filtering. During insertion, these fields are extracted from the serialized document
/// and stored as separate hash fields that RediSearch can index and filter on.
pub struct RedisVectorStore<M>
where
    M: EmbeddingModel,
{
    model: M,
    connection_manager: ConnectionManager,
    index_name: String,
    vector_field: String,
    key_prefix: Option<String>,
    metadata_fields: Vec<String>,
}

impl<M> RedisVectorStore<M>
where
    M: EmbeddingModel,
{
    /// Creates a new Redis vector store instance.
    ///
    /// Establishes a [`ConnectionManager`] from the provided client for automatic
    /// reconnection on transient network failures.
    ///
    /// # Arguments
    /// * `model` - Embedding model for query vectorization
    /// * `client` - Redis client instance
    /// * `index_name` - Name of the RediSearch index to query
    /// * `vector_field` - Name of the vector field in the index
    ///
    /// # Errors
    /// Returns an error if the initial connection to Redis cannot be established.
    pub async fn new(
        model: M,
        client: redis::Client,
        index_name: String,
        vector_field: String,
    ) -> Result<Self, VectorStoreError> {
        let connection_manager = ConnectionManager::new(client)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        Ok(Self {
            model,
            connection_manager,
            index_name,
            vector_field,
            key_prefix: None,
            metadata_fields: Vec::new(),
        })
    }

    /// Sets a key prefix for document keys.
    ///
    /// Documents stored via [`InsertDocuments`] will be keyed as `{prefix}{uuid}`.
    /// This prefix **must** match the index's `PREFIX` configuration for documents
    /// to be indexed and discoverable by `FT.SEARCH`.
    pub fn with_key_prefix(mut self, prefix: String) -> Self {
        self.key_prefix = Some(prefix);
        self
    }

    /// Configures metadata fields to extract from documents during insertion.
    ///
    /// When documents are inserted, the specified fields are extracted from the
    /// serialized JSON representation and written as separate hash fields. This
    /// makes them available for RediSearch filter queries (TAG, NUMERIC, TEXT).
    ///
    /// The field names must match top-level keys in the serialized document JSON
    /// **and** must be declared in the RediSearch index schema.
    ///
    /// Fields that are missing from a document or have null/complex values are
    /// silently skipped with a warning log.
    ///
    /// Reserved field names (`document`, `embedded_text`, and the configured vector
    /// field) are filtered out with a warning to prevent data corruption.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Given a document struct:
    /// #[derive(Serialize, Embed)]
    /// struct Product {
    ///     name: String,
    ///     category: String,
    ///     price: f64,
    ///     #[embed]
    ///     description: String,
    /// }
    ///
    /// // And an index created with:
    /// // FT.CREATE products ON HASH PREFIX 1 prod:
    /// //   SCHEMA document TEXT embedded_text TEXT embedding VECTOR FLAT 6 ...
    /// //   category TAG price NUMERIC
    ///
    /// let store = RedisVectorStore::new(model, client, "products".into(), "embedding".into())
    ///     .await?
    ///     .with_key_prefix("prod:".to_string())
    ///     .with_metadata_fields(vec!["category".to_string(), "price".to_string()]);
    /// ```
    pub fn with_metadata_fields(mut self, fields: Vec<String>) -> Self {
        let reserved = ["document", "embedded_text", self.vector_field.as_str()];
        self.metadata_fields = fields
            .into_iter()
            .filter(|f| {
                if reserved.contains(&f.as_str()) {
                    tracing::warn!(
                        target: "rig",
                        field = %f,
                        "Metadata field name conflicts with reserved hash field, skipping"
                    );
                    false
                } else {
                    true
                }
            })
            .collect();
        self
    }

    /// Converts f64 embedding vector to f32 little-endian bytes for Redis VECTOR fields.
    fn embedding_to_bytes(embedding: &[f64]) -> Vec<u8> {
        embedding
            .iter()
            .flat_map(|&x| (x as f32).to_le_bytes())
            .collect()
    }

    /// Extracts a UTF-8 string from a Redis bulk/simple string value.
    fn extract_string(value: &redis::Value) -> Option<String> {
        match value {
            redis::Value::BulkString(bytes) => Some(String::from_utf8_lossy(bytes).to_string()),
            redis::Value::SimpleString(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Extracts a cosine distance score and converts to similarity.
    ///
    /// RediSearch COSINE metric returns distance in [0, 2] where 0 = identical.
    /// We convert to similarity = 1.0 - distance so higher = more similar.
    fn extract_score(value: &redis::Value) -> Result<f64, VectorStoreError> {
        let distance = match value {
            redis::Value::BulkString(bytes) => {
                String::from_utf8_lossy(bytes).parse::<f64>().map_err(|e| {
                    VectorStoreError::DatastoreError(format!("Failed to parse score: {e}").into())
                })?
            }
            redis::Value::SimpleString(s) => s.parse::<f64>().map_err(|e| {
                VectorStoreError::DatastoreError(format!("Failed to parse score: {e}").into())
            })?,
            other => {
                return Err(VectorStoreError::DatastoreError(
                    format!("Unexpected Redis value type for score: {other:?}").into(),
                ));
            }
        };
        Ok(1.0 - distance)
    }

    /// Parses FT.SEARCH response into results with deserialized documents.
    fn parse_search_response<T>(
        response: redis::Value,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: for<'a> Deserialize<'a>,
    {
        Self::parse_response_generic(response, true).and_then(|items| {
            items
                .into_iter()
                .map(|(score, id, doc_json)| {
                    let doc = serde_json::from_str::<T>(&doc_json)?;
                    Ok((score, id, doc))
                })
                .collect()
        })
    }

    /// Parses FT.SEARCH response for IDs and scores only.
    fn parse_search_response_ids(
        response: redis::Value,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Self::parse_response_generic(response, false).map(|items| {
            items
                .into_iter()
                .map(|(score, id, _)| (score, id))
                .collect()
        })
    }

    /// Generic response parser that handles both full-document and ID-only modes.
    ///
    /// FT.SEARCH returns: [count, key1, [field1, val1, ...], key2, [field1, val1, ...], ...]
    fn parse_response_generic(
        response: redis::Value,
        include_document: bool,
    ) -> Result<Vec<(f64, String, String)>, VectorStoreError> {
        match response {
            redis::Value::Array(ref items) if !items.is_empty() => {
                let count = match items.first() {
                    Some(redis::Value::Int(n)) => *n as usize,
                    _ => {
                        return Err(VectorStoreError::DatastoreError(
                            "Invalid response format: expected count as first element".into(),
                        ));
                    }
                };

                if count == 0 {
                    return Ok(Vec::new());
                }

                let mut results = Vec::with_capacity(count);

                let mut iter = items.iter().skip(1);
                while let Some(key_val) = iter.next() {
                    let id = match Self::extract_string(key_val) {
                        Some(id) => id,
                        None => {
                            // Skip the fields array too
                            iter.next();
                            continue;
                        }
                    };

                    let fields_val = match iter.next() {
                        Some(redis::Value::Array(fields)) => fields,
                        _ => continue,
                    };

                    let mut score = 0.0;
                    let mut document_json = String::new();

                    let mut field_iter = fields_val.chunks(2);
                    while let Some([name_val, value_val]) = field_iter.next() {
                        let field_name = match Self::extract_string(name_val) {
                            Some(name) => name,
                            None => continue,
                        };

                        if field_name == "__vector_score" {
                            score = Self::extract_score(value_val)?;
                        } else if include_document && field_name == "document" {
                            match Self::extract_string(value_val) {
                                Some(json) => document_json = json,
                                None => {
                                    tracing::warn!(
                                        target: "rig",
                                        id = %id,
                                        "Document field present but could not be extracted as string"
                                    );
                                }
                            }
                        }
                    }

                    results.push((score, id, document_json));
                }

                Ok(results)
            }
            _ => Err(VectorStoreError::DatastoreError(
                "Invalid FT.SEARCH response format".into(),
            )),
        }
    }

    /// Builds and executes a FT.SEARCH KNN query.
    async fn execute_search(
        &self,
        vector_bytes: Vec<u8>,
        req: &VectorSearchRequest<Filter>,
        include_document: bool,
    ) -> Result<redis::Value, VectorStoreError> {
        let mut con = self.connection_manager.clone();

        let filter_str = req
            .filter()
            .as_ref()
            .map(|f| f.clone().into_inner())
            .unwrap_or_else(|| "*".to_string());

        let knn_query = format!(
            "{}=>[KNN {} @{} $vec AS __vector_score]",
            filter_str,
            req.samples(),
            self.vector_field
        );

        let mut cmd = redis::cmd("FT.SEARCH");
        cmd.arg(&self.index_name)
            .arg(&knn_query)
            .arg("PARAMS")
            .arg(2)
            .arg("vec")
            .arg(vector_bytes)
            .arg("SORTBY")
            .arg("__vector_score")
            .arg("RETURN");

        if include_document {
            cmd.arg(2).arg("__vector_score").arg("document");
        } else {
            cmd.arg(1).arg("__vector_score");
        }

        cmd.arg("DIALECT").arg(2);

        // Always specify LIMIT to override RediSearch's default of 10 results.
        cmd.arg("LIMIT").arg(0).arg(req.samples());

        cmd.query_async(&mut con)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
    }

    /// Converts a JSON value to a string suitable for storage in a Redis hash field.
    ///
    /// - Strings are stored as-is (unquoted).
    /// - Numbers and booleans are converted to their string representation.
    /// - Null, arrays, and objects return `None` (not storable as flat hash fields).
    fn json_value_to_hash_field(value: &serde_json::Value) -> Option<String> {
        match value {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            serde_json::Value::Bool(b) => Some(if *b { "1".to_string() } else { "0".to_string() }),
            serde_json::Value::Null
            | serde_json::Value::Array(_)
            | serde_json::Value::Object(_) => None,
        }
    }
}

impl<Model> InsertDocuments for RedisVectorStore<Model>
where
    Model: EmbeddingModel + Send + Sync,
{
    async fn insert_documents<Doc: Serialize + Embed + Send>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let mut con = self.connection_manager.clone();
        let mut pipe = redis::pipe();

        for (document, embeddings) in &documents {
            let json_value = serde_json::to_value(document)?;
            let json_document = json_value.to_string();

            // Extract metadata fields from the document JSON if configured.
            let metadata: Vec<(String, String)> = if self.metadata_fields.is_empty() {
                Vec::new()
            } else {
                self.metadata_fields
                    .iter()
                    .filter_map(|field_name| {
                        let value = json_value.get(field_name)?;
                        match Self::json_value_to_hash_field(value) {
                            Some(hash_value) => Some((field_name.clone(), hash_value)),
                            None => {
                                tracing::warn!(
                                    target: "rig",
                                    field = %field_name,
                                    value_type = %value,
                                    "Metadata field has unsupported type (null/array/object), skipping"
                                );
                                None
                            }
                        }
                    })
                    .collect()
            };

            for embedding in embeddings.iter() {
                let id = if let Some(ref prefix) = self.key_prefix {
                    format!("{}{}", prefix, uuid::Uuid::new_v4())
                } else {
                    uuid::Uuid::new_v4().to_string()
                };
                let embedding_bytes = Self::embedding_to_bytes(&embedding.vec);

                let cmd = pipe
                    .cmd("HSET")
                    .arg(&id)
                    .arg("document")
                    .arg(json_document.as_bytes())
                    .arg("embedded_text")
                    .arg(embedding.document.as_bytes())
                    .arg(&self.vector_field)
                    .arg(embedding_bytes);

                // Write metadata fields as separate hash fields.
                for (field_name, field_value) in &metadata {
                    cmd.arg(field_name).arg(field_value.as_bytes());
                }

                cmd.ignore();
            }
        }

        pipe.query_async::<()>(&mut con)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        tracing::debug!(
            target: "rig",
            index = %self.index_name,
            count = documents.len(),
            metadata_fields = ?self.metadata_fields,
            "Inserted documents into Redis vector store"
        );

        Ok(())
    }
}

impl<M> VectorStoreIndex for RedisVectorStore<M>
where
    M: EmbeddingModel + Send + Sync,
{
    type Filter = Filter;

    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let embedding = self.model.embed_text(req.query()).await?;
        let vector_bytes = Self::embedding_to_bytes(&embedding.vec);

        let response = self.execute_search(vector_bytes, &req, true).await?;
        let mut results = Self::parse_search_response(response)?;

        if let Some(threshold) = req.threshold() {
            results.retain(|(score, _, _)| *score >= threshold);
        }

        tracing::info!(
            target: "rig",
            index = %self.index_name,
            query = %req.query(),
            "Selected documents: {}",
            results.iter().map(|(score, id, _)| format!("{id} ({score:.4})")).collect::<Vec<_>>().join(", ")
        );

        Ok(results)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let embedding = self.model.embed_text(req.query()).await?;
        let vector_bytes = Self::embedding_to_bytes(&embedding.vec);

        let response = self.execute_search(vector_bytes, &req, false).await?;
        let mut results = Self::parse_search_response_ids(response)?;

        if let Some(threshold) = req.threshold() {
            results.retain(|(score, _)| *score >= threshold);
        }

        tracing::info!(
            target: "rig",
            index = %self.index_name,
            query = %req.query(),
            "Selected document IDs: {}",
            results.iter().map(|(score, id)| format!("{id} ({score:.4})")).collect::<Vec<_>>().join(", ")
        );

        Ok(results)
    }
}

impl<M> VectorStoreIndexDyn for RedisVectorStore<M>
where
    M: EmbeddingModel + Sync + Send,
{
    fn top_n<'a>(
        &'a self,
        req: VectorSearchRequest<CoreFilter<serde_json::Value>>,
    ) -> WasmBoxedFuture<'a, TopNResults> {
        Box::pin(async move {
            let req = req.try_map_filter(Filter::try_from)?;
            let results = <Self as VectorStoreIndex>::top_n::<serde_json::Value>(self, req).await?;
            Ok(results)
        })
    }

    fn top_n_ids<'a>(
        &'a self,
        req: VectorSearchRequest<CoreFilter<serde_json::Value>>,
    ) -> WasmBoxedFuture<'a, Result<Vec<(f64, String)>, VectorStoreError>> {
        Box::pin(async move {
            let req = req.try_map_filter(Filter::try_from)?;
            let results = <Self as VectorStoreIndex>::top_n_ids(self, req).await?;
            Ok(results)
        })
    }
}
