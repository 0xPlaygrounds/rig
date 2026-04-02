pub mod filter;

pub use filter::Filter;
use redis::{AsyncCommands, Client};
use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, TopNResults, VectorStoreError, VectorStoreIndex, VectorStoreIndexDyn,
        request::{Filter as CoreFilter, VectorSearchRequest},
    },
    wasm_compat::WasmBoxedFuture,
};
use serde::{Deserialize, Serialize};

/// Redis vector store implementation using RediSearch vector similarity search.
///
/// This implementation uses Redis's FT.SEARCH command with KNN vector queries
/// for similarity search operations.
pub struct RedisVectorStore<M>
where
    M: EmbeddingModel,
{
    /// Model used to generate embeddings for queries
    model: M,
    /// Redis client
    client: Client,
    /// Name of the RediSearch index
    index_name: String,
    /// Name of the vector field in the index
    vector_field: String,
    /// Optional key prefix for document keys
    key_prefix: Option<String>,
}

impl<M> RedisVectorStore<M>
where
    M: EmbeddingModel,
{
    /// Creates a new Redis vector store instance.
    ///
    /// # Arguments
    /// * `model` - Embedding model for query vectorization
    /// * `client` - Redis client instance
    /// * `index_name` - Name of the RediSearch index to query
    /// * `vector_field` - Name of the vector field in the index (default: "embedding")
    pub fn new(model: M, client: Client, index_name: String, vector_field: String) -> Self {
        Self {
            model,
            client,
            index_name,
            vector_field,
            key_prefix: None,
        }
    }

    /// Sets a key prefix for document keys
    pub fn with_key_prefix(mut self, prefix: String) -> Self {
        self.key_prefix = Some(prefix);
        self
    }

    /// Converts embedding vector to bytes for Redis
    fn embedding_to_bytes(embedding: &[f64]) -> Vec<u8> {
        embedding
            .iter()
            .flat_map(|&x| (x as f32).to_le_bytes())
            .collect()
    }

    /// Extracts string value from Redis value
    fn extract_string(value: &redis::Value) -> Option<String> {
        match value {
            redis::Value::BulkString(bytes) => Some(String::from_utf8_lossy(bytes).to_string()),
            redis::Value::SimpleString(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Extracts score from Redis value
    fn extract_score(value: &redis::Value) -> f64 {
        match value {
            redis::Value::BulkString(bytes) => {
                String::from_utf8_lossy(bytes).parse().unwrap_or(0.0)
            }
            redis::Value::SimpleString(s) => s.parse().unwrap_or(0.0),
            _ => 0.0,
        }
    }

    /// Parses FT.SEARCH response into results with documents
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

    /// Parses FT.SEARCH response for IDs only
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

    /// Generic response parser for both full and ID-only results
    fn parse_response_generic(
        response: redis::Value,
        include_document: bool,
    ) -> Result<Vec<(f64, String, String)>, VectorStoreError> {
        match response {
            redis::Value::Array(ref items) if !items.is_empty() => {
                let count = match &items[0] {
                    redis::Value::Int(n) => *n as usize,
                    _ => {
                        return Err(VectorStoreError::DatastoreError(
                            "Invalid response format: expected count as first element".into(),
                        ));
                    }
                };

                if count == 0 {
                    return Ok(Vec::new());
                }

                let mut results = Vec::new();

                for chunk in items[1..].chunks(2) {
                    if chunk.len() != 2 {
                        continue;
                    }

                    let id = match Self::extract_string(&chunk[0]) {
                        Some(id) => id,
                        None => continue,
                    };

                    if let redis::Value::Array(fields) = &chunk[1] {
                        let mut score = 0.0;
                        let mut document_json = String::new();

                        for field_chunk in fields.chunks(2) {
                            if field_chunk.len() != 2 {
                                continue;
                            }

                            let field_name = match Self::extract_string(&field_chunk[0]) {
                                Some(name) => name,
                                None => continue,
                            };

                            if field_name == "__vector_score" {
                                score = Self::extract_score(&field_chunk[1]);
                                if !include_document {
                                    break;
                                }
                            } else if include_document
                                && field_name == "document"
                                && let Some(json) = Self::extract_string(&field_chunk[1])
                            {
                                document_json = json;
                            }
                        }

                        results.push((score, id, document_json));
                    }
                }

                Ok(results)
            }
            _ => Err(VectorStoreError::DatastoreError(
                "Invalid FT.SEARCH response format".into(),
            )),
        }
    }

    /// Builds and executes FT.SEARCH command, optionally including document field
    async fn execute_search(
        &self,
        vector_bytes: Vec<u8>,
        req: &VectorSearchRequest<Filter>,
        include_document: bool,
    ) -> Result<redis::Value, VectorStoreError> {
        let mut con = self
            .client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

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

        if req.threshold().is_some() {
            cmd.arg("LIMIT").arg(0).arg(req.samples());
        }

        cmd.query_async(&mut con)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
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
        let mut con = self
            .client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        for (document, embeddings) in documents {
            let json_document = serde_json::to_string(&document)?;

            for embedding in embeddings.into_iter() {
                let id = if let Some(ref prefix) = self.key_prefix {
                    format!("{}{}", prefix, uuid::Uuid::new_v4())
                } else {
                    uuid::Uuid::new_v4().to_string()
                };
                let embedding_bytes = Self::embedding_to_bytes(&embedding.vec);

                con.hset_multiple::<_, _, _, ()>(
                    &id,
                    &[
                        ("document", json_document.as_bytes()),
                        ("embedded_text", embedding.document.as_bytes()),
                        (&self.vector_field, &embedding_bytes),
                    ],
                )
                .await
                .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
            }
        }

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
